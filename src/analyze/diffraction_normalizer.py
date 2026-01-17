import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
import csv
import statistics
from scipy.stats import bootstrap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Old, need to change file directory

class DiffractionNormalizer:
    """
    Second-pass normalizer for diffraction data.
    Processes filtered_xps_{}.parquet files from the first filtering step.
    """
    
    # Radial bins to use for normalization factor calculation
    NORMALIZATION_BINS = [f"radial_bin_{i:03d}" for i in range(60, 201, 20)]  # 60, 80, ..., 200
    AVG_TUNING_BINS = [f"radial_bin_{i:03d}" for i in range(330, 411, 20)]  # q8~10 at 0.024 A^-1/pixel
    
    def __init__(self, analysis_dir: str):
        """
        Initialize the normalizer with the analysis directory.
        
        Parameters:
        -----------
        analysis_dir : str
            Path to the analysis directory (e.g., 'results/analysis_{TIMESTAMP}')
        """
        self.analysis_dir = Path(analysis_dir)
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
        
        logger.info(f"Initialized DiffractionNormalizer for directory: {analysis_dir}")
    
    def find_filtered_parquet_files(self) -> List[Path]:
        """
        Find all filtered parquet files in the analysis directory.
        
        Returns:
        --------
        List of Path objects for each filtered_xps_{VALUE}.parquet file
        """
        filtered_files = list(self.analysis_dir.glob("**/filtered_xps_*.parquet"))
        
        logger.info(f"Found {len(filtered_files)} filtered parquet files")
        return sorted(filtered_files)
    
    def load_filtered_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load filtered parquet file.
        
        Parameters:
        -----------
        file_path : Path
            Path to the filtered parquet file
            
        Returns:
        --------
        pandas DataFrame or None if file cannot be loaded
        """
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            
            # Verify required columns exist
            required_columns = ['filename'] + self.NORMALIZATION_BINS
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in {file_path.name}: {missing_columns}")
                return None
            
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def calculate_normalization_factors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Calculate normalization factor for each row.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with radial bin columns
            
        Returns:
        --------
        (df_with_factors, avg_norm_factor)
        df_with_factors: DataFrame with added 'norm_factor' column
        avg_norm_factor: Average normalization factor
        """
        # Sum the specified radial bins for each row
        norm_factors = df[self.NORMALIZATION_BINS].sum(axis=1)
        
        # Add normalization factor column
        df_with_factors = df.copy()
        df_with_factors['norm_factor'] = norm_factors
        
        # Calculate average normalization factor
        avg_norm_factor = statistics.mean(norm_factors)
        
        logger.info(f"Calculated normalization factors: avg={avg_norm_factor:.5f}, "
                   f"min={norm_factors.min():.5f}, max={norm_factors.max():.5f}")
        
        return df_with_factors, avg_norm_factor
    
    def calculate_norm_factor_thresholds(self, norm_factors: pd.Series) -> Dict[str, float]:
        """
        Calculate MAD-based thresholds for normalization factors.
        
        Parameters:
        -----------
        norm_factors : pd.Series
            Series of normalization factors
            
        Returns:
        --------
        Dictionary with median, MAD, sigma, and thresholds
        """
        # Calculate median
        median_val = np.median(norm_factors)
        
        # Calculate MAD
        deviations = np.abs(norm_factors - median_val)
        mad = np.median(deviations)
        
        # Convert MAD to sigma (1.4826 for normal distribution)
        sigma = mad * 1.4826
        
        # Calculate ±2σ thresholds
        lower_threshold = median_val - 2 * sigma
        upper_threshold = median_val + 2 * sigma
        
        thresholds = {
            'median': median_val,
            'mad': mad,
            'sigma': sigma,
            'lower': lower_threshold,
            'upper': upper_threshold
        }
        
        logger.info(f"Norm factor thresholds: median={median_val:.2f}, "
                f"±2σ range=[{lower_threshold:.2f}, {upper_threshold:.2f}]")
        
        return thresholds
    
    def filter_by_norm_factor_mad(self, df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
        """
        Filter rows based on normalization factor MAD thresholds.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'norm_factor' column
        thresholds : dict
            Dictionary with threshold values
            
        Returns:
        --------
        Filtered DataFrame
        """
        # Create mask for rows within ±2σ
        mask = (df['norm_factor'] >= thresholds['lower']) & \
            (df['norm_factor'] <= thresholds['upper'])
        
        removed_count = (~mask).sum()
        retained_count = mask.sum()
        
        if removed_count > 0:
            logger.info(f"Norm factor MAD filtering: removed {removed_count} rows, "
                    f"retained {retained_count} rows ({retained_count/len(df)*100:.1f}%)")
            
            # Log some statistics about removed values
            removed_factors = df.loc[~mask, 'norm_factor']
            logger.debug(f"Removed norm factors range: [{removed_factors.min():.2f}, "
                        f"{removed_factors.max():.2f}]")
        else:
            logger.info("All rows passed norm factor MAD filtering")
        
        return df[mask].copy()

    def normalize_radial_profiles(self, df: pd.DataFrame, avg_norm_factor: float) -> pd.DataFrame:
        """
        Normalize radial profiles by scaling each row.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'norm_factor' column and radial_bin_* columns
        avg_norm_factor : float
            Average normalization factor to scale to
            
        Returns:
        --------
        DataFrame with normalized radial bins
        """
        # Create a copy to avoid modifying the original
        df_normalized = df.copy()
        
        # Identify radial bin columns (excluding metadata columns)
        radial_columns = [col for col in df.columns if col.startswith('radial_bin_')]
        
        # Calculate scaling factor for each row
        scaling_factors = avg_norm_factor / df['norm_factor']
        #scaling_factors = 100 / df['norm_factor']

        # Apply scaling to all radial bins
        for col in radial_columns:
            df_normalized[col] = df[col] * scaling_factors
        
        logger.info(f"Normalized radial profiles using scaling factors: "
                   f"avg_scale={scaling_factors.mean():.5f}")
        
        return df_normalized
    
    def save_normalization_factors(self, df: pd.DataFrame, xps_dir: Path, xps_value: float):
        """
        Save normalization factors to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'filename' and 'norm_factor' columns
        xps_dir : Path
            Directory to save the CSV file in
        xps_value : float
            XPS value for this group
        """
        # Create CSV filename
        csv_filename = f"norm_factors_xps{xps_value:.5f}.csv"
        csv_path = xps_dir / csv_filename
        
        # Select only filename and norm_factor columns
        factor_df = df[['filename', 'norm_factor']].copy()
        
        # Sort by filename for consistency
        factor_df = factor_df.sort_values('filename')
        
        # Save to CSV
        factor_df.to_csv(csv_path, index=False, float_format='%.5f')
        
        logger.info(f"Saved normalization factors to {csv_filename} "
                   f"({len(factor_df)} entries)")
    
    def calculate_intensity_profile_stats(self, df_normalized: pd.DataFrame, 
                                         avg_norm_factor: float,
                                         xps_dir: Path, xps_value: float):
        """
        Calculate average and std for each radial bin and save to CSV.
        
        Parameters:
        -----------
        df_normalized : pd.DataFrame
            Normalized DataFrame with radial_bin_* columns
        avg_norm_factor : float
            Average normalization factor
        xps_dir : Path
            Directory to save the CSV file in
        xps_value : float
            XPS value for this group
        """
        # Identify radial bin columns
        radial_columns = [col for col in df_normalized.columns if col.startswith('radial_bin_')]
        
        # Sort columns to ensure consistent bin order
        radial_columns.sort()
        
        # Calculate statistics
        averages = []
        stds = []
        sems = []
        bin_numbers = []
        
        for col in radial_columns:
            # Extract bin number from column name
            bin_num = int(col.replace('radial_bin_', ''))
            
            # Calculate average and std for this bin
            avg_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            sem_val = std_val / np.sqrt(len(df_normalized[col]))
            
            averages.append(avg_val)
            stds.append(std_val)
            sems.append(sem_val)
            bin_numbers.append(bin_num)
        
        # Create DataFrame for statistics
        stats_df = pd.DataFrame({
            'bin_number': bin_numbers,
            'average': averages,
            'std': stds,
            'sem': sems
        })
        
        # Create filename with intensity profile
        profile_filename = f"intensity_profile_xps{xps_value:.5f}_I{avg_norm_factor:.5f}.csv"
        profile_path = xps_dir / profile_filename
        
        # Save to CSV
        stats_df.to_csv(profile_path, index=False, float_format='%.5f')
        
        logger.info(f"Saved intensity profile statistics to {profile_filename} "
                   f"({len(stats_df)} bins)")
        
        return stats_df
    
    def calculate_intensity_bootstrap_stats(self, df_normalized: pd.DataFrame, 
                                        avg_norm_factor: float,
                                        xps_dir: Path, xps_value: float,
                                        n_bootstrap: int = 2000):
        """
        Calculate bootstrap statistics for each radial bin and save to CSV.
        
        Parameters:
        -----------
        df_normalized : pd.DataFrame
            Normalized DataFrame with radial_bin_* columns
        avg_norm_factor : float
            Average normalization factor
        xps_dir : Path
            Directory to save the CSV file in
        xps_value : float
            XPS value for this group
        n_bootstrap : int
            Number of bootstrap samples
        """
        # Identify radial bin columns
        radial_columns = [col for col in df_normalized.columns if col.startswith('radial_bin_')]
        
        # Sort columns to ensure consistent bin order
        radial_columns.sort()
        
        # Initialize lists for results
        averages = []
        stds = []
        sems = []  # Standard Error of the Mean
        ci_lowers = []
        ci_uppers = []
        bin_numbers = []
        
        logger.info(f"Calculating bootstrap statistics for {len(radial_columns)} bins "
                f"(n_bootstrap={n_bootstrap})...")
        
        for i, col in enumerate(radial_columns):
            # Extract bin number from column name
            bin_num = int(col.replace('radial_bin_', ''))
            
            # Get data for this bin
            data = df_normalized[col].values
            
            # Calculate basic statistics
            avg_val = np.mean(data)
            std_val = np.std(data)
            sem_val = std_val / np.sqrt(len(data))
            
            # Calculate bootstrap 95% CI for the mean
            try:
                bootstrap_result = bootstrap(
                    (data,), 
                    np.mean, 
                    n_resamples=n_bootstrap,
                    confidence_level=0.95,
                    method='percentile'
                )
                ci_lower = bootstrap_result.confidence_interval.low
                ci_upper = bootstrap_result.confidence_interval.high
            except Exception as e:
                logger.warning(f"Bootstrap failed for bin {bin_num}: {e}")
                # Use parametric approximation if bootstrap fails
                ci_lower = avg_val - 1.96 * sem_val
                ci_upper = avg_val + 1.96 * sem_val

            # Store results
            averages.append(avg_val)
            stds.append(std_val)
            sems.append(sem_val)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            bin_numbers.append(bin_num)
            
            # Log progress for every 100 bins
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{len(radial_columns)} bins")
        
        # Create DataFrame for statistics
        stats_df = pd.DataFrame({
            'bin_number': bin_numbers,
            'average': averages,
            'std': stds,
            'sem': sems,
            'lower_bound': ci_lowers,
            'upper_bound': ci_uppers
        })
        
        # Create filename with intensity profile
        profile_filename = f"intensity_profile_xps{xps_value:.5f}_I{avg_norm_factor:.5f}.csv"
        profile_path = xps_dir / profile_filename
        
        # Save to CSV
        stats_df.to_csv(profile_path, index=False, float_format='%.5f')
        
        logger.info(f"Saved bootstrap intensity profile to {profile_filename} "
                f"({len(stats_df)} bins)")
        
        return stats_df

    def process_xps_group(self, filtered_file: Path, statistic_type: str) -> Tuple[bool, str]:
        """
        Process a single XPS group: normalize and calculate statistics.
        
        Parameters:
        -----------
        filtered_file : Path
            Path to the filtered parquet file
            
        Returns:
        --------
        (success: bool, message: str)
        """
        try:
            # Extract XPS value from filename
            filename = filtered_file.stem  # 'filtered_xps_188.94000'
            xps_value_str = filename.replace('filtered_xps_', '')
            xps_value = float(xps_value_str)
            
            logger.info(f"Processing XPS group {xps_value:.5f} from {filtered_file.name}")
            
            # Load filtered data
            df = self.load_filtered_data(filtered_file)
            if df is None:
                return False, f"Failed to load data from {filtered_file.name}"
            
            # Step 1: Calculate normalization factors
            df_with_factors, avg_norm_factor = self.calculate_normalization_factors(df)
            
            # Step 2: Filter based on norm factor MAD (±2σ)
            norm_factor_thresholds = self.calculate_norm_factor_thresholds(df_with_factors['norm_factor'])
            df_filtered = self.filter_by_norm_factor_mad(df_with_factors, norm_factor_thresholds)
            # Update average norm factor based on filtered data
            # Sum the specified radial bins for each row
            norm_factors = df_filtered[self.NORMALIZATION_BINS].sum(axis=1)
            # Calculate average normalization factor
            avg_norm_factors = df_filtered[self.AVG_TUNING_BINS].sum(axis=1)
            #avg_norm_factor_filtered = 100 / statistics.mean(avg_norm_factors/norm_factors)
            avg_norm_factor_filtered = 100

            # Step 3: Save normalization factors to CSV
            self.save_normalization_factors(df_filtered, filtered_file.parent, xps_value)
            
            # Step 4: Normalize radial profiles
            df_normalized = self.normalize_radial_profiles(df_filtered, avg_norm_factor_filtered)
            
            # Step 5: Calculate intensity profile statistics and save to CSV
            if statistic_type == 'normal':
                stats_df = self.calculate_intensity_profile_stats(
                    df_normalized, avg_norm_factor_filtered, filtered_file.parent, xps_value
                )
            elif statistic_type == 'bootstrap':
                stats_df = self.calculate_intensity_bootstrap_stats(
                    df_normalized, avg_norm_factor_filtered, filtered_file.parent, xps_value
                )
            
            # Optional: Save normalized data to new parquet file
            normalized_filename = f"normalized_xps_{xps_value:.5f}.parquet"
            normalized_path = filtered_file.parent / normalized_filename
            df_normalized.to_parquet(normalized_path, index=False)
            logger.info(f"Saved normalized data to {normalized_filename}")
            
            # Log summary
            logger.info(f"Successfully processed XPS {xps_value:.5f}: "
                       f"{len(df)} files, avg_norm_factor={avg_norm_factor:.5f}")
            
            return True, f"Processed {len(df)} files, avg_norm_factor={avg_norm_factor:.5f}"
            
        except Exception as e:
            logger.error(f"Error processing {filtered_file.name}: {e}")
            return False, str(e)
    
    def run_normalization(self, statistic_type: str) -> dict:
        """
        Run the normalization process on all filtered XPS files.
        
        Parameters:
        -----------
        statistic_type : str
            "normal" for avg, std, sem
            "bootstrap" for n_bootsrap 2000 with 95 CI concerning avg, lower/upper_bounds and traditional statistic

        Returns:
        --------
        Dictionary with processing results for each XPS group
        """
        results = {}
        
        # Find all filtered parquet files
        filtered_files = self.find_filtered_parquet_files()
        
        if not filtered_files:
            logger.error("No filtered parquet files found. Run the first filter step first.")
            return results
        
        # Process each XPS group
        for filtered_file in filtered_files:
            logger.info(f"Processing {filtered_file.name}...")
            success, message = self.process_xps_group(filtered_file, statistic_type)
            results[filtered_file.name] = {
                'success': success,
                'message': message,
                'file_path': str(filtered_file)
            }
        
        # Print summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(f"Normalization complete: {successful}/{len(results)} groups successful")
        
        return results
    
    def generate_summary_report(self, results: dict, output_dir: Optional[Path] = None):
        """
        Generate a summary report of all processed XPS groups.
        
        Parameters:
        -----------
        results : dict
            Results from run_normalization()
        output_dir : Optional[Path]
            Directory to save summary report (default: analysis_dir)
        """
        if output_dir is None:
            output_dir = self.analysis_dir
        
        summary_data = []
        
        for filename, result in results.items():
            if result['success']:
                # Extract XPS value from filename
                try:
                    xps_value = float(filename.replace('filtered_xps_', '').replace('.parquet', ''))
                    
                    # Find the intensity profile file
                    xps_dir = Path(result['file_path']).parent
                    profile_files = list(xps_dir.glob(f"intensity_profile_xps{xps_value:.5f}_I*.csv"))
                    
                    if profile_files:
                        # Extract average norm factor from filename
                        profile_filename = profile_files[0].name
                        # Find pattern _I{value}.csv
                        import re
                        match = re.search(r'_I([\d\.]+)\.csv', profile_filename)
                        avg_norm_factor = float(match.group(1)) if match else None
                        
                        # Load stats to get number of bins
                        stats_df = pd.read_csv(profile_files[0])
                        num_bins = len(stats_df)
                        
                        summary_data.append({
                            'xps_value': xps_value,
                            'avg_norm_factor': avg_norm_factor,
                            'num_files': 'N/A',  # Could load from norm_factors CSV
                            'num_bins': num_bins,
                            'status': 'Success'
                        })
                except Exception as e:
                    logger.warning(f"Could not extract details for {filename}: {e}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('xps_value')
            
            # Save summary to CSV
            summary_path = output_dir / "normalization_summary.csv"
            summary_df.to_csv(summary_path, index=False, float_format='%.5f')
            
            logger.info(f"Saved normalization summary to {summary_path}")
            
            # Print summary table
            print("\n" + "="*80)
            print("NORMALIZATION SUMMARY")
            print("="*80)
            print(summary_df.to_string(index=False))
            print()
            
            return summary_df
        else:
            logger.warning("No successful results to summarize")
            return None


def main():
    """
    Example usage of the DiffractionNormalizer class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize diffraction data and calculate intensity profiles')
    parser.add_argument('analysis_dir', type=str, help='Path to analysis directory')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize and run the normalizer
    normalizer = DiffractionNormalizer(args.analysis_dir)
    results = normalizer.run_normalization()
    
    # Generate summary if requested
    if args.summary:
        normalizer.generate_summary_report(results)
    
    # Print processing results
    print("\n" + "="*60)
    print("NORMALIZATION RESULTS")
    print("="*60)
    for filename, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{filename}: {status}")
        print(f"  Message: {result['message']}")
        print()


if __name__ == "__main__":
    main()