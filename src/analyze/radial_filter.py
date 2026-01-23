import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging
from typing import List, Tuple, Optional
from scipy.stats import bootstrap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Old, may need bugfixing

class RadialProfileFilter:
    """
    First-pass filter for diffraction data.
    Step 1: Remove rows with NaN values in radial bins
    Step 2: Remove rows where specific radial bins are outside ±4σ using MAD
    """
    
    # Radial bins to check for filtering (005, 010, 015, ..., 060)
    FILTER_BINS = [f"radial_bin_{i:03d}" for i in range(60, 121, 20)]
    
    # Constant for normal distribution approximation
    MAD_TO_SIGMA = 1.4826
    
    def __init__(self, analysis_dir: str, n_boot: int = 5000, ci_level: float = 0.9995, random_seed: int = 42):
        """
        Initialize the filter with the analysis directory.
        
        Parameters:
        -----------
        analysis_dir : str
            Path to the analysis directory (e.g., 'results/analysis_{TIMESTAMP}')
        """
        self.n_boot = n_boot
        self.ci_level = ci_level
        self.random_seed = random_seed
        self.thresholds = {}  # Will store (lower, upper) for each bin

        self.analysis_dir = Path(analysis_dir)
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
        
        logger.info(f"Initialized RadialProfileFilter for directory: {analysis_dir}")
    
    def find_xps_directories(self) -> List[Path]:
        """
        Find all XPS paths in the analysis directory.
        
        Returns:
        --------
        List of Path objects for each xps_{VALUE} directory
        """
        results_dir = self.analysis_dir / 'results'
        parquet_files = list(results_dir.rglob("*.parquet"))
        
        logger.info(f"Found {len(parquet_files)} XPS directories")
        return sorted(parquet_files)
    
    def load_parquet_file(self, xps_dir: Path) -> Optional[pd.DataFrame]:
        """
        Load the parquet file from an XPS directory.
        
        Parameters:
        -----------
        xps_dir : Path
            Path to the XPS directory
            
        Returns:
        --------
        pandas DataFrame or None if file not found
        """
        # Try to find the parquet file
        parquet_files = list(xps_dir.glob("xps_*.parquet"))
        
        if not parquet_files:
            logger.warning(f"No parquet file found in {xps_dir}")
            return None
        
        # Use the first parquet file found
        parquet_file = parquet_files[0]
        
        try:
            df = pd.read_parquet(parquet_file)
            logger.info(f"Loaded {len(df)} rows from {parquet_file.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {parquet_file}: {e}")
            return None
    
    def filter_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Remove rows containing NaN values in radial bins.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with radial_bin_* columns
            
        Returns:
        --------
        Filtered dataframe without NaN values in radial bins
        """
        # Identify radial bin columns
        radial_columns = [col for col in df.columns if col.startswith('radial_bin_')]
        
        # Filter to only include columns with numbers greater than 040
        radial_columns_after_040 = [
            col for col in radial_columns 
            if col.replace('radial_bin_', '').isdigit() and int(col.replace('radial_bin_', '')) > 60
        ]

        # Check for NaN only in radial bins after 040
        nan_mask = df[radial_columns_after_040].isna().any(axis=1)
        
        if nan_mask.any():
            df_filtered = df[~nan_mask].copy()
            logger.info(f"Removed {nan_mask.sum()} rows with NaN values in radial bins")
        else:
            df_filtered = df.copy()
            logger.info("No NaN values found in radial bins")
        
        return df_filtered
    
    def calculate_mad_thresholds(self, df: pd.DataFrame) -> dict:
        """
        Calculate MAD-based thresholds for specified radial bins.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe containing radial bin data
            
        Returns:
        --------
        Dictionary with thresholds for each radial bin:
            {
                'radial_bin_005': {'median': value, 'lower': value, 'upper': value},
                ...
            }
        """
        thresholds = {}
        
        for bin_name in self.FILTER_BINS:
            if bin_name not in df.columns:
                logger.warning(f"Column {bin_name} not found in dataframe")
                continue
            
            # Get values for this radial bin
            values = df[bin_name].values
            
            # Calculate median
            median_val = np.median(values)
            
            # Calculate MAD
            deviations = np.abs(values - median_val)
            mad = np.median(deviations)
            
            # Convert MAD to sigma
            sigma = mad * self.MAD_TO_SIGMA
            
            # Calculate thresholds (±4σ)
            lower_threshold = median_val - 4 * sigma
            upper_threshold = median_val + 4 * sigma
            
            thresholds[bin_name] = {
                'median': median_val,
                'mad': mad,
                'sigma': sigma,
                'lower': lower_threshold,
                'upper': upper_threshold
            }
            
            logger.debug(f"{bin_name}: median={median_val:.2f}, "
                        f"mad={mad:.2f}, range=[{lower_threshold:.2f}, {upper_threshold:.2f}]")
        
        return thresholds
    
    def filter_by_mad(self, df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
        """
        Step 2: Remove rows where any specified radial bin is outside ±4σ.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        thresholds : dict
            Dictionary with thresholds for each radial bin
            
        Returns:
        --------
        Filtered dataframe
        """
        if not thresholds:
            logger.warning("No thresholds provided, returning original dataframe")
            return df
        
        # Initialize mask (all True initially)
        valid_mask = pd.Series(True, index=df.index)
        
        # Apply threshold for each radial bin
        for bin_name, thresh in thresholds.items():
            if bin_name not in df.columns:
                continue
            
            bin_mask = (df[bin_name] >= thresh['lower']) & (df[bin_name] <= thresh['upper'])
            valid_mask = valid_mask & bin_mask
            
            # Log how many rows fail this specific bin check
            failed_count = (~bin_mask).sum()
            if failed_count > 0:
                logger.debug(f"{bin_name}: {failed_count} rows outside ±4σ range")
        
        # Count total rows removed
        removed_count = (~valid_mask).sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows based on MAD filtering")
        else:
            logger.info("All rows passed MAD filtering")
        
        return df[valid_mask].copy()
    
    def calculate_bootstrap_thresholds(self, df: pd.DataFrame) -> dict:
        """
        Compute bootstrap 95% CI bounds for each radial bin.
        Values outside these bounds will be considered outliers.
        """
        thresholds = {}
        rng = np.random.default_rng(self.random_seed)

        for bin_name in self.FILTER_BINS:
            if bin_name not in df.columns:
                logger.warning(f"Column {bin_name} not found in dataframe")
                continue

            values = df[bin_name].dropna().values  # Bootstrap only on available data
            if len(values) < 10:
                logger.warning(f"{bin_name} has <10 valid points. Skipping bootstrap.")
                continue

            # Bootstrap the mean to get its sampling distribution
            # We use the percentile method (simple and robust)
            boot = bootstrap(
                (values,),
                statistic=np.median,
                n_resamples=self.n_boot,
                confidence_level=self.ci_level,
                method='percentile',
                random_state=rng
            )

            ci_low, ci_high = boot.confidence_interval
            median_val = np.median(values)
            bootstrap_sem = boot.standard_error

            thresholds[bin_name] = {
                'median': median_val,
                'bootstrap_sem': bootstrap_sem,
                'lower': ci_low,
                'upper': ci_high,
                'n_samples': len(values)
            }

            logger.debug(f"{bin_name}: n={len(values)}, "
                        f"median={median_val:.3f}, "
                        f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]")

        self.thresholds = thresholds
        return thresholds

    def filter_by_bootstrap_ci(self, df: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
        """
        Remove rows where any radial bin value falls outside its bootstrap 95% CI.
        """
        if thresholds is None:
            if not self.thresholds:
                raise ValueError("No thresholds computed. Run calculate_bootstrap_thresholds() first.")
            thresholds = self.thresholds

        if not thresholds:
            logger.warning("No valid thresholds. Returning original dataframe.")
            return df.copy()

        valid_mask = pd.Series(True, index=df.index)

        for bin_name, thresh in thresholds.items():
            if bin_name not in df.columns:
                continue

            col = df[bin_name]
            bin_mask = (col >= thresh['lower']) & (col <= thresh['upper'])
            valid_mask &= bin_mask

            failed = (~bin_mask & col.notna()).sum()
            if failed > 0:
                logger.debug(f"{bin_name}: {failed} values outside bootstrap 95% CI")

        removed = (~valid_mask).sum()
        logger.info(f"Bootstrap CI filtering removed {removed} rows "
                   f"({removed/len(df)*100:.2f}% of data)")

        return df[valid_mask].copy()

    def process_xps_directory(self, xps_dir: Path, filter_type: str) -> Tuple[bool, str]:
        """
        Process a single XPS directory: load, filter, and save results.
        
        Parameters:
        -----------
        xps_dir : Path
            Path to the XPS directory
            
        Returns:
        --------
        (success: bool, message: str)
        """
        try:
            # Load data
            try:
                df = pd.read_parquet(xps_dir)
                logger.info(f"Loaded {len(df)} rows from {xps_dir.name}")
            except Exception as e:
                logger.error(f"Error loading {xps_dir}: {e}")
            
            initial_count = len(df)
            
            # Step 1: Filter NaN values
            df_filtered = self.filter_nan_values(df)
            after_nan_count = len(df_filtered)
            
            if after_nan_count == 0:
                logger.warning(f"No data remaining after NaN filtering in {xps_dir.name}")
                return False, "No data after NaN filtering"
            
            if filter_type == "MAD":
                # Step 2: Calculate MAD thresholds
                thresholds = self.calculate_mad_thresholds(df_filtered)
                
                # Step 3: Apply MAD filtering
                df_final = self.filter_by_mad(df_filtered, thresholds)
                final_count = len(df_final)
            elif filter_type == "bootstrap":
                # Step 2: Calculate bootstrap thresholds
                thresholds = self.calculate_bootstrap_thresholds(df_filtered)
                
                # Step 3: Apply bootstrap filtering
                df_final = self.filter_by_bootstrap_ci(df_filtered, thresholds)
                final_count = len(df_final)
            else:
                print("unsupported method, the method should be -MAD- or -bootstrap-!")
            
            if final_count == 0:
                logger.warning(f"No data remaining after MAD filtering in {xps_dir.name}")
                return False, "No data after MAD filtering"
            
            # Save filtered data
            output_filename = f"filtered_{xps_dir.name}"
            output_path = xps_dir.parent.parent / "filtered" /output_filename
            
            df_final.to_parquet(output_path, index=False)
            
            # Log summary
            logger.info(f"Processed {xps_dir.name}: "
                       f"{initial_count} → {after_nan_count} → {final_count} rows "
                       f"({final_count/initial_count*100:.1f}% retained)")
            
            return True, f"Successfully saved {output_filename} with {final_count} rows"
            
        except Exception as e:
            logger.error(f"Error processing {xps_dir.name}: {e}")
            return False, str(e)
    
    def run_filtering(self, filter_type: str) -> dict:
        """
        Run the filtering process on all XPS directories.
        
        Parameters:
        -----------
        filter_type : str
            MAD for mean absolute deviation   (4σ, 99.9%)
            bootstrap for bootstrap filtering (95%CI)
            
        Returns:
        --------
        Dictionary with processing results for each XPS directory
        """
        results = {}
        
        # Find all XPS directories
        xps_dirs = self.find_xps_directories()
        
        if not xps_dirs:
            logger.error("No XPS directories found")
            return results
        
        # Process each XPS directory
        for xps_dir in xps_dirs:
            logger.info(f"Processing {xps_dir.name}...")
            success, message = self.process_xps_directory(xps_dir, filter_type)
            results[xps_dir.name] = {
                'success': success,
                'message': message,
                'directory': str(xps_dir)
            }
        
        # Print summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(f"Processing complete: {successful}/{len(results)} directories successful")
        
        return results


def main():
    """
    Example usage of the RadialProfileFilter class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter diffraction data using NaN removal and MAD-based filtering')
    parser.add_argument('analysis_dir', type=str, help='Path to analysis directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize and run the filter
    filter_processor = RadialProfileFilter(args.analysis_dir)
    results = filter_processor.run_filtering()
    
    # Print detailed results
    print("\n" + "="*60)
    print("FILTERING RESULTS SUMMARY")
    print("="*60)
    for xps_dir, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{xps_dir}: {status}")
        print(f"  Message: {result['message']}")
        print()


if __name__ == "__main__":
    main()