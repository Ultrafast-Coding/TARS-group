import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import logging
import random
import yaml

from src.processor.tiff_objects import EMCCDimage
from src.io.tiff_SL import load_tiff_single
from src.utils.timer import timer
from src.plots.quick_plot import plot_azimuthal_average, plot_ndarray
from src.utils.processing_utils import ProcessedResult
from src.utils.xps_value_sort import extract_xps_value

class XPSGroupProcessor:
    def __init__(self, 
                 background_data: np.ndarray,
                 data_mask_data: List[np.ndarray],
                 X_ray_config: List,
                 center_config: List,
                 azimuthal_config: List,
                 xps_value: float,
                 filelist: List[str],
                 resultdir: str):
        """
        Initialize XPS Group Processor for batch processing of images.
        
        Args:
            background_data: Background data for subtraction
            data_mask_data: Data mask for masking area of background and valid signal
            X_ray_config: [sigma_threshold, expansion_threshold_ratio]
            center_config: [ring_mask, initial_guess]
            azimuthal_config: [radial_masks, azimuthal_mask_dict]
            xps_value: XPS value for this group
            filelist: List of file paths to process
            resultdir: Directory to save results
        """
        self.background_data = background_data
        self.data_mask_data = data_mask_data
        self.X_ray_config = X_ray_config  # [sigma_threshold, expansion_threshold_ratio]
        self.center_config = center_config  # [ring_mask, initial_guess]
        self.azimuthal_config = azimuthal_config  # [radial_masks, azimuthal_mask_dict]
        self.xps_value = xps_value
        self.filelist = filelist
        self.resultdir = Path(resultdir)
        
        #Create list for storing median and mad ndarray to sort Xray spots
        self.X_ray_precompute = None
        
        # Create result directory if it doesn't exist
        self.resultdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.processed_files = []
        self.failed_files = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def precompute_xray_statistics(self, center_region_size: int = 40, sample_size: int = 50) -> None:
        """
        Precompute median and MAD arrays for X-ray removal using random sampling.
        
        Args:
            center_region_size: Size of center region to ignore (square around center)
            sample_size: Number of random images to use for statistics
        """
        if not self.filelist:
            raise ValueError("File list is empty")
        
        # Determine how many files to sample
        if len(self.filelist) <= sample_size:
            sample_files = self.filelist
            self.logger.info(f"Using all {len(sample_files)} files for X-ray statistics precomputation")
        else:
            # Randomly sample files
            sample_files = random.sample(self.filelist, sample_size)
            self.logger.info(f"Randomly sampled {len(sample_files)} files for X-ray statistics precomputation")
        
        all_processed_images = []
        
        # Load and process sample images
        for i, filepath in enumerate(sample_files):
            try:
                #self.logger.info(f"Processing sample {i+1}/{len(sample_files)}: {Path(filepath).name}")
                
                # Load image
                image = EMCCDimage(load_tiff_single(filepath))
                
                # Remove only background, no Xray filter 
                image.remove_background_legacy(
                    self.background_data
                )
                
                # Store the processed data
                all_processed_images.append(image.processed_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to process sample {filepath}: {str(e)}")
                continue
        
        if not all_processed_images:
            raise ValueError("No images were successfully processed for X-ray statistics")
        
        # Stack images and compute statistics
        image_stack = np.stack(all_processed_images, axis=0)
        
        # Compute median across the sample
        median_array = np.median(image_stack, axis=0)
        
        # Compute MAD (Median Absolute Deviation)
        abs_deviation = np.abs(image_stack - median_array)
        mad_array = np.median(abs_deviation, axis=0)
        
        # Get center position from center_config
        center_x, center_y = self.center_config[1]  # initial_guess tuple
        
        # Set center region to NaN to ignore bright diffraction center
        height, width = median_array.shape
        half_size = center_region_size // 2
        
        row_start = int(round(center_y - half_size))
        row_end = int(round(center_y + half_size))
        col_start = int(round(center_x - half_size))
        col_end = int(round(center_x + half_size))
        
        # Ensure indices are within bounds
        row_start = max(0, row_start)
        row_end = min(height, row_end)
        col_start = max(0, col_start)
        col_end = min(width, col_end)
        
        # Set center region to NaN
        median_array[row_start:row_end, col_start:col_end] = np.nan
        mad_array[row_start:row_end, col_start:col_end] = np.nan
        
        # Store precomputed statistics
        self.X_ray_precompute = [median_array, mad_array]
        
        self.logger.info(f"X-ray statistics precomputation completed. Center region ({center_region_size}x{center_region_size}) excluded.")

    def process_single(self, filepath: str, expanded_size: int = 1024) -> Optional['ProcessedResult']:
        """
        Process a single image file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            ProcessedResult object if successful, None if failed
        """
        try:
        #self.logger.info(f"Processing: {Path(filepath).name}")
        
            # Load image file
            image_file = EMCCDimage(load_tiff_single(filepath))
            
            # Remove background
            # with timer('bkg_removal'):
            image_file.remove_background(
                self.background_data,
                self.X_ray_precompute[0],  # median_array
                self.X_ray_precompute[1],  # mad_array
                self.X_ray_config[0],  # sigma_threshold
                self.X_ray_config[1]   # expansion_threshold_ratio
            )
            
            # Masking valid signal
            image_file.apply_data_mask(self.data_mask_data, expanded_size)
            
            # Find diffraction center
            # with timer('center_finding'):
            # center = image_file.iterative_ring_centroid(
            #     self.center_config[0],  # ring_mask
            #     self.center_config[1]   # initial_guess
            # )
            
            # with timer("center_finding"):
            center = image_file.find_center_with_std(
                radial_masks=self.center_config[0],  # ring_mask
                initial_guess=self.center_config[1]   # initial_guess
            )
            # Convert the center pos back
            pad_size = int(expanded_size / 2 - 1024 / 2)
            center = (center[0]-pad_size, center[1]-pad_size)
            
            # Calculate azimuthal average
            # with timer('azimuthal_avg'):
            bin_centers, radial_average = image_file.azimuthal_average_bincount(
                self.azimuthal_config[0],  # radial_masks
                self.azimuthal_config[1]   # azimuthal_mask dict
            )
            
            # Create result object
            result = ProcessedResult(
                filename=Path(filepath).name,
                center=center,
                radial_profile=radial_average,
                xps_value=extract_xps_value(Path(filepath).name)
            )
            
            #self.logger.info(f"Successfully processed: {Path(filepath).name}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to process {filepath}: {str(e)}")
            self.failed_files.append((filepath, str(e)))
            return None

    def process_single_debug(self, filepath: str, 
                             plot_min_raw: float = None, plot_max_raw: float = None,
                             plot_min: float = None, plot_max: float = None,
                             expanded_size: int = 1024) -> None:
        """
        Debug version: Process a single image file with extensive plotting and printing.
        
        Args:
            filepath: Path to the image file
            plot_min: Minimum value for plot display (optional)
            plot_max: Maximum value for plot display (optional)
        """
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG PROCESSING: {Path(filepath).name}")
            print(f"Under:{Path(filepath).parent}")
            print(f"{'='*60}")
            
            # Load image file
            image_file = EMCCDimage(load_tiff_single(filepath))
            
            # Plot original data before background removal
            print(f"\n1. ORIGINAL DATA:")
            plot_ndarray(image_file.raw_data, plot_min_raw, plot_max_raw)

            # Remove background
            print(f"\n2. BACKGROUND REMOVAL:")
            image_file.remove_background(
                self.background_data,
                self.X_ray_precompute[0],  # median_array
                self.X_ray_precompute[1],  # mad_array
                self.X_ray_config[0],  # sigma_threshold
                self.X_ray_config[1]   # expansion_threshold_ratio
            )
            
            # Plot after background removal
            print(f"After background removal:")
            print(f"NaN count: {np.sum(np.isnan(image_file.processed_data))}")
            plot_ndarray(image_file.processed_data, plot_min, plot_max)

            # Masking valid signal
            print(f"\n3. MASKING DATA:")
            image_file.apply_data_mask(self.data_mask_data, expanded_size)

            # Plot after masking
            print(f"After data masking:")
            plot_ndarray(image_file.processed_data, plot_min, plot_max)

            # Find diffraction center
            print(f"\n4. CENTER FINDING:")
            center = image_file.find_center_with_std(
                    radial_masks=self.center_config[0],  # ring_mask
                    initial_guess=self.center_config[1]   # initial_guess
                )
            # Convert the center pos back
            pad_size = int(expanded_size / 2 - 1024 / 2)
            center = (center[0]-pad_size, center[1]-pad_size)
            print(f"Found center: ({center[0]:.2f}, {center[1]:.2f})")

            # Calculate azimuthal average
            print(f"\n5. AZIMUTHAL AVERAGE:")
            bin_centers, radial_average = image_file.azimuthal_average_bincount(
                self.azimuthal_config[0],  # radial_masks
                self.azimuthal_config[1]   # azimuthal_mask dict
            )
            
            # Plot radial average
            print(f"Radial profile range: [{np.nanmin(radial_average):.3f}, {np.nanmax(radial_average):.3f}]")
            print(f"Non-NaN bins: {np.sum(~np.isnan(radial_average))}/{len(radial_average)}")
            plot_azimuthal_average(bin_centers, radial_average)
            
            print(f"\n✓ SUCCESSFULLY PROCESSED: {Path(filepath).name}")
            print(f"{'='*60}")

        except Exception as e:
            print(f"\n✗ FAILED TO PROCESS {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.failed_files.append((filepath, str(e)))

    def process_group(self, batch_size: int = 100) -> None:
        """
        Process all files in the filelist.
        
        Args:
            batch_size: Number of files to process before printing progress and saving
        """
        processed_filepath = self.resultdir / f"processed_{self.xps_value:.5f}.yaml"
        # if the .yaml already exists, resume off from it
        if processed_filepath.exists():
            # Load existing processed files
            with open(processed_filepath, 'r') as f:
                already_processed = yaml.safe_load(f) or []
            
            # Exclude already processed files from current filelist
            self.filelist = [f for f in self.filelist if f not in already_processed]
            
            # log how many files were excluded
            excluded_count = len(already_processed)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} already processed files for group {self.xps_value:.5f}")

        total_files = len(self.filelist)
        if total_files != 0:
            self.logger.info(f"Starting batch processing of {total_files} files")
        else: 
            self.logger.info(f"Skipping XPS group {self.xps_value:.5f}, no new files!")
            return

        start_time = time.time()
        batch_results = []
        
        # Precompute X-ray statistics once for the entire group
        self.precompute_xray_statistics()

        # Exract the expanded_size from mask used
        expanded_size = self.center_config[0].image_shape[0]

        for batch_num, i in enumerate(range(0, total_files, batch_size)):
            batch_files = self.filelist[i:i + batch_size]
            batch_results = []
            
            for filepath in batch_files:
                result = self.process_single(filepath, expanded_size)
                if result is not None:
                    batch_results.append(result)
                    self.processed_files += [filepath]
            
            # Save batch to the same Parquet file
            if batch_results:
                # updates the processed filepaths to .yaml
                self.update_filelists(self.processed_files)
                self.save_results(batch_results, batch_number=batch_num)
            
            batch_time = time.time() - start_time
            self.logger.info(f"Completed batch {batch_num} in {batch_time:.2f} seconds."
                             f"{len(batch_results)}/{len(batch_files)} successful")
            # Reset timer for next batch
            start_time = time.time()
        
        # Final summary
        self.logger.info(f"Processing completed. "
                       f"Successfully processed: {len(self.processed_files)}/{total_files} "
                       f"({len(self.processed_files)/total_files*100:.1f}%)")
        
        if self.failed_files:
            self.logger.warning(f"Failed files: {len(self.failed_files)}")
            # Save failed files list
            self.save_failed_files()

    def preprocess_test_group(self) -> np.ndarray:
        '''
        Preprocessing function for determining the data_mask, intended for test group only!
        '''
        self.logger.info(f"Preprocessing test group - computing mean of {len(self.filelist)} images")
    
        # Precompute X-ray statistics
        self.precompute_xray_statistics()

        all_processed_data = []
        for i, filepath in enumerate(self.filelist):
            # Load and process image
            image_file = EMCCDimage(load_tiff_single(filepath))
            image_file.remove_background(
                self.background_data,
                self.X_ray_precompute[0],
                self.X_ray_precompute[1],
                self.X_ray_config[0],
                self.X_ray_config[1]
            )
            
            file_data = image_file.get_processed_data()
            all_processed_data.append(file_data)
        
        # Compute mean
        image_stack = np.stack(all_processed_data, axis=0)
        #mean_image = np.mean(image_stack, axis=0)
        mean_image = np.nanmean(image_stack, axis=0)
        
        print(f"✓ Mean image computed from {len(all_processed_data)} images")
        return mean_image

    def save_results(self, results: List['ProcessedResult'], batch_number: int = None) -> None:
        """Save results to one Parquet file."""
        data = []
        for result in results:
            row = {
                'filename': result.filename,
                'center_x': result.center[0],
                'center_y': result.center[1], 
                'xps_value': result.xps_value
            }
            # Add radial bins
            for i, intensity in enumerate(result.radial_profile):
                row[f'radial_bin_{i:03d}'] = intensity
            data.append(row)
        
        new_df = pd.DataFrame(data)
        filename = self.resultdir / f"xps_{self.xps_value:.5f}.parquet"
        if filename.exists():
            # Read existing file and append
            existing_df = pd.read_parquet(filename)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_parquet(filename, index=False, compression='snappy')
            self.logger.info(f"Appended batch {batch_number} with {len(results)} files (total: {len(combined_df)})")
        else:
            # Create new file
            new_df.to_parquet(filename, index=False, compression='snappy')
            self.logger.info(f"Created new file with batch {batch_number} ({len(results)} files)")

    def save_failed_files(self) -> None:
        """
        Save list of failed files with error messages.
        Overwrites old failed_file.csv
        """
        if not self.failed_files:
            return
            
        failed_df = pd.DataFrame(self.failed_files, columns=['filename', 'error'])
        failed_filepath = self.resultdir / "failed_files.csv"
        failed_df.to_csv(failed_filepath, index=False)
        self.logger.info(f"Saved {len(self.failed_files)} failed files to {failed_filepath}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of processing, no longer active as self.results is empty."""
        if not self.results:
            return {}
        
        centers = np.array([result.center for result in self.results])
        
        return {
            'total_files_processed': len(self.processed_files),
            'total_files_failed': len(self.failed_files),
            'mean_center_x': np.mean(centers[:, 0]),
            'mean_center_y': np.mean(centers[:, 1]),
            'std_center_x': np.std(centers[:, 0]),
            'std_center_y': np.std(centers[:, 1])
        }

    def update_filelists(self, processed_filelist:List[str]):
        """
        Save the processed filelist as a YAML file.
        
        Parameters:
        -----------
        processed_filelist : List[str]
            List of processed filenames (just strings)
        group_no : float
            Group number (typically XPS value)
        dir : str
            Directory where the YAML file will be saved
        """
        # Convert to Path object
        directory = Path(self.resultdir)
        
        # Create filename with 5 decimal places
        filename = f"processed_{self.xps_value:.5f}.yaml"
        filepath = directory / filename
        
        # 1. Load existing data if it exists
        existing_files = []
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing_files = yaml.safe_load(f) or []

        # 2. Combine with new files (and remove duplicates just in case)
        # Using a list comprehension to preserve order while adding new items
        updated_list = existing_files + [f for f in processed_filelist if f not in existing_files]

        # 3. Save the full combined list back to disk
        with open(filepath, 'w') as f:
            yaml.dump(updated_list, f, default_flow_style=False)