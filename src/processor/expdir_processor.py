import numpy as np
import pandas as pd
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import asdict
import json

from src.processor.processor_objects import XPSGroupProcessor
from src.utils.xps_value_sort import group_tiff_files_with_info, merge_xps_groups_manual, merge_xps_groups_strategy
from src.io.tiff_SL import load_tiff_single
from src.masks.mask_class import precompute_ring_mask, precompute_radial_masks, precompute_azimuthal_average_masks, precompute_center_masks
from src.utils.processing_utils import ProcessingConfig

class DirectoryProcessor:
    def __init__(self,
                 result_directory: str,
                 data_directory:str,
                 xps_grouping_param: List[float],
                 xray_removal_param: List[float],
                 center_fitting_param: List[float],
                 azimuthal_avg_param: List[float],
                 background_directory: str = 'default',
                 data_mask_directory: str = 'default'):
        """
        Initialize Directory Processor for sorting and processing files.
        
        Args:
            result_directory: Directory for containing results 
            data_directory: Directory containing data for processing
            xps_grouping_param: [threshold, tolerance] for XPS grouping
            xray_removal_param: [target xps values set:float]
            center_fitting_param: [inner_radius, outer_radius, center_x, center_y]
            azimuthal_avg_param: [radius, num_bins]
            background_directory: Directory for background file or 'default'
            data_mask_directory: Directory for data mask or 'default'
                if empty, processor will use skip masking the data.
        """
        self.result_directory = Path(result_directory)
        self.xps_grouping_param = xps_grouping_param
        self.xray_removal_param = xray_removal_param
        self.center_fitting_param = center_fitting_param
        self.azimuthal_avg_param = azimuthal_avg_param
        self.background_directory = background_directory
        self.background_data = None
        self.data_mask_directory = data_mask_directory
        self.data_mask_data = None
        
        # Initialize empty configs
        self.center_config = []  # [ring_mask, initial_guess]
        self.azimuthal_config = []  # [radial_masks, azimuthal_mask_dict]
        
        # Other necessary variables
        self.merged_groups = []
        self.data_directory = data_directory
        self.lock = threading.Lock()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def initialize_masks(self) -> None:
        """
        Initialize all masks used for processing.
        """
        self.logger.info("Initializing processing masks...")
        
        # Extract parameters
        inner_radius, outer_radius, center_x, center_y = self.center_fitting_param
        radius, num_bins = self.azimuthal_avg_param
        initial_guess = (center_x, center_y)
        
        # Initialize ring mask
        # ring_mask = precompute_ring_mask(
        #     inner_radius=inner_radius, 
        #     outer_radius=outer_radius
        # )
        ring_mask = precompute_center_masks(
            inner_radius=inner_radius, 
            outer_radius=outer_radius
        )

        # Initialize radial masks
        radial_masks = precompute_radial_masks(
            radius=radius, 
            num_bins=num_bins
        )
        
        # Initialize azimuthal mask dictionary
        azimuthal_mask_dict = precompute_azimuthal_average_masks(radial_masks)
        
        # Set center and azimuthal configs
        self.center_config = [ring_mask, initial_guess]
        self.azimuthal_config = [radial_masks, azimuthal_mask_dict]
        
        self.logger.info("Masks initialized successfully")

    def initialize_bkg_and_datamask(self) -> None:
        """
        Initialize background data (and now data mask too!) from specified directory.
        """
        self.logger.info("Initializing background and data mask...")
        
        if self.background_directory == 'default':
            # Load from result_directory/background.tiff
            background_path = self.result_directory / "background.tiff"
            if not background_path.exists():
                raise FileNotFoundError(f"Default background file not found under: {background_path}")
            
            self.background_data = load_tiff_single(background_path)
            self.logger.info(f"Loaded background from: {background_path}")
        
        else:
            # Load from specified directory
            background_dir = Path(self.background_directory)
            tiff_files = list(background_dir.glob("*.tiff")) + list(background_dir.glob("*.tif"))
            
            if not tiff_files:
                raise FileNotFoundError(f"No TIFF files found in background directory: {background_dir}")
            
            # Use the first TIFF file found
            background_file = tiff_files[0]
            self.background_data = load_tiff_single(background_file)
            self.logger.info(f"Loaded background from: {background_file}")

        if self.data_mask_directory == 'default':
            # Same logic to load data_mask
            data_mask_path = self.result_directory / "data_mask.tiff"
            background_mask_path = self.result_directory / "background_mask.tiff"
            if not data_mask_path.exists():
                self.logger.info(f"Default data mask file not found under: {background_path}, proceed without masking!")
                background_mask = np.zeros((1024, 1024), dtype=int)
                background_mask[20:1024-20, :] = 1
                background_mask[:, 20:1024-20] = 1
                self.data_mask_data = [np.ones((1024, 1024), dtype=int),background_mask] 
            else:
                self.data_mask_data = [load_tiff_single(data_mask_path),
                                       load_tiff_single(background_mask_path)]
                self.logger.info(f"Loaded data_mask from: {self.result_directory}")
        
        else:
            data_mask_dir = Path(self.data_mask_directory)
            tiff_files = list(data_mask_dir.glob("*.tiff")) + list(data_mask_dir.glob("*.tif"))
            
            if not tiff_files:
                self.logger.info(f"No TIFF files found in data_mask directory: {background_dir}, proceed without masking!")
                background_mask = np.zeros((1024, 1024), dtype=int)
                background_mask[20:1024-20, :] = 1
                background_mask[:, 20:1024-20] = 1
                self.data_mask_data = [np.ones((1024, 1024), dtype=int),background_mask]
            else:
                #needs fixing!
                data_mask_file = tiff_files[0]
                self.data_mask_data = load_tiff_single(data_mask_file)
                self.logger.info(f"Loaded data_mask from: {data_mask_file}")

    def sort_file_into_xpsgroups(self) -> None:
        """
        Sort files into XPS groups and merge similar groups.
        """
        self.logger.info("Sorting files into XPS groups...")
        
        # Group files by XPS value
        groups_with_xps = group_tiff_files_with_info(self.data_directory)

        # Display group information
        print(f"\nXPS Groups Summary:")
        print("=" * 40)
        for xps_value, files in self.merged_groups:
            print(f"XPS {xps_value:.5f}: {len(files)} files")
        print(f"\nTotal: {len(self.merged_groups)} XPS groups")

        # Group manually to ensure xps consistancy
        manual_groups = self.xps_grouping_param
        self.merged_groups = merge_xps_groups_manual(
            groups_with_xps,
            manual_groups)
        
        # # Extract threshold and tolerance
        # threshold, tolerance = self.xps_grouping_param
        
        # # Merge similar XPS groups
        # self.merged_groups = merge_xps_groups_strategy(
        #     groups_with_xps, 
        #     threshold=threshold, 
        #     tolerance=tolerance
        # )
        
        # Display group information after sorting
        print(f"\nXPS Groups Summary:")
        print("=" * 40)
        for xps_value, files in self.merged_groups:
            print(f"XPS {xps_value:.5f}: {len(files)} files")
        print(f"\nTotal: {len(self.merged_groups)} XPS groups")

    def select_test_xps_group(self) -> None:
        """
        Select the longest XPS group for testing without merging.
        Let user confirm before setting it as self.merged_groups.
        """
        self.logger.info("Selecting test XPS group (longest group)...")
        
        # Group files by XPS value
        groups_with_xps = group_tiff_files_with_info(self.data_directory)
        
        if not groups_with_xps:
            raise ValueError("No XPS groups found in the data directory")
        
        # Find the group with the most files
        longest_group = max(groups_with_xps, key=lambda x: len(x[1]))
        xps_value, file_list = longest_group
        
        # Display information to user
        print(f"\n{'=' * 50}")
        print("TEST XPS GROUP SELECTION")
        print(f"{'=' * 50}")
        print(f"Selected group: XPS {xps_value:.5f}")
        print(f"Number of files: {len(file_list)}")
        print(f"File examples:")
        for i, filepath in enumerate(file_list[:5]):  # Show first 5 files
            print(f"  {Path(filepath).name}")
        if len(file_list) > 5:
            print(f"  ... and {len(file_list) - 5} more files")
        print(f"{'=' * 50}")
        
        # Get user confirmation
        while True:
            response = input("\nUse this group for testing? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                # Set as the only group in merged_groups
                self.merged_groups = [longest_group]
                self.logger.info(f"Test group set: XPS {xps_value:.5f} with {len(file_list)} files")
                
                # Also display other available groups for reference
                print(f"\nOther available XPS groups:")
                print(f"{'-' * 30}")
                for other_xps, other_files in groups_with_xps:
                    if other_xps != xps_value:
                        print(f"XPS {other_xps:.5f}: {len(other_files)} files")
                break
            elif response in ['n', 'no']:
                self.logger.info("User declined to use the test group")
                self.merged_groups = []
                break
            else:
                print("Please enter 'y' or 'n'")

    def select_test_xps_group_numbered(self, group_size: int = 50) -> List[Tuple[float, List[str]]]:
        """
        Alternative version with numbered selection as well as truncated group length.
        """
        self.logger.info("Numbered test XPS group selection...")
        
        # Group files by XPS value
        groups_with_xps = group_tiff_files_with_info(self.data_directory)
        
        if not groups_with_xps:
            raise ValueError("No XPS groups found in the data directory")
        
        # Sort groups by file count (descending)
        sorted_groups = sorted(groups_with_xps, key=lambda x: len(x[1]), reverse=True)
        
        print(f"\n{'=' * 60}")
        print("SELECT TEST XPS GROUP")
        print(f"{'=' * 60}")
        
        # Display all groups with numbers
        for i, (xps_value, file_list) in enumerate(sorted_groups, 1):
            print(f"{i:2d}. XPS {xps_value:.5f}: {len(file_list):4d} files")
        
        print(f"{'=' * 60}")
        
        # User selection
        while True:
            try:
                choice = input(f"\nEnter group number (1-{len(sorted_groups)}) or 'q' to quit: ").strip().lower()
                
                if choice in ['q', 'quit']:
                    print("No group selected.")
                    self.merged_groups = []
                    break
                    
                choice_num = int(choice)
                if 1 <= choice_num <= len(sorted_groups):
                    selected_group = sorted_groups[choice_num - 1]
                    xps_value, file_list = selected_group
                    print(f"✓ Selected group {choice_num}: XPS {xps_value:.5f} with {len(file_list)} files")
                    self.merged_groups = [selected_group]

                    #select group_size of evenly spaced files
                    selected_xps, original_file_list = selected_group
        
                    if len(original_file_list) > group_size:
                        # Select max_files evenly spaced files
                        indices = np.linspace(0, len(original_file_list) - 1, group_size, dtype=int)
                        final_file_list = [original_file_list[i] for i in indices]
                        self.merged_groups = [(selected_xps, final_file_list)]
                        print(f"Limited to {group_size} evenly spaced files (from {len(original_file_list)} total)")
                    else:
                        print("Truncation failed due to group_size bigger than selected group, proceed without truncation.")
                    return self.merged_groups
                    
                else:
                    print(f"Please enter a number between 1 and {len(sorted_groups)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit")

    def process_xps_group(self, xps_group: Tuple[float, List[str]], analyze_no: str) -> None:
        """
        Process a single XPS group.
        
        Args:
            xps_group: Tuple containing (xps_value, file_list)
            analyze_no: Analysis number for organizing results
        """
        # Extract values from tuple
        xps_value, filelist = xps_group
        
        # Create result directory for this XPS group
        group_result_dir = self.result_directory / analyze_no 
        
        with self.lock:
            if group_result_dir.exists():
                raise FileExistsError(f"Result directory already exists: {group_result_dir}")
            group_result_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing XPS group {xps_value:.5f} with {len(filelist)} files")
        
        # Initialize and run XPSGroupProcessor
        processor = XPSGroupProcessor(
            background_data=self.background_data,
            data_mask_data=self.data_mask_data,
            X_ray_config=self.xray_removal_param,
            center_config=self.center_config,
            azimuthal_config=self.azimuthal_config,
            xps_value=xps_value,
            filelist=filelist,
            resultdir=str(group_result_dir)
        )
        
        processor.process_group(batch_size=500)
        self.logger.info(f"Completed processing XPS group {xps_value:.5f}")

    def process_in_sequence(self, analyze_no: str) -> None:
        """
        Process XPS groups sequentially.
        
        Args:
            analyze_no: Analysis number for organizing results
        """
        self.logger.info(f"Starting sequential processing for analysis {analyze_no}")
        
        # Ensure masks and background are initialized
        if not self.center_config or not self.azimuthal_config:
            self.initialize_masks()
        if self.background_data is None or self.data_mask_data is None:
            self.initialize_bkg_and_datamask()
        
        # Sort files if not already sorted
        if not self.merged_groups:
            self.sort_file_into_xpsgroups()
        
        # Save config before processing
        self.save_config(analyze_no)
        
        # Process each group sequentially
        for i, group in enumerate(self.merged_groups):
            self.logger.info(f"Processing group {i+1}/{len(self.merged_groups)}")
            try:
                self.process_xps_group(group, analyze_no)
            except Exception as e:
                self.logger.error(f"Failed to process group {group['xps_value']:.2f}: {str(e)}")
        
        self.logger.info("Sequential processing completed")

    def process_in_parallel(self, max_workers: int, analyze_no: str) -> None:
        """
        Process XPS groups in parallel.
        
        Args:
            max_workers: Maximum number of parallel threads
            analyze_no: Analysis number for organizing results
        """
        self.logger.info(f"Starting parallel processing with {max_workers} workers for analysis {analyze_no}")
        
        # Ensure masks and background are initialized
        if not self.center_config or not self.azimuthal_config:
            self.initialize_masks()
        if self.background_data is None or self.data_mask_data is None:
            self.initialize_bkg_and_datamask()
        
        # Sort files if not already sorted
        if not self.merged_groups:
            self.sort_file_into_xpsgroups()
        
        # Save config before processing
        self.save_config(analyze_no)
        
        # Process groups in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_group = {
                executor.submit(self.process_xps_group, group, analyze_no): group 
                for group in self.merged_groups
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    future.result()
                    xps_val = group[0]  # Extract xps_value from tuple
                    self.logger.info(f"Successfully completed group {xps_val:.2f}")
                except Exception as e:
                    xps_val = group[0]  # Extract xps_value from tuple
                    self.logger.error(f"Failed to process group {xps_val:.2f}: {str(e)}")
        
        self.logger.info("Parallel processing completed")

    def preprocess_screening(self, group_size: int) -> np.ndarray:
        """
        Select group_size files from a chosen xps group to preprocess
        """
        self.logger.info(f"Starting preprocess screening with {group_size} images")
        
        # Ensure masks and background are initialized
        if not self.center_config or not self.azimuthal_config:
            self.initialize_masks()
        if self.background_data is None or self.data_mask_data is None:
            self.initialize_bkg_and_datamask()
        
        # Call select_test_xps_group_numbered to get test group
        if not self.merged_groups:
            self.select_test_xps_group_numbered(group_size)
        
        # Save config before processing
        self.save_config("preprocessing_screening_config")

        # Extract values from test group
        xps_value, filelist = self.merged_groups[0]

        preprocessor = XPSGroupProcessor(
            background_data=self.background_data,
            data_mask_data=self.data_mask_data,
            X_ray_config=self.xray_removal_param,
            center_config=self.center_config,
            azimuthal_config=self.azimuthal_config,
            xps_value=xps_value,
            filelist=filelist,
            resultdir=''
        )
        return preprocessor.preprocess_test_group()

    def save_config(self, analyze_no: str) -> None:
        """
        Save processing configuration to file.
        
        Args:
            analyze_no: Analysis number for organizing results
        """
        config_dir = self.result_directory / analyze_no
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config object
        config = ProcessingConfig(
            result_directory=str(self.result_directory),
            background_directory=self.background_directory,
            data_mask_directory=self.data_mask_directory,
            xps_grouping_param=self.xps_grouping_param,
            xray_removal_param=self.xray_removal_param,
            center_fitting_param=self.center_fitting_param,
            azimuthal_avg_param=self.azimuthal_avg_param
        )
        
        # Save as CSV
        config_df = pd.DataFrame([asdict(config)])
        config_csv_path = config_dir / "config.csv"
        config_df.to_csv(config_csv_path, index=False)
        
        # Also save as JSON for easier reading
        config_json_path = config_dir / "config.json"
        with open(config_json_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_csv_path}")

    def load_config(self, analyze_no: str) -> None:
        """
        Load processing configuration from file.
        
        Args:
            analyze_no: Analysis number to load configuration from
        """
        config_path = self.result_directory / analyze_no / "config.csv"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        config_df = pd.read_csv(config_path)
        config_dict = config_df.iloc[0].to_dict()
        
        # Update instance variables
        self.result_directory = Path(config_dict['result_directory'])
        self.background_directory = config_dict['background_directory']
        self.data_mask_directory = config_dict['data_mask_directory']
        self.xps_grouping_param = eval(config_dict['xps_grouping_param']) if isinstance(config_dict['xps_grouping_param'], str) else config_dict['xps_grouping_param']
        self.xray_removal_param = eval(config_dict['xray_removal_param']) if isinstance(config_dict['xray_removal_param'], str) else config_dict['xray_removal_param']
        self.center_fitting_param = eval(config_dict['center_fitting_param']) if isinstance(config_dict['center_fitting_param'], str) else config_dict['center_fitting_param']
        self.azimuthal_avg_param = eval(config_dict['azimuthal_avg_param']) if isinstance(config_dict['azimuthal_avg_param'], str) else config_dict['azimuthal_avg_param']
        
        # Reinitialize masks with loaded parameters
        self.initialize_masks()
        self.initialize_bkg_and_datamask()
        
        self.logger.info(f"Configuration loaded from {config_path}")

'''
define a class DirectoryProcessor to sort and process the files under result directory:
the class shoud have self variables:
result_directory
background_directory
background_data (initially empty)
xps_grouping_param List[threshold , tolerance]
xray_removal_param List[sigma_threshold, expansion_threshold_ratio]
center_fitting_param List[inner_radius,outer_radius,center_guess[float,float]]
center_config: [ring_mask, initial_guess](initially empty)
azimuthal_avg_param List[radius,num_bins]
azimuthal_config: [radial_masks, azimuthal_mask_dict](initially empty)
and other necessary variables.

the class should take advantage of the predefined XPSGroupProcessor to process

class functions:
initialize_masks: initialize all the mask used for processing by
    ring_mask = precompute_ring_mask(inner_radius=center_fitting_param[0], outer_radius=center_fitting_param[1])
    radial_masks = precompute_radial_masks(radius=azimuthal_avg_param[0], num_bins=azimuthal_avg_param[1])
    azimuthal_mask_dict = precompute_azimuthal_average_masks(radial_masks)
    then initialize center_config and azimuthal_config 

initialize_background: 
    if the background_directory == 'default', then background is at result_directory/background.tiff, load using TiffLoader.
    otherwise, background_data = TiffLoader(Path(filepath).parent, Path(filepath).name)

sort_file_into_xpsgroups: main logic as follows
        # Group files by XPS value
        groups_with_xps = group_tiff_files_with_info(self.data_directory)
        
        # Merge similar XPS groups
        self.merged_groups = merge_xps_groups_strategy(
            groups_with_xps, 
            threshold=threshold, 
            tolerance=tolerance
        )

process_in_sequence(analyze_no): initialize XPSGroupProcessor, process xps group one by one
process_in_parallel(max_workers,analyze_no): 
    process in multiple threads, each xps group gats one thread, there should be no more 
    then max_workers of threads running at the same time.

    for the process_in_sequence and process_in_parallel, the result directory for a xpsgroup
    should be:self.result_directory/analyze_no/xps_value, create the directory before 
    launching XPSGroupProcessor, raise Error if xps_value folder is already created 

load_config(analyze_no): load the parameters required upon initializing from result_directory/analyze_no/config.csv 
    use this for initializing the DirectoryProcessor class

also define a seperate function and dataclass outside of the class to save all the parameters 
    required upon initializing to result_directory/analyze_no/config.csv
'''
#xps format needs to be updated for background scans