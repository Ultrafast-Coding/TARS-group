import numpy as np
from typing import List
from pathlib import Path
import random

from src.io.tiff_SL import load_tiff_batch 
from src.processor.tiff_objects import EMCCDimage
    
class CalcBackground:
    """
    Process background images by computing mean pixel values across multiple TIFF images
    after removing X-ray spots from each individual image.
    """
    
    def __init__(self, bkg_directory: str):
        """
        Initialize CrudeBackground with directory containing background TIFF images.
        
        Parameters:
        -----------
        bkg_directory : str
            Path to directory containing background TIFF images
        """
        self.bkg_directory = Path(bkg_directory)
        self.processed_bkg = None
        
        # Validate directory exists
        if not self.bkg_directory.exists():
            raise FileNotFoundError(f"Background directory not found: {self.bkg_directory}")
        if not self.bkg_directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.bkg_directory}")
        
    def process_background_choice(self, method = str) -> np.ndarray:
        """
        Load all TIFF images in background directory and compute background with given method for each pixel..
        
        Parameters:
        -----------
        method : str
            'mean'
            'median'
            'di'  for calculating median-mean to show Xray and deviation
        """

        print(f"Processing background images from: {self.bkg_directory}")
        
        # Load all TIFF images using TiffLoaderBatch
        try:
            images: List[np.ndarray] = load_tiff_batch(str(self.bkg_directory))
        except Exception as e:
            raise RuntimeError(f"Failed to load background images: {e}") from e
        if not images:
            raise ValueError(f"No TIFF images found in {self.bkg_directory}")
        
        print(f"Loaded {len(images)} background images")
        
        # Check that all images have the same shape
        first_shape = images[0].shape
        for i, img in enumerate(images):
            if img.shape != first_shape:
                raise ValueError(
                    f"Image shape mismatch: Image {i} has shape {img.shape}, "
                    f"expected {first_shape}. All background images must have identical dimensions."
                )
        
        # Convert list to numpy array stack
        image_stack = np.array(images)
        # Compute background along the stack axis (axis=0)
        if method == "mean":
            self.processed_bkg = np.mean(image_stack, axis=0)
        elif method == "median":
            self.processed_bkg = np.median(image_stack, axis=0)
        elif method == "di":
            self.processed_bkg = np.median(image_stack, axis=0)-np.mean(image_stack, axis=0)
        else:
            print("Unknown method, processing with mean!")
            self.processed_bkg = np.mean(image_stack, axis=0)
        
        print(f"Computed crude median background - Shape: {self.processed_bkg.shape}, ")
        
        return self.processed_bkg
    
    def process_background_Xray(self, 
                         sigma_threshold: float = 10,
                         expansion_threshold_ratio: float = 0.7,
                         center_pos: List[int] = [0,0,0]) -> np.ndarray:
        """
        Load all TIFF images, remove X-ray spots from each, and compute mean background.
        
        Parameters:
        -----------
        sigma_threshold: Threshold for identifying X-ray hits
        expansion_threshold_ratio: Ratio for expanding X-ray region detection
        center_pos: Center position_x, _y and center size 
            X-ray filtering will be excluded for this central area 
        
        Returns:
        --------
        np.ndarray: Processed background image (mean of cleaned images)
        """
        SAMPLE_SIZE = 50

        print(f"Processing background images from: {self.bkg_directory}")
        
        # Load all TIFF images
        try:
            images: List[np.ndarray] = load_tiff_batch(str(self.bkg_directory))
        except Exception as e:
            raise RuntimeError(f"Failed to load background images: {e}") from e
        
        if not images:
            raise ValueError(f"No TIFF images found in {self.bkg_directory}")
        
        print(f"Loaded {len(images)} background images")
        
        # Check that all images have the same shape
        first_shape = images[0].shape
        for i, img in enumerate(images):
            if img.shape != first_shape:
                raise ValueError(
                    f"Image shape mismatch: Image {i} has shape {img.shape}, "
                    f"expected {first_shape}. All background images must have identical dimensions."
                )
        
        # Determine how many files to sample
        if len(images) <= SAMPLE_SIZE:
            sampled_images = images
            print(f"Using all {len(sampled_images)} files for X-ray statistics precomputation")
        else:
            # Randomly sample files
            sampled_images = random.sample(images, SAMPLE_SIZE)
            print(f"Randomly sampled {len(sampled_images)} files for X-ray statistics precomputation")
        
        # Stack images and compute statistics
        image_stack = np.stack(sampled_images, axis=0)
        
        # Compute median across the sample
        median_array = np.median(image_stack, axis=0)
        
        # Compute MAD (Median Absolute Deviation)
        abs_deviation = np.abs(image_stack - median_array)
        mad_array = np.median(abs_deviation, axis=0)

        # in case of checking if the xray removing method was effective for datasets
        center_x, center_y = center_pos[0], center_pos[1]  # initial_guess tuple
        
        # Set center region to NaN to ignore bright diffraction center
        height, width = median_array.shape
        half_size = center_pos[2] // 2
        
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

        # Process each image to remove X-ray spots
        cleaned_images = []
        
        for i, img_data in enumerate(images):
            print(f"Processing image {i+1}/{len(images)} for X-ray removal...")
            
            # Create EMCCDimage instance
            img_obj = EMCCDimage(img_data)
            img_obj.copy_as_processed()

            # Apply X-ray filtering in place
            img_obj.filter_xray_light(
                median_array = median_array,
                mad_array = mad_array,
                sigma_threshold = sigma_threshold,
                expansion_threshold_ratio = expansion_threshold_ratio
            )
            
            # Store the cleaned image (raw_data now contains NaN for removed pixels)
            cleaned_images.append(img_obj.processed_data)
        
        # Convert to numpy array stack
        image_stack = np.array(cleaned_images)
        
        # Compute mean along the stack axis, ignoring NaN values
        self.processed_bkg = np.nanmean(image_stack, axis=0)
        
        # Optional: Fill remaining NaN values with overall mean
        if np.any(np.isnan(self.processed_bkg)):
            overall_mean = np.nanmean(self.processed_bkg)
            self.processed_bkg = np.nan_to_num(self.processed_bkg, nan=overall_mean)
            print(f"Filled {np.sum(np.isnan(cleaned_images[0]))} NaN pixels with mean value")
        
        print(f"Computed X-ray cleaned mean background - Shape: {self.processed_bkg.shape}")
        print(f"Background value range: [{np.min(self.processed_bkg):.1f}, {np.max(self.processed_bkg):.1f}]")
        
        return self.processed_bkg
    
    def get_processed_background(self) -> np.ndarray:
        #Get the processed background image. Processes if not already computed.

        if self.processed_bkg is None:
            return self.process_background_Xray(sigma_threshold= 10, expansion_threshold_ratio= 0.7, center_pos=[0,0,0])
        return self.processed_bkg