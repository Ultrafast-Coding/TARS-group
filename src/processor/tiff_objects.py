import numpy as np
from typing import Optional, Tuple, List
from scipy.ndimage import shift
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize

#from processor_code.utils.gaussian_fitting import fit_circular_gaussian_ring
from src.masks.mask_class import RadialMasks, RingMask
from src.utils.timer import timer

class EMCCDimage:
    """
    EMCCD image class for storing and processing EMCCD camera data.
    """
    
    def __init__(self, raw_data: np.ndarray):
        """
        Initialize EMCCDimage with raw data.
        
        Parameters:
        -----------
        raw_data : np.ndarray
            Raw image data from EMCCD camera
        """
        self.raw_data = raw_data
        self.processed_data: Optional[np.ndarray] = None
        self.center_pos: Optional[Tuple] = None
        self.total_count: Optional[float] = None
    
    def filter_xray_spots_inplace(self, 
                                 chunk_size: int = 32,
                                 sigma_threshold: float = 29.6,
                                 beam_threshold: float = 300) -> None:
        """
        Remove X-ray spots from raw data by local statistical filtering.
        
        X-ray spots appear as 4~10 pixel bright spots significantly brighter than
        local background. This method divides the image into chunks and applies
        statistical filtering to identify and remove outliers in place.
        
        Args:
            chunk_size: Size of square chunks for local processing (default: 64)
            sigma_threshold: Number of standard deviations above mean for outlier detection
            beam_threshold: Intensity threshold to identify central beam regions
            
        Raises:
            ValueError: If raw_data is not available
            ValueError: If chunk_size is not a divisor of image dimensions
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available")
        
        height, width = self.processed_data.shape
        
        # Validate chunk_size
        if height % chunk_size != 0 or width % chunk_size != 0:
            raise ValueError(f"chunk_size {chunk_size} must divide both image dimensions {height}x{width}")
        
        # Convert to float for NaN support if not already
        if not np.issubdtype(self.processed_data.dtype, np.floating):
            self.processed_data = self.processed_data.astype(float)
        
        # Process each chunk
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                row_end = row_start + chunk_size
                col_end = col_start + chunk_size
                
                # Extract current chunk
                chunk = self.processed_data[row_start:row_end, col_start:col_end]
                
                # Calculate chunk statistics
                chunk_median = np.median(chunk)

                # Calculate Median Absolute Deviation (MAD),29.6~20sigma for gaussian
                chunk_mad = np.median(np.abs(chunk - chunk_median))
                
                # Skip processing if chunk contains central beam (high average intensity)
                if chunk_median > beam_threshold:
                    continue
                
                # Calculate outlier threshold
                outlier_threshold = chunk_median + sigma_threshold * chunk_mad
                
                # Find pixels exceeding threshold and replace with NaN in place
                outlier_mask = chunk > outlier_threshold
                if np.any(outlier_mask):
                    outlier_mask = chunk > outlier_threshold - sigma_threshold * 0.25 * chunk_mad
                    self.processed_data[row_start:row_end, col_start:col_end][outlier_mask] = np.nan

    def filter_xray_light(self, 
                         median_array: np.ndarray,
                         mad_array: np.ndarray,
                         sigma_threshold: float = 10,
                         expansion_threshold_ratio: float = 0.7) -> None:
        """
        Remove X-ray hits using precomputed median and MAD arrays.
        
        Args:
            median_array: Precomputed median values for each pixel
            mad_array: Precomputed MAD values for each pixel  
            sigma_threshold: Threshold for identifying X-ray hits
            expansion_threshold_ratio: Ratio for expanding X-ray region detection
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available")
        
        # Convert to float for NaN support if not already
        if not np.issubdtype(self.processed_data.dtype, np.floating):
            self.processed_data = self.processed_data.astype(float)
        
        # Convert sigma to mad assuming normal distribution
        mad_threshold = sigma_threshold / 1.4286

        # Calculate outlier threshold for each pixel
        outlier_threshold = median_array + mad_threshold * mad_array
        
        # Find initial X-ray hits (pixels significantly brighter than group median)
        xray_mask = self.processed_data > outlier_threshold
        
        if not np.any(xray_mask):
            return  # No X-rays found
        
        # Expand mask to include surrounding pixels with lower threshold
        expansion_threshold = median_array + (mad_threshold * expansion_threshold_ratio) * mad_array
        
        # Use binary dilation to expand the X-ray regions
        # Create structure for dilation (3x3 cross)
        structure = np.array([[0, 1, 0],
                            [1, 1, 1], 
                            [0, 1, 0]], dtype=bool)
        
        # Dilate the initial X-ray mask
        expanded_mask = binary_dilation(xray_mask, structure=structure)
        
        # Include pixels that exceed the expansion threshold within the dilated region
        final_xray_mask = expanded_mask & (self.processed_data > expansion_threshold)
        
        # Replace X-ray pixels with NaN
        self.processed_data[final_xray_mask] = np.nan

    def filter_xray_spots(self, 
                         chunk_size: int = 64,
                         sigma_threshold: float = 5.0,
                         beam_threshold: float = 5000) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Remove X-ray spots from raw data by local statistical filtering.
        
        X-ray spots appear as 4~10 pixel bright spots significantly brighter than
        local background. This method divides the image into chunks and applies
        statistical filtering to identify and remove outliers.
        
        Args:
            chunk_size: Size of square chunks for local processing (default: 64)
            sigma_threshold: Number of standard deviations above mean for outlier detection
            beam_threshold: Intensity threshold to identify central beam regions
            
        Returns:
            tuple: Contains two elements:
                - processed_data: X-ray filtered numpy array (NaN for removed pixels)
                - removed_positions: List of (row, col) coordinates of removed pixels
                
        Raises:
            ValueError: If raw_data is not available
            ValueError: If chunk_size is not a divisor of image dimensions
            
        Example:
            >>> filtered_data, removed_pixels = image.filter_xray_spots()
            >>> print(f"Removed {len(removed_pixels)} X-ray affected pixels")
        """
        if self.raw_data is None:
            raise ValueError("Raw data not available")
        
        height, width = self.raw_data.shape
        
        # Validate chunk_size
        if height % chunk_size != 0 or width % chunk_size != 0:
            raise ValueError(f"chunk_size {chunk_size} must divide both image dimensions {height}x{width}")
        
        # Create copy of raw data for processing
        processed_data = self.raw_data.copy().astype(float)
        removed_positions = []
        
        print(f"Filtering X-ray spots: {height}x{width} image, {chunk_size}x{chunk_size} chunks")
        print(f"Threshold: {sigma_threshold}σ above local mean, beam threshold: {beam_threshold}")
        
        # Process each chunk
        chunks_processed = 0
        chunks_with_beam = 0
        total_pixels_removed = 0
        
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                row_end = row_start + chunk_size
                col_end = col_start + chunk_size
                
                # Extract current chunk
                chunk = self.raw_data[row_start:row_end, col_start:col_end]
                
                # Calculate chunk statistics
                chunk_mean = np.mean(chunk)
                chunk_std = np.std(chunk)
                
                # Skip processing if chunk contains central beam (high average intensity)
                if chunk_mean > beam_threshold:
                    chunks_with_beam += 1
                    continue
                
                # Calculate outlier threshold
                outlier_threshold = chunk_mean + sigma_threshold * chunk_std
                
                # Find pixels exceeding threshold
                outlier_mask = chunk > outlier_threshold
                outlier_positions = np.where(outlier_mask)
                
                if np.any(outlier_mask):
                    # Convert local chunk coordinates to global image coordinates
                    global_rows = outlier_positions[0] + row_start
                    global_cols = outlier_positions[1] + col_start
                    
                    # Record removed positions
                    for global_row, global_col in zip(global_rows, global_cols):
                        removed_positions.append((int(global_row), int(global_col)))
                    
                    # Replace outliers with NaN in processed data
                    processed_data[global_rows, global_cols] = np.nan
                    total_pixels_removed += len(global_rows)
                
                chunks_processed += 1
        
        # Print processing summary
        self._print_filtering_summary(chunks_processed, chunks_with_beam, total_pixels_removed, removed_positions)
        
        return processed_data, removed_positions
    
    def _print_filtering_summary(self, 
                               chunks_processed: int,
                               chunks_with_beam: int,
                               total_pixels_removed: int,
                               removed_positions: List[Tuple[int, int]]) -> None:
        """
        Print detailed summary of X-ray filtering results.
        
        Args:
            chunks_processed: Number of chunks processed for X-ray filtering
            chunks_with_beam: Number of chunks skipped due to beam presence
            total_pixels_removed: Total number of pixels removed
            removed_positions: List of removed pixel coordinates
        """
        print("\n" + "="*50)
        print("X-RAY FILTERING SUMMARY")
        print("="*50)
        print(f"Chunks processed: {chunks_processed}")
        print(f"Chunks with beam (skipped): {chunks_with_beam}")
        print(f"Total pixels removed: {total_pixels_removed}")
        print(f"Removal rate: {total_pixels_removed / (1024*1024) * 100:.4f}%")
        
        if removed_positions:
            print(f"\nFirst 10 removed pixel positions (row, col):")
            for i, (row, col) in enumerate(removed_positions[:10]):
                intensity = self.raw_data[row, col]
                print(f"  ({row:4d}, {col:4d}) - Intensity: {intensity:8.1f}")
            
            if len(removed_positions) > 10:
                print(f"  ... and {len(removed_positions) - 10} more positions")
            
            # Calculate spatial distribution
            rows, cols = zip(*removed_positions)
            print(f"\nSpatial distribution:")
            print(f"  Row range: {min(rows)} - {max(rows)}")
            print(f"  Col range: {min(cols)} - {max(cols)}")
            
            # Check for clusters (potential multi-pixel X-ray events)
            self._analyze_xray_clusters(removed_positions)
        else:
            print("No X-ray spots detected")
    
    def _analyze_xray_clusters(self, removed_positions: List[Tuple[int, int]]) -> None:
        """
        Analyze spatial clustering of removed pixels to identify multi-pixel X-ray events.
        
        Args:
            removed_positions: List of (row, col) coordinates of removed pixels
        """
        
        # Group pixels by proximity (4-connectivity)
        visited = set()
        clusters = []
        
        for position in removed_positions:
            if position in visited:
                continue
            
            # Start new cluster
            cluster = []
            stack = [position]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.append(current)
                
                # Check 4-connected neighbors
                row, col = current
                neighbors = [
                    (row-1, col), (row+1, col),
                    (row, col-1), (row, col+1)
                ]
                
                for neighbor in neighbors:
                    if neighbor in removed_positions and neighbor not in visited:
                        stack.append(neighbor)
            
            if cluster:
                clusters.append(cluster)
        
        # Analyze cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        print(f"\nX-ray cluster analysis:")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Cluster size range: {min(cluster_sizes)} - {max(cluster_sizes)} pixels")
        
        # Count clusters by size and track large clusters
        size_bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20)]
        large_clusters = []
        
        for low, high in size_bins:
            matching_clusters = [cluster for cluster in clusters if low <= len(cluster) <= high]
            count = len(matching_clusters)
            
            if count > 0:
                print(f"  Clusters with {low}-{high} pixels: {count}")
                
                # Store large clusters for detailed reporting
                if low == 7 or low == 11:  # 7-20 pixel clusters
                    large_clusters += matching_clusters
        
        # Print position from 7-20 pixel clusters
        if large_clusters:
            print(f"\n7-20 pixel clusters:")
            for i in range(len(large_clusters)):
                # for all cluster from 7-20 pixel clusters
                selected_cluster = large_clusters[i]
                # Select a pixel from that cluster
                pixel = selected_cluster[0]
                row, col = pixel
                intensity = self.raw_data[row, col]
                print(f"  Position: ({row}, {col}), Intensity: {intensity:.1f}")

    def remove_background_legacy(self, background: np.ndarray) -> None:
        """
        Subtract background from raw_data to produce processed_data.
        
        Parameters:
        -----------
        background : np.ndarray
            Background image to subtract from raw_data
        """

        self.processed_data = self.raw_data - background
        #self.filter_xray_spots_inplace(chunk_size, sigma_threshold, beam_threshold)
    
    def remove_background(self, 
                          background: np.ndarray,
                          median_array: np.ndarray,
                          mad_array: np.ndarray,
                          sigma_threshold: float = 10,
                          expansion_threshold_ratio: float = 0.7):
        '''
        Subtract background from raw_data AND filter Xray to produce processed_data.
        
        Parameters:
        -----------
        background : np.ndarray
            Background image to subtract from raw_data
        median_array: Precomputed median values for each pixel
        mad_array: Precomputed MAD values for each pixel  
        sigma_threshold: Threshold for identifying X-ray hits
        expansion_threshold_ratio: Ratio for expanding X-ray region detection
        '''
        self.processed_data = self.raw_data - background
        self.filter_xray_light(median_array, mad_array, sigma_threshold, expansion_threshold_ratio)

    def apply_data_mask(self, filter_mask: List[np.ndarray], expanded_size: int = 1024) -> None:
        '''
        Apply the filter_mask:[data_mask, bkg_mask] (1 to keep, 0 to discard) to processed_data. 
        
        Parameters:
        -----------
        filter_mask : List[np.ndarray, np.ndarray]
            Mask to remove background drift and noise/area without signal
        '''
        # Removing background drift
        try:
            background_mean = np.nanmean(self.processed_data[filter_mask[1] == 0])
            self.processed_data -= background_mean
        except Exception as e:
            # If the bkg mask was set to 1, skip processing
            return

        # Setting noise/background to np.nan
        condition = filter_mask[0] == 1
        self.processed_data = np.where(condition, self.processed_data, np.nan)

        # Pad to expanded_size for large-q data 
        if expanded_size != 1024 and expanded_size%2 == 0:
            pad_size = expanded_size/2 - 512
            self.processed_data = np.pad(self.processed_data, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=np.nan)

    def get_processed_data(self) -> np.ndarray:
        """
        Get the processed data after background removal.
        
        Returns:
        --------
        np.ndarray
            Processed image data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call remove_background() first.")
        return self.processed_data
    
    def copy_as_processed(self):
        # Use With Care!!
        self.processed_data = self.raw_data
    
    def fit_diffraction_center(self, 
                           initial_guess: Tuple[float, float, float, float, float],
                           inner_radius: float = 40,
                           outer_radius: float = 200) -> Optional[Tuple[float, float]]:
        '''
        Find diffraction center using circular Gaussian fitting with tight bounds
        
        Parameters:
        -----------
        initial_guess : tuple
            [amplitude, x_center, y_center, sigma, offset]
        inner_radius : float
            Inner radius of fitting ring
        outer_radius : float
            Outer radius of fitting ring
        
        Returns:
        --------
        tuple or None : Found center coordinates (x, y)
        '''
        if self.processed_data is None:
            raise ValueError("Call remove_background() first")
    
        '''print(f"Fitting circular Gaussian to ring region {inner_radius}-{outer_radius} pixels")
        print(f"Initial guess: A={initial_guess[0]:.1f}, center=({initial_guess[1]:.1f}, {initial_guess[2]:.1f}), "
            f"σ={initial_guess[3]:.1f}, offset={initial_guess[4]:.1f}")'''
    
        # Perform fitting
        fit_result = fit_circular_gaussian_ring(
            self.processed_data, initial_guess, inner_radius, outer_radius
        )
    
        if fit_result['success']:
            #self.fit_result = fit_result
            #self.diffraction_center = (fit_result['x_center'], fit_result['y_center'])
            self.center_pos = (fit_result['x_center'], fit_result['y_center'])

            '''print(f"Fitting successful!")
            print(f"Diffraction center: ({fit_result['x_center']:.2f} ± {fit_result['x_center_err']:.2f}, "
              f"{fit_result['y_center']:.2f} ± {fit_result['y_center_err']:.2f})")
            print(f"Gaussian width: {fit_result['sigma']:.2f} ± {fit_result['sigma_err']:.2f} pixels")
            print(f"Amp,offset: {fit_result['amplitude']:.2f} , {fit_result['offset']:.2f} pixels")'''

            y_coords, x_coords = np.mgrid[0:self.processed_data.shape[0], 0:self.processed_data.shape[1]]
        
            # Create ring mask
            distances = np.sqrt((x_coords - initial_guess[1])**2 + (y_coords - initial_guess[2])**2)
            ring_mask = (distances >= inner_radius) & (distances <= outer_radius)
            
            # Extract ring region
            ring_data = self.processed_data[ring_mask]
        
            # Calculate total intensity
            self.total_intensity = np.sum(ring_data)
            return self.center_pos
        else:
            print("Fitting failed")
            return None

    def ring_centroid(self,
                  ring_mask: RingMask,
                  center_guess: Tuple[float, float],
                  save_total_count: bool = False) -> Tuple[float, float]:
        """
        Calculate centroid using precomputed ring mask and specified center guess.
        
        This optimized version shifts the image data to align the specified center
        guess with the precomputed mask center, eliminating the need to recalculate
        the ring mask for each image and center combination.
        
        Args:
            ring_mask: Precomputed RingMask object centered at image center
            center_guess: Center coordinates to use for centroid calculation (x, y)
            save_total_count: If True, save total intensity count to self.total_count
            
        Returns:
            tuple: Centroid coordinates (x_center, y_center)
            
        Raises:
            ValueError: If processed_data is not available
            ValueError: If image shape doesn't match mask shape
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available. Call remove_background() first.")
        
        # Validate image shape matches mask shape
        if self.processed_data.shape != ring_mask.image_shape:
            raise ValueError(
                f"Image shape {self.processed_data.shape} doesn't match "
                f"mask shape {ring_mask.image_shape}"
            )
        
        # Calculate shift needed to align specified center with mask center
        center_x, center_y = center_guess
        mask_center_x, mask_center_y = (
            ring_mask.image_shape[1] // 2, 
            ring_mask.image_shape[0] // 2
        )
        
        shift_x = int(round(center_x - mask_center_x))
        shift_y = int(round(center_y - mask_center_y))
        
        # Use scipy.ndimage.shift instead of np.roll to handle NaN values properly
        shifted_data = shift(self.processed_data, 
                            shift=(-shift_y, -shift_x), 
                            order=0,  # nearest neighbor
                            mode='constant', 
                            cval=np.nan)
        
        # Extract intensity values within ring region using precomputed mask
        ring_data = shifted_data[ring_mask.mask]
        
        # Filter out NaN values before centroid calculation
        valid_mask = ~np.isnan(ring_data)
        
        if np.sum(valid_mask) == 0:
            # No valid data in ring region, return original guess
            if save_total_count:
                self.total_count = 0
            return center_guess
        
        valid_ring_data = ring_data[valid_mask]
        valid_x_coords = ring_mask.x_coords_ring[valid_mask]
        valid_y_coords = ring_mask.y_coords_ring[valid_mask]
        
        # Calculate weighted centroid using only valid data
        total_intensity = np.sum(valid_ring_data)
        
        if total_intensity > 0:
            x_center = np.sum(valid_x_coords * valid_ring_data) / total_intensity
            y_center = np.sum(valid_y_coords * valid_ring_data) / total_intensity
        else:
            # Fallback to center guess if no intensity in ring
            x_center, y_center = mask_center_x, mask_center_y
        
        # Apply reverse shift to get coordinates in original image space
        final_x_center = x_center + shift_x
        final_y_center = y_center + shift_y
        
        if save_total_count:
            self.total_count = total_intensity
        
        return final_x_center, final_y_center

    def ring_centroid_legacy(self,
                  center_guess: Tuple[float, float],
                  inner_radius: float = 40,
                  outer_radius: float = 200,
                  save_total_count: bool = False) -> Tuple[float, float]:
        """
        Calculate centroid within ring region
        """
        y_coords, x_coords = np.mgrid[0:self.processed_data.shape[0], 0:self.processed_data.shape[1]]
        
        # Create ring mask
        distances = np.sqrt((x_coords - center_guess[0])**2 + (y_coords - center_guess[1])**2)
        ring_mask = (distances >= inner_radius) & (distances <= outer_radius)
        
        # Extract ring region
        ring_data = self.processed_data[ring_mask]
        ring_x = x_coords[ring_mask]
        ring_y = y_coords[ring_mask]
    
        # Calculate weighted centroid
        total_intensity = np.sum(ring_data)
        if total_intensity > 0:
            x_center = np.sum(ring_x * ring_data) / total_intensity
            y_center = np.sum(ring_y * ring_data) / total_intensity
        else:
            x_center, y_center = center_guess
        
        if save_total_count:
            self.total_count = total_intensity
        return x_center, y_center
 
    def iterative_ring_centroid(self,
                           ring_mask: RingMask,
                           initial_guess: Tuple[float, float],
                           max_iter: int = 10,
                           tolerance: float = 1.0) -> Tuple[float, float]:
        """
        Iteratively refine center coordinates using ring centroid method.
        
        This method performs multiple iterations of ring centroid calculation,
        using the result from each iteration as the center for the next ring.
        The process continues until convergence or maximum iterations reached.

        Note: When the data is heavily polluted or asymmetrical, centroid method is unreliable
        
        Args:
            initial_guess: Initial center coordinates as (x, y) tuple in pixels.
            max_iter: Maximum number of iterations to perform. Defaults to 10.
            tolerance: Convergence tolerance in pixels. Iteration stops when center 
                      shift is less than this value. Defaults to 1.0.
        
        Returns:
            Final refined center coordinates as (x, y) tuple.

        Note:
            The ring centroid method calculates the intensity-weighted center of mass
            within a ring region around the current center estimate. This iterative
            approach helps converge to the true diffraction pattern center.
        """
        current_center = initial_guess
        
        for iteration in range(max_iter):
            
            new_center = self.ring_centroid(ring_mask, current_center)
            # Check convergence
            shift_distance = np.sqrt((new_center[0] - current_center[0])**2 + 
                                    (new_center[1] - current_center[1])**2)
            #print(f"Iteration {iteration+1}: Center shift = {shift_distance:.3f} pixels")

            current_center = new_center
            if shift_distance < tolerance:
                break
        
        self.center_pos = self.ring_centroid(ring_mask, current_center, save_total_count=True)
        return self.center_pos
    
    def calculate_std_sum_with_masks(self,
                                    center: Tuple[float, float],
                                    radial_masks: RadialMasks) -> float:
        """
        Calculate weighted sum of standard deviations using precomputed masks.
        
        Args:
            image_data: Processed diffraction pattern data
            radial_masks: Precomputed radial masks
            center: Center coordinates (x, y)
            
        Returns:
            float: Sum of bin_center^2 * std_dev for all valid radial bins
        """
        center_x, center_y = center
        shift_x = int(round(center_x - radial_masks.image_shape[1] // 2))
        shift_y = int(round(center_y - radial_masks.image_shape[0] // 2))
        
        # Shift image to align with precomputed masks
        if shift_x == 0 and shift_y == 0:
            shifted_data = self.processed_data
        else:
            shifted_data = shift(self.processed_data, 
                            shift=(-shift_y, -shift_x), 
                            order=0,  # nearest neighbor
                            mode='constant', 
                            cval=np.nan)
        
        total_weighted_std = 0.0
        valid_bins = 0
        
        #with timer("std_calc"):
        for i, mask in enumerate(radial_masks.masks):
            # Extract intensities using precomputed mask
            intensities = shifted_data[mask]

            # Need at least 2 points to calculate standard deviation
            if len(intensities) >= 2:
                std_dev = np.nanstd(intensities)
                bin_center = radial_masks.bin_centers[i]
                weighted_std = (bin_center ** 2) * std_dev
                total_weighted_std += weighted_std
                valid_bins += 1
        
        return total_weighted_std / valid_bins

    def find_center_with_std(self,
                                    radial_masks: RadialMasks,
                                    initial_guess: Tuple[float, float],
                                    max_iter: int = 50) -> Tuple[float, float]:
        """
        Find diffraction center by minimizing weighted standard deviation.
        
        Args:
            image_data: Processed diffraction pattern data
            radial_masks: Precomputed radial masks
            initial_guess: Initial center coordinates (x, y)
            max_iter: Maximum iterations for optimization
            
        Returns:
            tuple: Optimized center (x, y) and final weighted std sum
        """
        # Validate input shapes
        if self.processed_data.shape != radial_masks.image_shape:
            raise ValueError(
                f"Image shape {self.processed_data.shape} doesn't match "
                f"mask shape {radial_masks.image_shape}"
            )
        
        # Set initial_simplex for faster optimization (+delta vs default 5% pixel around initial guess)
        delta = 5.0
        x0 = np.array([initial_guess[0], initial_guess[1]])
        
        initial_simplex = np.array([
            x0,
            x0 + [delta, 0],
            x0 + [0, delta],
        ])

        # Perform optimization
        result = minimize(
            fun=self.calculate_std_sum_with_masks,
            x0=x0,
            args=(radial_masks,),
            method='Nelder-Mead', 
            options={
                'disp': True,
                'initial_simplex': initial_simplex,
                'maxiter': max_iter,
                'xatol': 1,  # Coordinate tolerance
            }
        )
        
        optimized_center = result.x
        self.center_pos = optimized_center

        return optimized_center

    def ring_centroid_debug(self,
                           ring_mask: RingMask,
                           initial_guess: Tuple[float, float],
                           max_iter: int = 10,
                           tolerance: float = 1.0) -> Tuple[float, float]:
        """
        Debug version of iterative_ring_centroid.
        """
        current_center = initial_guess
        
        for iteration in range(max_iter):
            
            print(f"Current guess: {current_center}")
            new_center = self.ring_centroid(ring_mask, current_center)
            # Check convergence
            shift_distance = np.sqrt((new_center[0] - current_center[0])**2 + 
                                    (new_center[1] - current_center[1])**2)
            print(f"Iteration {iteration+1}: Center shift = {shift_distance:.3f} pixels")

            current_center = new_center
            if shift_distance < tolerance:
                break
        
        self.center_pos = self.ring_centroid(ring_mask, current_center, save_total_count=True)
        return self.center_pos
    
    def azimuthal_average_bincount(self, 
                      radial_masks: RadialMasks,
                      precomputed_masks: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate azimuthal average using precomputed radial masks.
        
        This optimized version shifts the image data to align with the precomputed
        masks instead of recalculating masks for each image. This provides significant
        performance improvement when processing multiple images with the same dimensions.
        
        Args:
            radial_masks: Precomputed RadialMasks object
            radius: Maximum radius for azimuthal average (must match radial_masks.radius)
            num_bins: Number of radial bins (must match radial_masks.num_bins)
            
        Returns:
            tuple: Contains two arrays:
                - radial_positions: Bin center coordinates (num_bins,)
                - average_intensities: Azimuthally averaged intensities (num_bins,)
                
        Raises:
            ValueError: If processed_data or center_pos is not set
            ValueError: If image shape doesn't match mask shape
            ValueError: If radius or num_bins don't match precomputed masks
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available. Call remove_background() first.")
        
        if self.center_pos is None:
            raise ValueError("Center position not set. Call find_diffraction_center() first.")
        
        # Validate input parameters match precomputed masks
        if self.processed_data.shape != radial_masks.image_shape:
            raise ValueError(
                f"Image shape {self.processed_data.shape} doesn't match "
                f"mask shape {radial_masks.image_shape}"
            )
        
        # Calculate shift
        center_x, center_y = self.center_pos
        mask_center_x, mask_center_y = (
            radial_masks.image_shape[1] // 2, 
            radial_masks.image_shape[0] // 2
        )
        
        shift_x = int(round(center_x - mask_center_x))
        shift_y = int(round(center_y - mask_center_y))

        # Shift image data
        shifted_data = shift(self.processed_data, 
                            shift=(-shift_y, -shift_x), 
                            order=0,
                            mode='constant', 
                            cval=np.nan)
        
        # Flatten the data for fast indexing
        flat_data = shifted_data.ravel()
        
        # Preallocate results
        radial_average = np.full(precomputed_masks['num_bins'], np.nan)
        pixel_counts = np.zeros(precomputed_masks['num_bins'])
        
        # Only process valid bins
        valid_pixels_total = 0
        for i in precomputed_masks['valid_bins']:
            indices = precomputed_masks['flat_indices'][i]
            if len(indices) > 0:
                # Extract values using precomputed indices
                values = flat_data[indices]
                
                # Filter out NaN values efficiently
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) > 0:
                    radial_average[i] = np.mean(valid_values)
                    pixel_counts[i] = len(valid_values)
                    valid_pixels_total += len(valid_values)
            
        return radial_masks.bin_centers, radial_average

    def azimuthal_average_legacy(self, 
                               radius: float = 300,
                               num_bins: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy implementation of azimuthal average (original version).
        
        This is kept for backward compatibility and for cases where
        precomputed masks are not available.
        
        Args:
            radius: Maximum radius for azimuthal average in pixels
            num_bins: Number of radial bins
            
        Returns:
            tuple: (radial_positions, average_intensities)
        """
        if self.processed_data is None:
            raise ValueError("Processed data not available. Call remove_background() first.")
        
        if self.center_pos is None:
            raise ValueError("Center position not set. Call find_diffraction_center() first.")
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:self.processed_data.shape[0], 0:self.processed_data.shape[1]]
        
        # Calculate distances from center
        center_x, center_y = self.center_pos
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Create radial bins
        bin_edges = np.linspace(0, radius, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize arrays for results
        radial_average = np.zeros(num_bins)
        pixel_counts = np.zeros(num_bins)
        
        # Calculate azimuthal average
        for i in range(num_bins):
            # Create mask for current radial bin
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            
            if np.any(mask):
                radial_average[i] = np.mean(self.processed_data[mask])
                pixel_counts[i] = np.sum(mask)
        
        return bin_centers, radial_average
