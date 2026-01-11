import tifffile 
import numpy as np

from typing import List
from pathlib import Path
#automatically handles path separators across different operating systems.

def load_tiff_single(file_path: str) -> np.ndarray:
    """
    Load a TIFF file from the specified directory and return as numpy ndarray.
    
    Parameters:
    -----------
    file_path : str
        Path to the TIFF file
    
    Returns:
    --------
    np.ndarray
        The loaded TIFF image as a numpy array
    """
    tiff_path = Path(file_path)
    
    try:
        # Load and return the TIFF file
        return tifffile.imread(tiff_path)
    except Exception as e:
        print(e)

def load_tiff_batch(directory: str) -> List[np.ndarray]:
    """
    Load all TIFF files from the specified directory and return as list of numpy ndarrays.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing TIFF files
    
    Returns:
    --------
    List[np.ndarray]
        List of loaded TIFF images as numpy arrays
    """
    # Convert to Path object for better handling
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    
    # Find all TIFF files in directory
    file_paths = []
    for ext in ['*.tiff', '*.tif']:
        file_paths.extend(dir_path.glob(ext))
    if not file_paths:
        raise FileNotFoundError(f"No TIFF files found in {directory}")
    print(f"Found {len(file_paths)} TIFF files")
    
    # Load all TIFF files
    image_list = []
    for file_path in file_paths:
        try:
            image = tifffile.imread(file_path)
            image_list.append(image)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    return image_list

def save_as_tiff(array: np.ndarray, 
                 dirpath: str,
                 filename: str,
                 compress: bool = True) -> None:
    """
    Save a numpy ndarray as a TIFF file.
    
    Parameters:
    -----------
    array : np.ndarray
        The numpy array to save as TIFF
    filepath : str or Path
        Path where the TIFF file will be saved
    metadata : dict, optional
        Additional metadata to include in TIFF file
    compress : bool, optional
        Whether to use compression (default: True)
    
    Raises:
    -------
    ValueError
        If array is not a numpy ndarray or is empty
    IOError
        If file cannot be written
    """
    # Validate input array
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Input must be a numpy ndarray, got {type(array)}")
    
    if array.size == 0:
        raise ValueError("Cannot save empty array")
    
    # Convert to Path object and handle file extension
    filepath = Path(dirpath) / filename
    
    # Ensure .tiff extension
    if filepath.suffix.lower() not in ['.tiff', '.tif']:
        filepath = filepath.with_suffix('.tiff')
    
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare TIFF options
        tiff_options = {}
        
        if compress:
            tiff_options['compression'] = 'lzw'  # Lossless compression
        
        # Save the array
        tifffile.imwrite(filepath, array, **tiff_options)
        
        print(f"Successfully saved array to: {filepath}")
        print(f"  - Shape: {array.shape}")
        print(f"  - Data type: {array.dtype}")
        print(f"  - File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        raise IOError(f"Failed to save TIFF file '{filepath}': {e}") from e