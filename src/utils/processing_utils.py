import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class ProcessingConfig:
    """Dataclass to store all processing parameters."""
    result_directory: str
    background_directory: str
    data_mask_directory: str
    xps_grouping_param: List[float]  # [threshold, tolerance]
    xray_removal_param: List[float]  # [chunk_size, sigma_threshold, beam_threshold]
    center_fitting_param: List[float]  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param: List[float]  # [radius, num_bins]
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

# External function to save configuration
def save_processing_config(config: ProcessingConfig, save_path: Path) -> None:
    """
    Save processing configuration to file.
    
    Args:
        config: ProcessingConfig dataclass instance
        save_path: Path to save configuration
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    config_df = pd.DataFrame([asdict(config)])
    config_df.to_csv(save_path, index=False)
    
    # Also save as JSON
    '''json_path = save_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)'''

# External function to load configuration
def load_processing_config(load_path: Path) -> ProcessingConfig:
    """
    Load processing configuration from file.
    
    Args:
        load_path: Path to configuration file
        
    Returns:
        ProcessingConfig dataclass instance
    """
    if not load_path.exists():
        raise FileNotFoundError(f"Config file not found: {load_path}")
    
    config_df = pd.read_csv(load_path)
    config_dict = config_df.iloc[0].to_dict()
    
    return ProcessingConfig.from_dict(config_dict)

class ProcessedResult:
    """Container for processed image results."""
    def __init__(self, 
                 filename: str,
                 center: Tuple[float, float],
                 radial_profile: np.ndarray,
                 xps_value: float):
        self.filename = filename
        self.center = center
        self.radial_profile = radial_profile
        self.xps_value = xps_value
