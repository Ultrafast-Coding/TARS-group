import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path
from src.processor.expdir_processor import DirectoryProcessor

Expdata_folder = Path(r"E:\20251005\day2_61deg_longscan3\fist_AndorEMCCD")
Result_folder = Path(r"C:\Users\ab177\Desktop\diffraction_results\test")

# Initialize DirectoryProcessor
processor = DirectoryProcessor(
    result_directory=Result_folder,
    data_directory=Expdata_folder,
    xps_grouping_param=[173.1845,173.0346,172.50995,172.49496,
                        172.47997,172.46498,172.44999,172.435,
                        172.42001,172.40502,172.39003,172.37504,
                        172.36005,172.34506,172.31508,172.2851],  # [threshold, tolerance]
    xray_removal_param=[15, 0.7],  # [beam_threshold, expansion_threshold_ratio]
    center_fitting_param=[60, 120, 542, 485],  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param=[720, 720],  # [radius, num_bins]
    background_directory="default",
    data_mask_directory="default"
)

# Process sequentially
#processor.process_in_sequence("analysis_in_sequence")

# Or process in parallel
processor.process_in_parallel(max_workers=8, analyze_no="analysis_center_test")

# Load existing configuration
#processor.load_config("analysis_001")

#.\venv\Scripts\Activate
# git config --global user.email "xsr23@mails.tsinghua.edu.cn"
# git config --global user.name "Theory"

#2026-01-13 14:51:43,591 - ERROR - Failed to process E:\20251005\day2_61deg_longscan3\fist_AndorEMCCD\AndorEMCCD-1070_xps173.185000_scan7_labtime13-47-46p573931.tiff: index can't contain negative values