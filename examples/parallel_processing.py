import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path
from src.processor.expdir_processor import DirectoryProcessor

Expdata_folder = Path(r"E:\20251011\5_longscan_42deg_uv62p4deg_1\fist_AndorEMCCD")
Result_folder = Path(r"C:\Users\ab177\Desktop\diffraction_results\1011long")

# Initialize DirectoryProcessor
processor = DirectoryProcessor(
    result_directory=Result_folder,
    data_directory=Expdata_folder,
    xps_grouping_param='180.362	180.2121	179.67246	179.664965	179.65747	179.649975	\
        179.643979	179.639482	179.634985	179.630488	179.625991	179.621494	179.616997	\
        179.6125	179.608003	179.603506	179.599009	179.594512	179.590015	179.585518	\
        179.581021	179.575025	179.56753	179.560035	179.55254	179.545045	179.53755	\
        179.52256	179.50757	179.49258	179.47759	179.4626',  # str of xps values copied from data_acquisition_logfile.
    xray_removal_param=[15, 0.7],  # [beam_threshold, expansion_threshold_ratio]
    center_fitting_param=[70, 100, 721, 692],  # [inner_radius, outer_radius, center_x, center_y]
    azimuthal_avg_param=[720, 720],  # [radius, num_bins]
    background_directory="default",
    data_mask_directory="default"
)

# Process sequentially
#processor.process_in_sequence("analysis_in_sequence")

# Or process in parallel
processor.process_in_parallel(max_workers=8, analyze_no="analysis_logic_test")

# Load existing configuration
#processor.load_config("analysis_001")

#.\venv\Scripts\Activate
# git config --global user.email "xsr23@mails.tsinghua.edu.cn"
# git config --global user.name "Theory"

#2026-01-13 14:51:43,591 - ERROR - Failed to process E:\20251005\day2_61deg_longscan3\fist_AndorEMCCD\AndorEMCCD-1070_xps173.185000_scan7_labtime13-47-46p573931.tiff: index can't contain negative values