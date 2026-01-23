import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.analyze.radial_filter import RadialProfileFilter
from src.analyze.diffraction_normalizer import DiffractionNormalizer
# from Iavg_loader import load_intensity_profiles

analyze_directory = r"C:\Users\ab177\Desktop\diffraction_results\1013long\analysis_logic_test"

# Initialize the filter
filter_processor = RadialProfileFilter(analyze_directory)
# Run the filtering process
results = filter_processor.run_filtering(filter_type="MAD")

# Check results
for xps_dir, result in results.items():
    if result['success']:
        print(f"{xps_dir}: Processed successfully")
    else:
        print(f"{xps_dir}: Failed - {result['message']}")

# Initialize the normalizer
normalizer = DiffractionNormalizer(analyze_directory)
# Run the normalization process
results = normalizer.run_normalization(statistic_type="normal")

# Check individual results
for filename, result in results.items():
    if result['success']:
        print(f"{filename}: Processed successfully")
    else:
        print(f"{filename}: Failed - {result['message']}")

# Load all intensity profiles
# intensity_data = load_intensity_profiles(analyze_directory, 
#                                          std_check=True,
#                                          calibration_factor=0.024,
#                                          xps_0=198.08)

