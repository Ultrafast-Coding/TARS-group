import pandas as pd
import yaml
import re
from pathlib import Path
from typing import List

def csv_to_yaml_intensity(directory_path: str, output_yaml: str):
    directory = Path(directory_path)
    data_list = []
    
    # Regex to find 'xps' followed by numbers/dots
    # Matches intensity_profile_xps198.08000_I100.00000.csv
    pattern = re.compile(r"xps([-+]?\d*\.\d+|\d+)")

    # Iterate through all CSV files in the directory
    for filepath in directory.glob("*.csv"):
        match = pattern.search(filepath.name)
        
        if match:
            # Extract XPS value as a float
            xps_val = float(match.group(1))
            
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Extract columns as lists
            # Structure: [xps, [avg], [bin_number], [sem]]
            entry = [
                xps_val,
                df['average'].tolist(),
                df['bin_number'].tolist(),
                df['sem'].tolist()
            ]
            
            data_list.append(entry)
    
    # Optional: Sort the list by XPS value for better organization
    data_list.sort(key=lambda x: x[0])

    # # Save to YAML
    # with open(output_yaml, 'w') as f:
    #     yaml.dump(data_list, f, default_flow_style=None)
        
    # print(f"Successfully processed {len(data_list)} files into {output_yaml}")
    return data_list

# Example Usage:
# csv_to_yaml_intensity('./data_folder', 'combined_profiles.yaml')