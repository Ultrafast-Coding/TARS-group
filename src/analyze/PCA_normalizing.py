import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import List

class XPSAggregator:
    def __init__(self, output_dir: Path, max_bin: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_bin = max_bin

    def _get_radial_cols(self, df):
        return [f'radial_bin_{i:03d}' for i in range(self.max_bin)]

    def aggregate_directory(self, filtered_dir: Path, rad_min: int, rad_max: int, cluster_list: List, result_name: str='full'):
        '''
        aggregate_directory Docstring
        
        :param filtered_dir: should be /filteredGMM
        :type filtered_dir: Path
        :param rad_min, rad_max: normalizing min/max radius 
        :type rad_min, rad_max: int
        :param cluster_list: list of cluster number to keep, input empty list to skip this
        :type cluster_list: List
        :param result_name: Used to distinguish clusters
        '''

        files = list(filtered_dir.glob("*auto_filtered_xps_*.parquet"))
        results = {}

        for f in files:
            try:
                xps_val_str = f.stem.split('_')[-1]
                xps_key = f"xps_{xps_val_str}"
            except Exception:
                xps_key = f.stem

            df_raw = pd.read_parquet(f)
            if df_raw.empty:
                continue

            # Identify all columns (0-719)
            all_cols = self._get_radial_cols(df_raw)

            if len(cluster_list) != 0:
                # We create a mask to keep only the selecterd clusters
                cluster_mask = df_raw['cluster_pass2'].isin(cluster_list)
                # Apply the mask to the dataframe
                df = df_raw[cluster_mask].copy()
                # Print the number of rows kept for transparency
                print(f"Keeping {len(df)} rows from clusters {cluster_list} (discarded {len(df_raw) - len(df)} ).")
            else:
                df = df_raw.copy()

            data_all = df[all_cols].values

            norm_cols = [f'radial_bin_{i:03d}' for i in range(rad_min,rad_max,4)]
            data_norm = df[norm_cols].values
            # --- Step 1: Global Normalization (All 720 bins) ---
            # We calculate the sum across the entire detector range
            row_sums = data_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            
            # Normalize the full 720-element arrays
            norm_data_all = data_all / row_sums

            # --- Step 2: Statistics on Full Range ---
            avg_profile = np.mean(norm_data_all, axis=0)
            std_profile = np.std(norm_data_all, axis=0)
            
            num_samples = len(df)
            sem_profile = std_profile / (avg_profile * np.sqrt(num_samples))

            results[xps_key] = {
                "metadata": {
                    "source_file": f.name,
                    "sample_count": int(num_samples),
                    "xps_value": float(xps_val_str) if xps_val_str.replace('.','').isdigit() else xps_val_str
                },
                "statistics": {
                    "avg": avg_profile.tolist(),
                    "std": std_profile.tolist(),
                    "sem": sem_profile.tolist()
                }
            }
            print(f"Aggregated {f.name}: {num_samples} frames.")

        # Save to YAML
        output_file = self.output_dir / f"xps_statistics_{result_name}.yaml"
        with open(output_file, 'w') as y_file:
            yaml.dump(results, y_file, default_flow_style=False, sort_keys=True)
        
        print(f"\nSaved full-range statistics to: {output_file}")