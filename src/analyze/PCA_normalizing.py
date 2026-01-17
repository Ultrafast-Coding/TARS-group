import pandas as pd
import numpy as np
import yaml
from pathlib import Path

class XPSAggregator:
    def __init__(self, output_dir: Path, max_bin: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_bin = max_bin

    def _get_radial_cols(self, df):
        return [f'radial_bin_{i:03d}' for i in range(self.max_bin)]

    def aggregate_directory(self, filtered_dir: Path, rad_min: int, rad_max: int):
        files = list(filtered_dir.glob("*xps_*.parquet"))
        results = {}

        for f in files:
            try:
                xps_val_str = f.stem.split('_')[-1]
                xps_key = f"xps_{xps_val_str}"
            except Exception:
                xps_key = f.stem

            df = pd.read_parquet(f)
            if df.empty:
                continue

            # Identify all columns (0-719)
            all_cols = self._get_radial_cols(df)
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
            print(f"Aggregated {f.name}: {num_samples} frames (Full 720 bins).")

        # Save to YAML
        output_file = self.output_dir / "xps_statistics_full.yaml"
        with open(output_file, 'w') as y_file:
            yaml.dump(results, y_file, default_flow_style=False, sort_keys=True)
        
        print(f"\nSaved full-range statistics to: {output_file}")