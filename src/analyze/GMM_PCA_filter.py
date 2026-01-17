import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class XPSFilter:
    def __init__(self, n_pca=3):
        self.n_pca = n_pca
        
    def _get_radial_cols(self, df):
        return [f'radial_bin_{i:03d}' for i in range(0,255,8)]

    def _apply_gmm(self, df, n_comp, label_suffix):
        """Internal helper to apply PCA + GMM logic."""
        radial_cols = self._get_radial_cols(df)

        # --- SAFETY CHECK: Remove NaN rows ---
        initial_rows = len(df)
        # Drop rows only if NaN exists in the specific radial columns
        df = df.dropna(subset=radial_cols).copy()
        removed_count = initial_rows - len(df)
        
        if removed_count > 0:
            print(f"Safety Check [{label_suffix}]: Removed {removed_count} rows containing NaN values.")

        # Ensure we still have enough data to perform clustering
        if len(df) < n_comp:
            raise ValueError(f"Not enough data rows ({len(df)}) to form {n_comp} clusters.")

        data = df[radial_cols].values
        
        # Scaling and PCA
        scaled = StandardScaler().fit_transform(data)
        pca_features = PCA(n_components=self.n_pca).fit_transform(scaled)
        
        gmm = GaussianMixture(n_components=n_comp, random_state=42)
        labels = gmm.fit_predict(pca_features)
        
        # --- CONSISTENCY FIX: Sort by Size ---
        # Find the mapping from old labels to new labels based on count
        unique, counts = np.unique(labels, return_counts=True)
        # Sort indices by counts descending
        sorted_indices = np.argsort(-counts) 
        # Create a mapping dictionary: {old_label: new_rank}
        label_map = {old: new for new, old in enumerate(sorted_indices)}
        
        # Apply the mapping
        labels = np.array([label_map[l] for l in labels])
        # -------------------------------------

        df[f'cluster_{label_suffix}'] = labels
        return df, pca_features, labels

    def process_with_manual_filter(self, file_path: Path):
        """One-file testing mode: lets you choose params interactively."""
        df = pd.read_parquet(file_path)
        original_count = len(df)

        # PASS 1
        n1 = int(input("Pass 1: How many components (n1)? "))
        df, pca1, labels1 = self._apply_gmm(df, n1, "pass1")
        self.visualize_gmm(pca1, labels1, f"{file_path.name} - Pass 1")
        
        print(df['cluster_pass1'].value_counts(normalize=True) * 100)
        discard1 = input("Pass 1: Cluster IDs to DISCARD (e.g. 0,1): ")
        
        if discard1:
            ids = [int(x.strip()) for x in discard1.split(',')]
            df = df[~df['cluster_pass1'].isin(ids)].copy()

        # PASS 2
        n2 = int(input("Pass 2: How many components (n2)? "))
        df, pca2, labels2 = self._apply_gmm(df, n2, "pass2")
        self.visualize_gmm(pca2, labels2, f"{file_path.name} - Pass 2")
        
        print(df['cluster_pass2'].value_counts(normalize=True) * 100)
        discard2 = input("Pass 2: Cluster IDs to DISCARD: ")

        if discard2:
            ids = [int(x.strip()) for x in discard2.split(',')]
            df = df[~df['cluster_pass2'].isin(ids)].copy()

        print(f"\nFinal retention: {len(df)}/{original_count} ({(len(df)/original_count)*100:.1f}%)")
        return df

    def batch_filter_directory(self, input_dir: Path, output_dir: Path, n1, discard1, n2, discard2):
        """Applies a fixed 2-pass filter to all parquet files in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        files = list(input_dir.glob("xps_*.parquet"))
        log_data = []

        print(f"Batch processing {len(files)} files...")

        for f in files:
            df_orig = pd.read_parquet(f)
            count_0 = len(df_orig)

            # Pass 1
            df_1, _, _ = self._apply_gmm(df_orig, n1, "pass1")
            df_1 = df_1[~df_1['cluster_pass1'].isin(discard1)].copy()
            count_1 = len(df_1)

            # Pass 2
            df_2, _, _ = self._apply_gmm(df_1, n2, "pass2")
            df_2 = df_2[~df_2['cluster_pass2'].isin(discard2)].copy()
            count_2 = len(df_2)

            # Save and Log
            df_2.to_parquet(output_dir / f"filtered_{f.name}")
            
            log_entry = {
                "file": f.name,
                "initial": count_0,
                "after_pass1": count_1,
                "after_pass2": count_2,
                "retention_pct": (count_2 / count_0) * 100 if count_0 > 0 else 0
            }
            log_data.append(log_entry)
            print(f"Processed {f.name}: {count_2}/{count_0} left.")

        # Save log to CSV
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(output_dir / "filtering_log.csv", index=False)
        print("\nBatch Processing Complete. Log saved to filtering_log.csv")

    def auto_batch_filter(self, input_dir: Path, output_dir: Path, n1: int, n2: int, threshold=20.0):
        """
        Automatically filters all files. 
        Discards clusters representing < threshold % of the data in each pass.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        files = list(input_dir.glob("xps_*.parquet"))
        log_entries = []

        for f in files:
            df = pd.read_parquet(f)
            initial_count = len(df)
            
            # --- PASS 1 ---
            df, _, labels1 = self._apply_gmm(df, n1, "pass1")
            counts1 = df['cluster_pass1'].value_counts(normalize=True) * 100
            # Keep clusters where % >= threshold
            keep_ids1 = counts1[counts1 >= threshold].index.tolist()
            df = df[df['cluster_pass1'].isin(keep_ids1)].copy()
            count_after_p1 = len(df)

            # --- PASS 2 ---
            if count_after_p1 > n2:
                df, _, labels2 = self._apply_gmm(df, n2, "pass2")
                counts2 = df['cluster_pass2'].value_counts(normalize=True) * 100
                keep_ids2 = counts2[counts2 >= threshold].index.tolist()
                df = df[df['cluster_pass2'].isin(keep_ids2)].copy()
            
            count_final = len(df)

            # Save and Log
            df.to_parquet(output_dir / f"auto_filtered_{f.name}")
            log_entries.append({
                "file": f.name,
                "initial": initial_count,
                "p1_kept_ids": keep_ids1,
                "p2_kept_ids": keep_ids2 if count_after_p1 > n2 else "N/A",
                "final_count": count_final,
                "retention": (count_final / initial_count) * 100
            })
            print(f"File {f.name}: Retained {count_final} samples ({log_entries[-1]['retention']:.1f}%)")

        # Save Log
        pd.DataFrame(log_entries).to_csv(output_dir / "auto_filter_log.csv", index=False)
        print("\nAuto-batch processing complete.")

    def visualize_gmm(self, pca_features, labels, title):
        plt.figure(figsize=(6, 4))
        # Plot against PC2 and PC3 since PC1 was most likely intensity
        plt.scatter(pca_features[:, 1], pca_features[:, 2], c=labels, cmap='viridis', s=5)
        plt.title(title)
        plt.colorbar(label="Cluster ID")
        plt.show()

