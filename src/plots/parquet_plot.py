import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import List

class ParquetPlotting:
    def __init__(self):
        self.parquet_list = []

    def list_parquet(self):
        for filepath in self.parquet_list:
            print(filepath.name)

    def add_parquet(self, parquet_dir: str):
        parquet_path = Path(parquet_dir)
        self.parquet_list += [parquet_path]
        self.load_parquet(-1)
        
    def load_parquet(self, parquet_no: int):
        self.df = pd.read_parquet(self.parquet_list[parquet_no])
        self.path = self.parquet_list[parquet_no]

    def density_mapping(self, str_x: str, str_y: str):
        sns.set_style("whitegrid")

        try:
            df = self.df
            print(f'Using parquet file: {self.path.name}')
        except Exception as e:
            print('Failed to load plotting data, try calling load_parquet() first!')
        x = df[str_x]
        y = df[str_y]

        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[mask]
        y_valid = y[mask]
        # Calculate density with gaussian_kde
        kde = gaussian_kde(np.vstack([x_valid, y_valid]))
        density = kde(np.vstack([x_valid, y_valid]))

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=x_valid,
            y=y_valid,
            c=density,  
            cmap="viridis", 
            s=20,  # Dot size
            alpha=0.7,  # Opacity
            edgecolor=None,  # Dot without edge
            ax=ax
        )

        # Colorbar and labels
        cbar = plt.colorbar(scatter.collections[0], ax=ax)
        cbar.set_label("Point Density", fontsize=10)
        ax.set_title(f"Density Scatter Plot: {str_x} vs {str_y}", fontsize=14, pad=10)
        ax.set_xlabel(str_x, fontsize=12)
        ax.set_ylabel(str_y, fontsize=12)
        plt.tight_layout()
        plt.show()

    def heatmap_plot(self, radial_list: List[int]):
        try:
            df = self.df
            print(f'Using parquet file: {self.path.name}')
        except Exception as e:
            print('Failed to load plotting data, try calling load_parquet() first!')

        radial_columns = [f'radial_bin_{i:03d}' for i in radial_list]

        # Filter only columns that exist in the dataframe
        available_columns = [col for col in radial_columns if col in df.columns]
        print(f"Found columns: {available_columns}")

        # Select only the specified columns and drop rows with NaN in any of them
        df_selected = df[available_columns].dropna()

        print(f"Original rows: {len(df)}, After dropping NaN: {len(df_selected)}")
        print(f"Removed {len(df) - len(df_selected)} rows with NaN values")

        # Calculate correlation matrix
        correlation_matrix = df_selected.corr()

        # Create the heatmap
        plt.figure(figsize=(12, 10))

        display_num = len(radial_list) < 15
        # Create heatmap with annotations
        sns.heatmap(
            correlation_matrix,
            annot=display_num,            # Show correlation values in cells
            fmt='.2f',             # Format to 2 decimal places
            #cmap='coolwarm',       # Color map: blue (-1) to red (+1)
            #center=0,              # Center color map at 0
            square=True,           # Make cells square
            linewidths=0.5,        # Add lines between cells
            cbar_kws={'shrink': 0.8}  # Adjust color bar size
        )

        # Add title and adjust layout
        plt.title('Correlation Heatmap of Radial Bin Columns', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Show the plot
        plt.show()

    def parquet_to_csv(self, output_dir: str):
        try:
            df = self.df
            print(f'Using parquet file: {self.path.name}')
        except Exception as e:
            print('Failed to load plotting data, try calling load_parquet() first!')

        # Write the DataFrame to a CSV file
        df.to_csv(output_dir, index=False)
        
        print(f"Successfully converted and saved to: {output_dir}")
        