import yaml
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import xarray as xr

# The energy were actually delay, too lazy to fix

@dataclass
class XPSDataSet:
    """Represents one loaded YAML file / dataset"""
    energies: List[float]           # XPS binding energies
    avgs: List[List[float]]         # list of average intensity lists
    sems: List[List[float]]         # list of standard error lists
    stds: List[List[float]]         # list of standard deviation lists
    source_files: List[str]         # which yaml file each energy came from


class XPSIntensityMultiLoader:
    """
    Loads multiple YAML files containing XPS intensity statistics.
    
    Expected YAML structure per entry:
    xps_XXX.XXXXX:
      metadata:
        sample_count: ...
        source_file: ...
        xps_value: ...
      statistics:
        avg: [float, float, ...]
        sem: [float, float, ...]
        std: [float, float, ...]
        # (optionally other keys)
    
    Output format:
    List of [energy, avg_list, x_positions, sem_list]
    where x_positions = (np.arange(len(avg)) + 1) * step
    """
    
    def __init__(self, step: float = 1.0):
        self.step = step
        self.all_data: List[Tuple[float, List[float], List[float], List[float]]] = []
        self.datasets: List[XPSDataSet] = []
    
    def load_yaml(self, yaml_path: str, xps0: float) -> None:
        """Load one YAML file and append its data"""

        C0 = 299792458
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            if not isinstance(raw_data, dict):
                raise ValueError("YAML root must be a dictionary")
                
            energies = []
            avgs = []
            sems = []
            stds = []
            source_files = []
            
            for energy_str, entry in raw_data.items():
                try:
                    xps_value = float(energy_str.replace('xps_', ''))
                    energy = (xps0 - xps_value) / (C0 * 0.5 * 10 ** -9)
                except ValueError:
                    print(f"Warning: Skipping invalid energy key: {energy_str}")
                    continue
                    
                stats = entry.get('statistics', {})
                
                # Get the three required arrays - must exist and have same length
                avg_list = stats.get('avg')
                sem_list = stats.get('sem')
                std_list = stats.get('std')
                
                if not all(isinstance(lst, list) for lst in [avg_list, sem_list, std_list]):
                    print(f"Warning: Missing or invalid stats for energy {energy:.5f} in {yaml_path}")
                    continue
                    
                lengths = [len(lst) for lst in [avg_list, sem_list, std_list]]
                if len(set(lengths)) != 1:
                    print(f"Warning: Length mismatch in stats for energy {energy:.5f} in {yaml_path}")
                    continue
                    
                energies.append(energy)
                avgs.append(avg_list)
                sems.append(sem_list)
                stds.append(std_list)
                source_files.append(yaml_path)
            
            if energies:
                dataset = XPSDataSet(
                    energies=sorted(energies),
                    avgs=avgs,
                    sems=sems,
                    stds=stds,
                    source_files=source_files
                )
                self.datasets.append(dataset)
                print(f"Loaded {len(energies)} spectra from {yaml_path}")
                
        except Exception as e:
            print(f"Error loading {yaml_path}: {e}")
    
    def load_multiple(self, yaml_paths: List) -> None:
        """Load multiple YAML files at once"""
        for path, xps0 in yaml_paths:
            self.load_yaml(path, xps0)
    
    def get_combined_data(self) -> List[List]:
        """
        Returns list in the requested format:
        [[energy1, [avg1, avg2, ...], [x1, x2, ...], [sem1, sem2, ...]], ...]
        Sorted by energy
        """
        combined = []
        
        # Collect everything
        all_entries = []
        for dataset in self.datasets:
            for i, energy in enumerate(dataset.energies):
                avg = dataset.avgs[i]
                n_points = len(avg)
                x_pos = (np.arange(n_points) + 1) * self.step
                sem = dataset.sems[i]
                
                all_entries.append((energy, avg, x_pos.tolist(), sem))
        
        # Sort by energy
        all_entries.sort(key=lambda x: x[0])
        
        # Convert to final format
        for energy, avg, x_pos, sem in all_entries:
            combined.append([energy, avg, x_pos, sem])
            
        return combined
    
    def get_combined_data_baseline(self) -> List[List]:
        """
        Returns list in the requested format:
        [[energy1, [avg1, avg2, ...], [x1, x2, ...], [sem1, sem2, ...]], ...]
        Sorted by energy
        """
        combined = []
        
        # Collect everything
        all_entries = []
        for dataset in self.datasets:
            for i, energy in enumerate(dataset.energies):
                avg = dataset.avgs[i]
                n_points = len(avg)
                x_pos = (np.arange(n_points) + 1) * self.step
                sem = dataset.sems[i]

                bscale = 0
                for normi in range(400,500):
                    bscale += avg[normi]
                for k in range(n_points):
                    avg[k] /= bscale
                
                all_entries.append((energy, avg, x_pos.tolist(), sem))
        
        # Sort by energy
        all_entries.sort(key=lambda x: x[0])
        
        # Convert to final format
        for energy, avg, x_pos, sem in all_entries:
            combined.append([energy, avg, x_pos, sem])
            
        return combined
    
    def summary(self) -> str:
        total_spectra = sum(len(ds.energies) for ds in self.datasets)
        total_files = len(self.datasets)
        return (f"Loaded {total_spectra} XPS spectra from {total_files} YAML files\n"
                f"Step size: {self.step}\n"
                f"Energy range: {self.get_energy_range_str()}")
    
    def get_energy_range_str(self) -> str:
        if not self.datasets:
            return "no data"
        all_energies = [e for ds in self.datasets for e in ds.energies]
        return f"{min(all_energies):.3f} – {max(all_energies):.3f} eV"

def convert_to_xarray(
    data_list: List[List],
    energy_dim_name: str = "time",
    spectral_dim_name: str = "spectral",
    intensity_var_name: str = "data",
    sem_var_name: str = "sem"
) -> xr.Dataset:
    """
    Convert list of [[energy, avg_list, x_pos_list, sem_list], ...]
    into an xarray Dataset with dimensions [time, spectral].
    
    Parameters
    ----------
    data_list : List[List]
        Format: [[energy, [avg...], [x_pos...], [sem...]], ...]
        Usually already sorted by energy
    energy_dim_name : str, optional
        Name of the energy/time dimension (default: "time")
    spectral_dim_name : str, optional
        Name of the spectral/x_position dimension (default: "spectral")
    intensity_var_name : str, optional
        Name of the main intensity variable (default: "data")
    sem_var_name : str, optional
        Name of the standard error variable (default: "sem")

    Returns
    -------
    xr.Dataset
        Dataset with variables 'data' and 'sem', coordinates 'time' and 'spectral'
    """
    if not data_list:
        raise ValueError("Input data_list is empty")

    # Extract components
    energies = []
    intensities = []
    positions = []
    # sems = []

    for item in data_list:
        # if len(item) != 4:
        #     raise ValueError(f"Each item must have exactly 4 elements: [energy, avg_list, x_pos_list, sem_list]")
        
        energy, avg_vals, x_pos_vals, sem_vals = item
        
        # Basic validation
        n = len(avg_vals)
        if not (len(x_pos_vals) == n and len(sem_vals) == n):
            raise ValueError(f"Length mismatch at energy {energy}: "
                            f"avg={len(avg_vals)}, x_pos={len(x_pos_vals)}, sem={len(sem_vals)}")
        
        energies.append(energy)
        intensities.append(avg_vals)
        positions.append(x_pos_vals)
        # sems.append(sem_vals)

    # Convert to numpy arrays
    energies = np.array(energies)
    intensities = np.array(intensities)     # shape: (n_energies, n_points)
    # sems = np.array(sems)

    # Check if x_positions are consistent across all spectra
    first_xpos = np.array(positions[0])
    for p in positions[1:]:
        if not np.allclose(p, first_xpos, atol=1e-8):
            print("Warning: x_positions are not identical across energies. "
                  "Using the first one as reference.")

    # Use the x_positions from the first spectrum (or you could take mean/median)
    spectral_coords = first_xpos

    # Create DataArray for intensities
    da_intensity = xr.DataArray(
        data=intensities,
        dims=[energy_dim_name, spectral_dim_name],
        coords={
            energy_dim_name: energies,
            spectral_dim_name: spectral_coords
        },
        name=intensity_var_name
    )

    # # Create DataArray for SEM
    # da_sem = xr.DataArray(
    #     data=sems,
    #     dims=[energy_dim_name, spectral_dim_name],
    #     coords={
    #         energy_dim_name: energies,
    #         spectral_dim_name: spectral_coords
    #     },
    #     name=sem_var_name
    # )

    # # Combine into Dataset
    # ds = xr.merge([da_intensity, da_sem])

    # # Optional: sort by energy just to be safe
    # ds = ds.sortby(energy_dim_name)

    return da_intensity

# ──────────────────────────────────────────────────────────────────────────────
# Example usage:
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = XPSIntensityMultiLoader(step=0.5)  # e.g. 0.5 eV per point, adjust as needed
    
    # Load one or many files
    loader.load_multiple([
        "xps_low_energy.yaml",
        "xps_mid_energy.yaml",
        "xps_high_energy.yaml"
    ])
    
    print(loader.summary())
    
    # Get the final data structure you wanted
    data = loader.get_combined_data()
    
    # Example: print first spectrum
    if data:
        energy, avg, x_pos, sem = data[0]
        print(f"\nFirst spectrum at {energy:.5f} eV:")
        print(f"  x_positions: {x_pos[:5]} ...")
        print(f"  avg:         {avg[:5]} ...")
        print(f"  sem:         {sem[:5]} ...")


