import matplotlib.pyplot as plt
import numpy as np

def plot_ndarray(Bkg: np.ndarray, Vmin: int, Vmax: int):

    plt.figure(figsize=(10, 8))
    # Create the plot
    im = plt.imshow(Bkg, cmap='viridis', aspect='equal',
        vmin=Vmin,  # <--- Manual minimum colorbar value
        vmax=Vmax ) 
    # Add colorbar
    plt.colorbar(im, label='Intensity')
    
    # Add labels and title
    plt.title('Diffraction Image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

def plot_azimuthal_average(radii, intensities, pixel_to_q: int = 0.024):
    """
    Plot the azimuthal average.
    """
    plt.figure(figsize=(8, 6))
    q_range = [i*pixel_to_q for i in radii]
    plt.plot(q_range, intensities, colour= 'SlateGray', label='Azimuthal Average')

    # Add labels and title
    plt.xlabel('Radial Distance (q)')
    plt.ylabel('Average Intensity')
    plt.title('Azimuthal Average Profile')
    plt.grid(True, axis='x')
    plt.legend()
    plt.show()