import numpy as np
import matplotlib.pyplot as plt

def plot_waterfall_diffraction(data: list, scale: float, width_to_height_ratio: float, filename: str = ''):
    """
    Generates the final customized waterfall plot of Percentage Difference (PD).

    Incorporates fixed width, flipped XPS order, aligned baselines, and specific 
    y-axis range calculation.

    Args:
        data: list of [xps_value, [intensity values], [radial distance values], [sem, optional]].
        xps_0: The reference XPS value for time calculation.
        pixel_to_q: Factor to convert radial distance to q (x-axis).
        scale: Vertical scaling factor. Used for PD labels' step size (e.g., 1.0 = 1% step).
        width_to_height_ratio: Desired ratio of the plot width to height.
        filename: Name for the saved plot image file. waterfall_plot_final.png
    """
    if not data or len(data) < 2:
        print("Error: The data list is empty or has fewer than two groups, for static data call static plotting instead.")
        return

    # 1. Sort data by XPS value and Reverse Order (Biggest XPS at the bottom, i=0)
    sorted_data = sorted(data, key=lambda x: x[0], reverse=False)
    
    # 2. Identify and calculate Background (I_0) from the two largest XPS groups (now at the end)
    bg_group_1 = sorted_data[0] 
    bg_group_2 = sorted_data[1]
    I_0 = (np.array(bg_group_1[1]) + np.array(bg_group_2[1])) / 2
    xps_bg_1, xps_bg_2 = bg_group_1[0], bg_group_2[0]

    # 3. Setup Figure Dimensions (Width fixed at 8)
    plot_width = 8
    plot_height = plot_width / width_to_height_ratio
    
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    time_tick_positions = []
    
    # Define the constant vertical step in %PD for visual separation (e.g., 4% PD separation)
    vertical_step_PD = 1.0 * scale 

    colors = plt.cm.gist_earth(np.linspace(0.05, 0.8, 8))

    # 4. Process and Plot Each Group
    num_curves = len(sorted_data)
    for i, group in enumerate(sorted_data):
        xps_value = group[0]
        I = np.array(group[1])
        radial_distance = np.array(group[2])

        # Calculate PD
        with np.errstate(divide='ignore', invalid='ignore'):
            PD = np.divide(I - I_0, I_0, out=np.zeros_like(I_0, dtype=float), where=I_0!=0) * 100
        
        q = radial_distance
        
        # Vertical offset is proportional to the index 'i'
        vertical_offset = i * vertical_step_PD
        PD_curve = PD + vertical_offset
        
        curve_color = colors[i % len(colors)]
        # Plot the curve
        ax.plot(q, PD_curve, color=curve_color, linewidth=1.2)

        if has_std_dev := (len(group) > 3):
            std_dev = np.array(group[3]) * np.sqrt(2) * 100  # Estimate std of A-B/A = sqrt(2)*std(A)/avg(A)
            # Calculate upper and lower bounds for 3σ (68.3% confidence)
            upper_bound = PD + 3 * std_dev + vertical_offset
            lower_bound = PD - 3 * std_dev + vertical_offset
            # Plot the shaded area for ±3σ
            ax.fill_between(
                q,               
                lower_bound,         
                upper_bound,         
                color="#9C8C9B",   
                alpha=0.1,           # transparency level
            )
        
        # ALIGNMENT: Draw the horizontal baseline (PD=0) at the curve's offset position
        #ax.axhline(vertical_offset, color='gray', linestyle='--', linewidth=0.8, zorder=0)

        time_tick_positions.append(vertical_offset)

    
    # 5. Set Y-Axis Limits (to leave room for one extra offset on top and bottom)
    
    # The total vertical range of the plotted data is:
    # Max Y: (Max index * vertical_step_PD) + Max positive PD deviation
    max_data_y = time_tick_positions[-1] 
    # Min Y: (Min index * vertical_step_PD) + Min negative PD deviation
    min_data_y = time_tick_positions[0]  # Since time_tick_positions[0] is 0
    
    # Add one extra vertical step to the top and bottom limits
    y_limit_top = max_data_y + vertical_step_PD
    y_limit_bottom = min_data_y - vertical_step_PD
    
    ax.set_ylim(y_limit_bottom, y_limit_top)

    
    # 6. Set up the Right-Hand (Time) Axis
    ax_time = ax.twinx()
    ax_time.set_ylim(ax.get_ylim()) 

    # Calculate time
    time_labels = [f"{group[0]:.3f} ps" for group in sorted_data]
    ax_time.set_yticks(time_tick_positions)
    ax_time.set_yticklabels(time_labels, fontsize=10, ha='left', va='center') # va='center' ensures perfect alignment with baseline
    ax_time.tick_params(axis='y', length=0)
    
    
    # 7. Set up the Left-Hand (PD) Axis Ticks
    
    # The PD labels should be placed at the baseline of each curve (vertical_offset)
    # The labels should be 0%, (N-1)*scale %, (N-1)*scale*2 % ...
    
    # Calculate the label value for each baseline (offset)
    # PD Label at i=0 is 0%.
    # PD Label at i=1 is vertical_step_PD * scale % (if scale is 1, it's 4%)
    # PD Label at i=2 is vertical_step_PD * 2 * scale % (if scale is 1, it's 8%)
    
    # NOTE: Since the curves are already shifted by vertical_step_PD, 
    # the labels should reflect the CUMULATIVE SHIFT.
    
    # PD labels for each baseline: 0%, 4%, 8%, ...
    # The `vertical_offset` array already represents the baseline positions (0, 4, 8, ...)
    pd_baseline_labels = [f"{v:.0f}\\%" for v in time_tick_positions]

    ax.set_yticks(time_tick_positions)
    ax.set_yticklabels(pd_baseline_labels)
    ax.tick_params(axis='y', which='major', length=5) 
    
    # 8. Set Labels and Title
    title = (f"Waterfall Plot of Percentage Difference (PD) - Aligned\n"
             f"Background $I_0$ averaged from $XPS$ at ${xps_bg_1}$ and ${xps_bg_2}$ (Reversed Order)")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$q$ (Normalized Radial Distance)', fontsize=12)
    ax.set_ylabel('Percentage Difference ($\%PD$)', fontsize=12, loc='top')
    
    ax.grid() # Turn off all default grid lines
    plt.tight_layout()

    if filename != '':
        plt.savefig(filename)
        print(f"Final aligned waterfall plot saved as '{filename}'")


