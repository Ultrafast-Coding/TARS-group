import re
from typing import List, Dict, Tuple
from pathlib import Path

def group_tiff_files_by_xps(directory: str) -> List[List[str]]:
    """
    Group TIFF files by their XPS values extracted from filenames.
    
    All files in the directory are TIFF files with naming pattern like:
    "AndorEMCCD-2_xps188.975000_scan1_labtime22-54-12p146934"
    
    Args:
        directory: Path to directory containing TIFF files
        
    Returns:
        List of file groups, where each group contains filenames with same XPS value.
        Groups are sorted by XPS value in ascending order.
        
    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: If no TIFF files found or filename pattern doesn't match
        
    Example:
        >>> groups = group_tiff_files_by_xps("/path/to/tiff/files")
        >>> print(f"Found {len(groups)} XPS groups")
        >>> for i, group in enumerate(groups):
        ...     xps_value = extract_xps_value(group[0])
        ...     print(f"Group {i}: XPS={xps_value}, {len(group)} files")
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Get all TIFF files
    tiff_files = list(dir_path.glob("*.tiff")) + list(dir_path.glob("*.tif"))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {directory}")
    
    # Group files by XPS value
    xps_groups = {}
    
    for file_path in tiff_files:
        filename = file_path.name
        xps_value = extract_xps_value(filename)
        
        if xps_value is not None:
            if xps_value not in xps_groups:
                xps_groups[xps_value] = []
            xps_groups[xps_value].append(str(file_path))
        else:
            print(f"Warning: Could not extract XPS value from {filename}")
    
    if not xps_groups:
        raise ValueError("No valid XPS values found in filenames")
    
    # Sort groups by XPS value and return as list of lists
    sorted_xps_values = sorted(xps_groups.keys())
    sorted_groups = [xps_groups[xps] for xps in sorted_xps_values]
    
    print(f"Grouped {len(tiff_files)} files into {len(sorted_groups)} XPS groups")
    
    return sorted_groups

def extract_xps_value(filename: str) -> float:
    """
    Extract XPS value from filename.
    
    Args:
        filename: Filename string containing XPS pattern
        
    Returns:
        XPS value as float, or None if pattern not found
    """
    # Pattern to match '_xps188.975000_' 
    pattern = r'_xps([0-9]+\.[0-9]+)_'
    match = re.search(pattern, filename)
    
    if match:
        return float(match.group(1))
    # In case of no matching pattern, return 0 (..._noxps0_...)
    return 0

def group_tiff_files_with_info(directory: str) -> List[Tuple[float, List[str]]]:
    """
    Group TIFF files by XPS values and return with XPS information.
    
    Args:
        directory: Path to directory containing TIFF files
        
    Returns:
        List of tuples: (xps_value, list_of_filenames)
        Sorted by XPS value in ascending order
    """
    dir_path = Path(directory)
    tiff_files = list(dir_path.glob("*.tiff")) + list(dir_path.glob("*.tif"))
    
    xps_groups = {}
    
    for file_path in tiff_files:
        filename = file_path.name
        xps_value = extract_xps_value(filename)
        
        if xps_value is not None:
            if xps_value not in xps_groups:
                xps_groups[xps_value] = []
            xps_groups[xps_value].append(str(file_path))
    
    # Sort by XPS value and return as (xps_value, filenames) tuples
    sorted_groups = sorted([(xps, files) for xps, files in xps_groups.items()])
    
    return sorted_groups

def print_group_summary(groups: List[List[str]]):
    """
    Print summary information about the grouped files.
    
    Args:
        groups: List of file groups from group_tiff_files_by_xps()
    """
    print("XPS Group Summary:")
    print("=" * 50)
    
    for i, group in enumerate(groups):
        xps_value = extract_xps_value(group[0])
        print(f"Group {i:2d}: XPS = {xps_value:10.6f}, Files = {len(group):3d}")
        
        # Show first few filenames as example
        if len(group) <= 3:
            for f in group:
                print(f"         {Path(f).name}")
        else:
            for f in group[:2]:
                print(f"         {Path(f).name}")
            print(f"         ... and {len(group)-2} more files")
        
        if i < len(groups) - 1:  # Don't print separator after last group
            print("-" * 50)

# Alternative version using the tuple format
def print_detailed_summary(groups_with_xps: List[Tuple[float, List[str]]]):
    """
    Print detailed summary for groups in tuple format.
    
    Args:
        groups_with_xps: List from group_tiff_files_with_info()
    """
    print("Detailed XPS Group Summary:")
    print("=" * 60)
    
    for xps_value, files in groups_with_xps:
        print(f"XPS: {xps_value:12.6f} | Files: {len(files):3d}")
        
        # Show file count by unique patterns (optional)
        patterns = {}
        for f in files:
            # Extract part after XPS for pattern analysis
            name = Path(f).name
            pattern_match = re.search(r'_xps[0-9.]+_(.*)', name)
            if pattern_match:
                pattern = pattern_match.group(1)
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        if len(patterns) > 1:
            print(f"       Patterns: {len(patterns)} different naming patterns")
    
    print("=" * 60)
    total_files = sum(len(files) for _, files in groups_with_xps)
    print(f"Total: {len(groups_with_xps)} XPS groups, {total_files} files")

def merge_xps_groups_strategy(groups_with_xps: List[Tuple[float, List[str]]],
                                threshold: int = 350,
                                tolerance: float = 0.0025) -> List[Tuple[float, List[str]]]:
    """
    Merge XPS groups using a threshold-based strategy.
    
    Strategy:
    1. For groups with files > threshold: 
       - If separated by > tolerance, start new group
       - Otherwise, merge and start new group
    2. For groups with files <= threshold:
       - Merge into the nearest existing group
    
    Args:
        groups_with_xps: Output from group_tiff_files_with_info()
        threshold: File count threshold for considering a group as 'large'
        tolerance: XPS difference tolerance for merging large groups
        
    Returns:
        List of merged groups with weighted average XPS values
    """
    if not groups_with_xps:
        return []
    
    # Sort groups by XPS value
    sorted_groups = sorted(groups_with_xps, key=lambda x: x[0])
    
    # Separate large and small groups
    large_groups = []
    small_groups = []
    
    for xps, files in sorted_groups:
        if len(files) > threshold:
            large_groups.append((xps, files))
        else:
            small_groups.append((xps, files))
    
    print(f"Found {len(large_groups)} large groups (> {threshold} files)")
    print(f"Found {len(small_groups)} small groups (<= {threshold} files)")
    
    # Step 1: Process large groups to form initial clusters
    clusters = []
    current_cluster = []
    
    for xps, files in large_groups:
        if not current_cluster:
            # Start first cluster
            current_cluster.append((xps, files))
            continue
        
        # Check if should merge with current cluster
        last_xps = current_cluster[-1][0]
        xps_diff = xps - last_xps
        
        if xps_diff <= tolerance:
            # Merge with current cluster
            current_cluster.append((xps, files))
        else:
            # Finalize current cluster and start new one
            clusters.append(current_cluster)
            current_cluster = [(xps, files)]
    
    # Don't forget the last cluster
    if current_cluster:
        clusters.append(current_cluster)
    
    print(f"Formed {len(clusters)} initial clusters from large groups")
    
    # Step 2: Merge small groups into nearest clusters
    for small_xps, small_files in small_groups:
        best_cluster_idx = None
        min_distance = float('inf')
        
        # Find nearest cluster
        for i, cluster in enumerate(clusters):
            # Use the weighted average XPS of the cluster for distance calculation
            cluster_xps = calculate_weighted_xps(cluster)
            distance = abs(small_xps - cluster_xps)
            
            if distance < min_distance:
                min_distance = distance
                best_cluster_idx = i
        
        # Merge small group into nearest cluster
        if best_cluster_idx is not None:
            clusters[best_cluster_idx].append((small_xps, small_files))
            print(f"Merged small group XPS {small_xps:.5f} ({len(small_files)} files) "
                  f"into cluster {best_cluster_idx} (distance: {min_distance:.5f})")
    
    # Step 3: Create final merged groups
    merged_groups = []
    for cluster in clusters:
        if cluster:  # Ensure cluster is not empty
            weighted_xps = calculate_weighted_xps(cluster)
            all_files = []
            for _, files in cluster:
                all_files.extend(files)
            merged_groups.append((weighted_xps, all_files))
    
    # Sort by XPS value
    merged_groups.sort(key=lambda x: x[0])
    
    return merged_groups

# --- New Manual Merging Function ---
def merge_xps_groups_manual(groups_with_xps: List[Tuple[float, List[str]]],
                                manual_groups_xps_values: List[List[float]]) -> List[Tuple[float, List[str]]]:
    """
    Merges XPS groups based on a manually defined list of XPS values.

    Args:
        groups_with_xps: The original list of (xps_value, [file_list]) tuples 
                         from 'group_tiff_files_with_info()'.
        manual_groups_xps_values: A list of lists, where each inner list 
                                  contains the XPS values (e.g., [xps1, xps2, ...])
                                  that should be merged into one final group. 
                                  The XPS values are expected to be floats.

    Returns:
        List of merged groups with calculated weighted average XPS values 
        and the combined list of files.
    """
    if not groups_with_xps or not manual_groups_xps_values:
        return []

    # 1. Create a lookup dictionary for fast access: XPS value -> (XPS, File List) tuple
    # Note: Use a precise float representation for the key, as specified in the prompt 
    # (though typically using decimals for keys is safer, floats should work here 
    # if the original XPS values were also formatted).
    xps_lookup: Dict[float, Tuple[float, List[str]]] = {
        round(xps, 5): (xps, files) for xps, files in groups_with_xps
    }
    
    merged_groups: List[Tuple[float, List[str]]] = []

    # 2. Iterate through the manual groups list
    for group_xps_list in manual_groups_xps_values:
        
        # This will hold the (xps, files) tuples for all initial groups 
        # that belong to the current final group
        current_cluster: List[Tuple[float, List[str]]] = []

        # 3. Collect the data for the manual cluster
        for xps_value in group_xps_list:
            # We must round the input XPS value to match the lookup key precision (.5f)
            # as specified in the prompt
            lookup_key = round(xps_value, 5) 
            
            if lookup_key in xps_lookup:
                # Retrieve the original (xps, files) tuple
                original_group_data = xps_lookup[lookup_key]
                current_cluster.append(original_group_data)
            else:
                # Optionally handle case where a specified XPS value doesn't exist 
                # in the original data (e.g., skip or raise error)
                print(f"Warning: XPS value {xps_value:.5f} not found in original groups. Skipping.")

        # 4. Calculate the weighted average and combine files for the final group
        if current_cluster:
            weighted_xps = calculate_weighted_xps(current_cluster)
            
            all_files: List[str] = []
            for _, files in current_cluster:
                all_files.extend(files)
            
            merged_groups.append((weighted_xps, all_files))

    # 5. Sort the final merged groups by their new weighted XPS value
    merged_groups.sort(key=lambda x: x[0])
    
    return merged_groups

def calculate_weighted_xps(cluster: List[Tuple[float, List[str]]]) -> float:
    """
    Calculate weighted average XPS for a cluster.
    
    Args:
        cluster: List of (xps_value, files) tuples
        
    Returns:
        Weighted average XPS value
    """
    total_files = sum(len(files) for _, files in cluster)
    if total_files == 0:
        return 0.0
    
    weighted_sum = sum(xps * len(files) for xps, files in cluster)
    return weighted_sum / total_files
