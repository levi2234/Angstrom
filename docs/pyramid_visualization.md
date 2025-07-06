# Pyramid Phase Visualization

This module provides comprehensive visualization tools for analyzing pyramid phases from motion amplification output.

## Overview

The visualization module (`src/angstrom/utils/visualization.py`) contains functions to visualize and analyze the complex steerable pyramid decomposition used in motion amplification. These tools help understand:

- Pyramid structure and organization
- Phase evolution over time
- Comparison between original and amplified phases
- Spatial distribution of phase information

## Functions

### 1. `visualize_pyramid_phases()`

Visualizes pyramid phases for a specific frame, showing all levels and orientation bands.

**Parameters:**
- `phase_coeffs_list`: List of phase coefficients for each frame
- `frame_idx`: Index of frame to visualize (default: 0)
- `max_levels`: Maximum number of pyramid levels to display (default: 5)
- `max_bands`: Maximum number of orientation bands to display per level (default: 4)
- `figsize`: Figure size for the plot (default: (20, 15))
- `save_path`: Optional path to save the visualization

**Usage:**
```python
from angstrom.utils.visualization import visualize_pyramid_phases

# After extracting phase coefficients
visualize_pyramid_phases(
    phase_coeffs_list,
    frame_idx=0,
    save_path="pyramid_phases.png"
)
```

### 2. `visualize_phase_temporal_evolution()`

Shows how phase evolves over time at a specific pyramid level and orientation band.

**Parameters:**
- `phase_coeffs_list`: List of phase coefficients for each frame
- `level_idx`: Pyramid level to analyze
- `band_idx`: Orientation band to analyze
- `pixel_pos`: Specific pixel position (y, x) to track (default: center pixel)
- `figsize`: Figure size for the plot (default: (15, 10))
- `save_path`: Optional path to save the visualization

**Usage:**
```python
from angstrom.utils.visualization import visualize_phase_temporal_evolution

visualize_phase_temporal_evolution(
    phase_coeffs_list,
    level_idx=0,
    band_idx=0,
    save_path="temporal_evolution.png"
)
```

### 3. `visualize_phase_comparison()`

Compares original and amplified phases side by side.

**Parameters:**
- `original_phases`: List of original phase coefficients
- `amplified_phases`: List of amplified phase coefficients
- `frame_idx`: Frame index to compare (default: 0)
- `level_idx`: Pyramid level to compare
- `band_idx`: Orientation band to compare
- `figsize`: Figure size for the plot (default: (20, 10))
- `save_path`: Optional path to save the visualization

**Usage:**
```python
from angstrom.utils.visualization import visualize_phase_comparison

visualize_phase_comparison(
    original_phases,
    amplified_phases,
    level_idx=0,
    band_idx=0,
    save_path="phase_comparison.png"
)
```

### 4. `create_phase_video()`

Creates a video visualization of phase evolution over time.

**Parameters:**
- `phase_coeffs_list`: List of phase coefficients for each frame
- `level_idx`: Pyramid level to visualize
- `band_idx`: Orientation band to visualize
- `output_path`: Path to save the output video (default: "phase_visualization.mp4")
- `fps`: Frames per second for the output video (default: 30)
- `colormap`: Matplotlib colormap to use (default: 'twilight')

**Usage:**
```python
from angstrom.utils.visualization import create_phase_video

create_phase_video(
    phase_coeffs_list,
    level_idx=0,
    band_idx=0,
    output_path="phase_evolution.mp4",
    fps=10
)
```

### 5. `visualize_pyramid_structure()`

Creates a comprehensive visualization of the pyramid structure.

**Parameters:**
- `phase_coeffs_list`: List of phase coefficients for each frame
- `frame_idx`: Frame index to visualize (default: 0)
- `figsize`: Figure size for the plot (default: (15, 10))
- `save_path`: Optional path to save the visualization

**Usage:**
```python
from angstrom.utils.visualization import visualize_pyramid_structure

visualize_pyramid_structure(
    phase_coeffs_list,
    frame_idx=0,
    save_path="pyramid_structure.png"
)
```

## Complete Example

See `examples/visualize_pyramid_phases.py` for a complete example that demonstrates all visualization functions.

```python
import os
import sys
import torch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier
from angstrom.processing.phase import extract_phase
from angstrom.utils.visualization import (
    visualize_pyramid_phases,
    visualize_phase_temporal_evolution,
    visualize_pyramid_structure
)

# Load and process video
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amplifier = MotionAmplifier(device=device)
amplifier.load_video("path/to/video.mp4")
amplifier.process()

# Extract phase coefficients
phase_coeffs_list = []
for frame_coeffs in amplifier.pyramid_coeffs:
    phase_coeffs = extract_phase(frame_coeffs)
    phase_coeffs_list.append(phase_coeffs)

# Create visualizations
visualize_pyramid_structure(phase_coeffs_list, save_path="structure.png")
visualize_pyramid_phases(phase_coeffs_list, save_path="phases.png")
visualize_phase_temporal_evolution(phase_coeffs_list, level_idx=0, band_idx=0, save_path="evolution.png")
```

## Understanding the Output

### Pyramid Structure
The complex steerable pyramid decomposes an image into multiple levels and orientation bands:

- **Levels**: Different spatial scales (coarse to fine)
- **Bands**: Different orientations (typically 4-8 orientations)
- **Types**:
  - Bandpass filters: Capture oriented features at specific scales
  - Lowpass/Highpass filters: Capture overall structure

### Phase Information
- **Phase values**: Range from -π to π radians
- **Phase differences**: Show motion between frames
- **Amplified phases**: Enhanced motion after amplification

### Visualization Colors
- **Twilight colormap**: Used for phase visualization (cyclic)
- **Viridis colormap**: Used for magnitude/amplitude visualization
- **RdBu_r colormap**: Used for phase differences (red=positive, blue=negative)

## Tips for Analysis

1. **Start with structure**: Use `visualize_pyramid_structure()` to understand the pyramid organization
2. **Focus on bandpass levels**: These contain the most motion information
3. **Check temporal evolution**: Look for consistent patterns over time
4. **Compare before/after**: Use comparison functions to see amplification effects
5. **Adjust parameters**: Modify `max_levels` and `max_bands` based on your pyramid configuration

## Troubleshooting

- **No data found**: Ensure `phase_coeffs_list` is not empty
- **Index errors**: Check that `level_idx` and `band_idx` are within valid ranges
- **Memory issues**: Reduce `max_levels` or `max_bands` for large pyramids
- **Video creation fails**: Ensure OpenCV is installed and the output directory is writable