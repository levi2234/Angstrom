# Video Motion Amplification Examples

This directory contains examples for using the Angstrom video motion amplification library.

## Installation

First, install the required dependencies:

```bash
# Install the package in development mode
pip install -e .

# Or install dependencies manually
pip install torch numpy opencv-python scipy tqdm
```

## Usage

### Command Line Interface

The `video_motion_amplification.py` script provides a command-line interface for processing videos:

```bash
# Basic usage with frequency filtering
python video_motion_amplification.py --input input.mp4 --output output.mp4 --amplification 10 --freq-low 0.5 --freq-high 2.0

# Amplify all frequencies (no filtering)
python video_motion_amplification.py --input input.mp4 --output output.mp4 --amplification 10 --no-frequency-filter

# Use CPU instead of GPU
python video_motion_amplification.py --input input.mp4 --output output.mp4 --device cpu
```

### Command Line Options

- `--input, -i`: Input video file path (required)
- `--output, -o`: Output video file path (required)
- `--amplification, -a`: Amplification factor (default: 10.0)
- `--freq-low`: Lower frequency cutoff in Hz (default: 0.5)
- `--freq-high`: Upper frequency cutoff in Hz (default: 2.0)
- `--filter-order`: Temporal filter order (default: 5)
- `--device`: Device to use for computation: "cpu", "cuda", or "auto" (default: auto)
- `--no-frequency-filter`: Disable frequency filtering (amplify all frequencies)

### Running Examples

If you run the script without any arguments, it will automatically run predefined examples using test videos:

```bash
python video_motion_amplification.py
```

This will create four example outputs:
1. **Basic amplification**: All frequencies amplified
2. **Low-frequency amplification**: 0.1-1.0 Hz (very slow motion)
3. **Medium-frequency amplification**: 0.5-2.0 Hz (slow to medium motion)
4. **High-frequency amplification**: 1.0-5.0 Hz (medium to fast motion)

### Frequency Range Guidelines

- **0.1-1.0 Hz**: Very slow motion (breathing, subtle movements)
- **0.5-2.0 Hz**: Slow to medium motion (heartbeat, gentle swaying)
- **1.0-5.0 Hz**: Medium to fast motion (walking, quick gestures)
- **5.0+ Hz**: Fast motion (running, rapid movements)

### Python API

You can also use the MotionAmplifier class directly in your Python code:

```python
from angstrom.core.motion_amplifier import MotionAmplifier

# Initialize the amplifier
amplifier = MotionAmplifier()

# Process a video with frequency range boosting
amplifier.process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    amplification_factor=10,
    frequency_range=(0.5, 2.0),  # Boost 0.5-2 Hz frequencies
    temporal_filter_order=5
)
```

## Test Videos

The script will automatically look for test videos in:
```
src/angstrom/data/testvideos/
```

If no test videos are found, you'll need to provide your own video file using the command line arguments.

## Output

Processed videos will be saved to:
```
examples/output/
```

Each example will create a video with a descriptive filename indicating the processing parameters used.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'steerable'**
   - This has been fixed by using the custom steerable pyramid implementation
   - Make sure you have the latest version of the code

2. **CUDA out of memory**
   - Use `--device cpu` to process on CPU instead of GPU
   - Try processing shorter videos or lower resolution videos

3. **Video not found**
   - Make sure the input video path is correct
   - Check that the video file exists and is readable

### Performance Tips

- Use GPU (CUDA) for faster processing if available
- Lower resolution videos process faster
- Shorter videos require less memory
- Higher amplification factors may require more memory

## Examples Output

The script will show progress bars and information about:
- Video loading and properties
- Pyramid decomposition progress
- Temporal filtering (if enabled)
- Amplification and reconstruction progress
- Final video statistics

Example output:
```
Using device: cuda
Initializing MotionAmplifier...
Frequency range: 0.5-2.0 Hz
Amplification factor: 10

Processing video: input.mp4
Output will be saved to: output.mp4
Loaded video: 300 frames, 30.0 FPS, Resolution: 1920x1080
Decomposing video frames into pyramid coefficients...
Applying temporal filter: 0.5-2.0 Hz...
Amplifying and reconstructing video frames...
Video saved to: output.mp4

✅ Video processing completed successfully!
Output video saved to: output.mp4
Output video shape: torch.Size([300, 1, 1080, 1920])
Output video range: [0.000, 1.000]
```