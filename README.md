# Angstrom: Phase-Based Motion Amplification

[![CI/CD Pipeline](https://github.com/levi2234/Angstrom/actions/workflows/ci.yml/badge.svg)](https://github.com/levi2234/Angstrom/actions/workflows/ci.yml)
[![Documentation](https://github.com/levi2234/Angstrom/actions/workflows/docs.yml/badge.svg)](https://github.com/levi2234/Angstrom/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://levi2234.github.io/Angstrom/)
[![PyPI](https://img.shields.io/pypi/v/angstrom.svg)](https://pypi.org/project/angstrom/)
[![Python](https://img.shields.io/pypi/pyversions/angstrom.svg)](https://pypi.org/project/angstrom/)

Angstrom is a Python library for **phase-based motion amplification** in videos. It uses complex steerable pyramids to decompose video frames and amplify subtle motion by manipulating phase coefficients. This technique is particularly useful for revealing imperceptible motion in videos, such as breathing, heartbeat, or structural vibrations.

## 🚀 Features

- **Phase-based motion amplification**: Uses complex steerable pyramids for accurate motion detection
- **Temporal filtering**: Apply bandpass filters to target specific motion frequencies (e.g., 0.1-2.0 Hz for human motion)
- **GPU acceleration**: Leverages PyTorch for efficient computation
- **Multiple output formats**: Support for various video formats
- **Configurable parameters**: Fine-tune amplification factors and frequency ranges
- **Real-time processing**: Optimized for processing video sequences

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install angstrom
```

### From Source
```bash
git clone https://github.com/levi2234/Angstrom.git
cd Angstrom
pip install -e .
```

### Dependencies
- Python 3.8+
- PyTorch 1.9.0+
- OpenCV 4.5.0+
- NumPy 1.21.0+
- SciPy 1.7.0+
- tqdm 4.62.0+

## 🎯 Quick Start

### Basic Motion Amplification

```python
from angstrom.core.motion_amplifier import MotionAmplifier

# Initialize the motion amplifier
amplifier = MotionAmplifier()

# Process a video with motion amplification
amplifier.process_video(
    input_path="input_video.mp4",
    output_path="amplified_video.mp4",
    amplification_factor=10,
    frequency_range=(0.1, 2.0)  # Hz - typical human motion frequencies
)
```

### Advanced Usage

```python
from angstrom.core.motion_amplifier import MotionAmplifier
import torch

# Initialize with specific device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amplifier = MotionAmplifier(device=device)

# Load and process video step by step
amplifier.load_video("input_video.mp4")
amplifier.process()  # Decompose frames into pyramid coefficients

# Amplify motion with custom parameters
amplified_video = amplifier.amplify(
    amplification_factor=20,
    frequency_range=(1.8, 2.2)  # Target specific frequency band
)

# Save the result
amplifier.save_video(amplified_video, "amplified_output.mp4")
```

### Processing Different Motion Types

```python
# For breathing motion (0.1-0.5 Hz)
amplifier.process_video(
    input_path="breathing_video.mp4",
    output_path="amplified_breathing.mp4",
    amplification_factor=50,
    frequency_range=(0.1, 0.5)
)

# For heartbeat motion (0.8-2.0 Hz)
amplifier.process_video(
    input_path="heartbeat_video.mp4",
    output_path="amplified_heartbeat.mp4",
    amplification_factor=100,
    frequency_range=(0.8, 2.0)
)

# For structural vibrations (5-20 Hz)
amplifier.process_video(
    input_path="vibration_video.mp4",
    output_path="amplified_vibration.mp4",
    amplification_factor=200,
    frequency_range=(5.0, 20.0)
)
```

## 🔬 How It Works

Angstrom uses a **phase-based motion amplification** approach:

1. **Video Decomposition**: Each frame is decomposed using complex steerable pyramids
2. **Phase Extraction**: Phase coefficients are extracted from the complex pyramid coefficients
3. **Motion Detection**: Temporal differences between frames reveal motion information
4. **Frequency Filtering**: Bandpass filters isolate motion at specific frequencies
5. **Motion Amplification**: The filtered motion is amplified by a specified factor
6. **Reconstruction**: Amplified motion is added back to the original phase and reconstructed

### Key Components

- **Complex Steerable Pyramids**: Multi-scale, multi-orientation decomposition
- **Phase Manipulation**: Direct manipulation of phase coefficients for motion amplification
- **Temporal Filtering**: Frequency-domain filtering to isolate specific motion types
- **PyTorch Integration**: GPU-accelerated computation for efficient processing

## 📁 Project Structure

```
Angstrom/
├── src/angstrom/
│   ├── core/
│   │   └── motion_amplifier.py      # Main motion amplification class
│   ├── processing/
│   │   ├── phase.py                 # Phase extraction and manipulation
│   │   ├── pyramid.py               # Complex steerable pyramid wrapper
│   │   └── temporal_filter.py       # Temporal filtering utilities
│   ├── pyramids/
│   │   ├── steerable_pyramid.py     # Complex steerable pyramid implementation
│   │   └── pyramid_utils.py         # Pyramid utility functions
│   ├── io/
│   │   └── video_io.py              # Video input/output utilities
│   └── utils/
│       └── helpers.py               # Helper functions
├── examples/                        # Usage examples and test scripts
├── tests/                          # Unit tests
├── docs/                           # Documentation
└── pyproject.toml                  # Project configuration
```

## 🧪 Examples

Check out the `examples/` directory for comprehensive usage examples:

- `video_motion_amplification.py`: Complete video processing pipeline
- `test_motion_amplification.py`: Motion amplification on test videos
- `debug_motion_amplification.py`: Debugging and analysis tools
- `amplify_and_save_video.py`: Simple amplification script

### Running Examples

```bash
# Basic motion amplification
python examples/amplify_and_save_video.py

# Test with provided test videos
python examples/test_motion_amplification.py

# Debug motion amplification
python examples/debug_motion_amplification.py
```

## 📊 Performance

- **Processing Speed**: ~2-5 frames/second on CPU, ~10-20 frames/second on GPU
- **Memory Usage**: Scales with video resolution and number of frames
- **Accuracy**: High-quality motion amplification with minimal artifacts
- **Scalability**: Supports videos of various resolutions and frame rates

## 🔧 Configuration

### Amplification Parameters

- **`amplification_factor`**: How much to amplify motion (typically 10-200)
- **`frequency_range`**: Target frequency band in Hz (e.g., (0.1, 2.0) for human motion)
- **`fps`**: Video frame rate (automatically detected)

### Pyramid Parameters

- **`height`**: Number of pyramid levels (default: 5)
- **`nbands`**: Number of orientation bands (default: 4)
- **`scale_factor`**: Scaling factor between levels (default: 2)

## 🐛 Troubleshooting

### Common Issues

1. **"No motion detected"**: Try increasing `amplification_factor` or adjusting `frequency_range`
2. **"Video appears frozen"**: Check if motion is within the specified frequency range
3. **"Out of memory"**: Reduce video resolution or process in smaller chunks
4. **"Poor quality output"**: Ensure input video has sufficient motion and good lighting

### Debug Mode

```python
# Enable debug output
amplifier = MotionAmplifier()
amplifier.load_video("input.mp4")
amplifier.process()

# Check pyramid coefficients
print(f"Number of frames: {len(amplifier.pyramid_coeffs)}")
print(f"Pyramid structure: {type(amplifier.pyramid_coeffs[0])}")
```

## 📚 Documentation

- **Full Documentation**: [https://levi2234.github.io/Angstrom/](https://levi2234.github.io/Angstrom/)
- **API Reference**: [https://levi2234.github.io/Angstrom/modules.html](https://levi2234.github.io/Angstrom/modules.html)
- **Examples**: See the `examples/` directory

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/levi2234/Angstrom.git
cd Angstrom

# Install development dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run linting
flake8 src/
black src/
pylint src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Complex Steerable Pyramids**: Based on the implementation from [PyTorchSteerablePyramid](https://github.com/tomrunia/PyTorchSteerablePyramid)
- **Motion Amplification Theory**: Inspired by the work of Wadhwa et al. on Eulerian Video Magnification
- **PyTorch**: For efficient GPU-accelerated computation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/levi2234/Angstrom/issues)
- **Discussions**: [GitHub Discussions](https://github.com/levi2234/Angstrom/discussions)
- **Email**: levi2234@hotmail.com

---

**Made with ❤️ for the computer vision community**
