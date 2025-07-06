import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib


# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier
from angstrom.processing.phase import extract_phase
from angstrom.processing.filters import temporal_ideal_filter

def plot_phase_differences():
    """
    Plot phase differences at different stages of motion amplification.
    """
    # Check if test video exists
    test_video_path = "src/angstrom/data/testvideos/turtleshort.mp4"
    print(test_video_path)
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        print("Please ensure you have a test video file.")
        return

    print("Loading and processing video for phase difference analysis...")

    # Initialize motion amplifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amplifier = MotionAmplifier(device=device)

    # Load and process the video
    amplifier.load_video(test_video_path)
    amplifier.process()

    # Extract phase coefficients for all frames
    phase_coeffs_list = []
    for frame_coeffs in amplifier.pyramid_coeffs:
        phase_coeffs = extract_phase(frame_coeffs)
        phase_coeffs_list.append(phase_coeffs)

    # Parameters for analysis
    amplification_factor = 10
    low_freq, high_freq = 0.1, 2.0
    fps = float(amplifier.video_fps) if amplifier.video_fps is not None else 30.0

    # Get structure from first frame
    first_frame = phase_coeffs_list[0]
    num_frames = len(phase_coeffs_list)

    # Find a bandpass level to analyze (skip lowpass/highpass)
    level_idx = None
    band_idx = None
    for i, level in enumerate(first_frame):
        if isinstance(level, list) and len(level) > 0:
            level_idx = i
            band_idx = 0
            break

    if level_idx is None:
        print("No bandpass levels found in pyramid decomposition.")
        return

    print(f"Analyzing phase differences for level {level_idx}, band {band_idx}")

    # Extract temporal sequence of PHASE coefficients for this band
    phase_temporal_sequence = []
    for frame_coeffs in phase_coeffs_list:
        band_data = frame_coeffs[level_idx][band_idx]
        if isinstance(band_data, torch.Tensor):
            phase_temporal_sequence.append(band_data.cpu().numpy())
        elif isinstance(band_data, np.ndarray):
            phase_temporal_sequence.append(band_data)
        else:
            phase_temporal_sequence.append(np.array(0.0))

    # Stack into [T, H, W] array of phase coefficients
    phase_band = np.stack(phase_temporal_sequence, axis=0)

    # Step 1: Calculate phase differences from base phase (frame 0)
    base_phase = phase_band[0]  # ϕ(0)
    phase_differences = np.zeros_like(phase_band)
    for t in range(num_frames):
        phase_diff = phase_band[t] - base_phase
        # Handle phase wrapping (ensure differences are in [-π, π])
        phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
        phase_differences[t] = phase_diff

    # Step 2: Apply temporal bandpass filter to phase differences
    filtered_phase_differences = temporal_ideal_filter(phase_differences, low_freq, high_freq, fps)

    # Step 3: Amplify the filtered phase deviations
    amplified_phase_deviations = filtered_phase_differences * amplification_factor

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase-Based Motion Amplification Analysis', fontsize=16)

    # Choose a pixel to track (center of the image)
    h, w = phase_band.shape[1], phase_band.shape[2]
    pixel_y, pixel_x = h // 2, w // 2

    # Plot 1: Original phase over time
    original_phase_temporal = phase_band[:, pixel_y, pixel_x]
    axes[0, 0].plot(original_phase_temporal, 'b-', linewidth=2)
    axes[0, 0].set_title('Original Phase ϕ(t)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Phase (radians)')
    axes[0, 0].grid(True)

    # Plot 2: Phase differences Δϕ(t) = ϕ(t) - ϕ(0)
    phase_diff_temporal = phase_differences[:, pixel_y, pixel_x]
    axes[0, 1].plot(phase_diff_temporal, 'r-', linewidth=2)
    axes[0, 1].set_title('Phase Differences Δϕ(t)')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Phase Difference (radians)')
    axes[0, 1].grid(True)

    # Plot 3: Filtered phase differences
    filtered_diff_temporal = filtered_phase_differences[:, pixel_y, pixel_x]
    axes[0, 2].plot(filtered_diff_temporal, 'g-', linewidth=2)
    axes[0, 2].set_title(f'Filtered Phase Differences\nbandpass({low_freq}-{high_freq} Hz)')
    axes[0, 2].set_xlabel('Frame')
    axes[0, 2].set_ylabel('Filtered Phase Difference (radians)')
    axes[0, 2].grid(True)

    # Plot 4: Amplified phase deviations
    amplified_dev_temporal = amplified_phase_deviations[:, pixel_y, pixel_x]
    axes[1, 0].plot(amplified_dev_temporal, 'm-', linewidth=2)
    axes[1, 0].set_title(f'Amplified Phase Deviations\nα = {amplification_factor}')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Amplified Phase Deviation (radians)')
    axes[1, 0].grid(True)

    # Plot 5: Final amplified phase
    final_phase = base_phase[pixel_y, pixel_x] + amplified_dev_temporal
    axes[1, 1].plot(final_phase, 'c-', linewidth=2)
    axes[1, 1].set_title('Final Amplified Phase ϕ̃(t)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Amplified Phase (radians)')
    axes[1, 1].grid(True)

    # Plot 6: Comparison of original vs amplified phase
    axes[1, 2].plot(original_phase_temporal, 'b-', linewidth=2, label='Original')
    axes[1, 2].plot(final_phase, 'r-', linewidth=2, label='Amplified')
    axes[1, 2].set_title('Original vs Amplified Phase')
    axes[1, 2].set_xlabel('Frame')
    axes[1, 2].set_ylabel('Phase (radians)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('phase_differences_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional analysis: Show spatial distribution of phase differences
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Spatial Distribution of Phase Differences', fontsize=16)

    # Plot spatial distribution at a specific frame (middle frame)
    mid_frame = num_frames // 2

    # Original phase differences
    im1 = axes2[0, 0].imshow(phase_differences[mid_frame], cmap='viridis', aspect='auto')
    axes2[0, 0].set_title(f'Phase Differences Δϕ(t={mid_frame})')
    axes2[0, 0].set_xlabel('X')
    axes2[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes2[0, 0])

    # Filtered phase differences
    im2 = axes2[0, 1].imshow(filtered_phase_differences[mid_frame], cmap='viridis', aspect='auto')
    axes2[0, 1].set_title(f'Filtered Phase Differences\nbandpass({low_freq}-{high_freq} Hz)')
    axes2[0, 1].set_xlabel('X')
    axes2[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes2[0, 1])

    # Amplified phase deviations
    im3 = axes2[1, 0].imshow(amplified_phase_deviations[mid_frame], cmap='viridis', aspect='auto')
    axes2[1, 0].set_title(f'Amplified Phase Deviations\nα = {amplification_factor}')
    axes2[1, 0].set_xlabel('X')
    axes2[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes2[1, 0])

    # Difference between original and filtered
    diff_map = np.abs(phase_differences[mid_frame] - filtered_phase_differences[mid_frame])
    im4 = axes2[1, 1].imshow(diff_map, cmap='hot', aspect='auto')
    axes2[1, 1].set_title('|Original - Filtered| Phase Differences')
    axes2[1, 1].set_xlabel('X')
    axes2[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes2[1, 1])

    plt.tight_layout()
    plt.savefig('spatial_phase_differences.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Phase difference analysis complete!")
    print("Plots saved as 'phase_differences_analysis.png' and 'spatial_phase_differences.png'")

if __name__ == "__main__":
    plot_phase_differences()