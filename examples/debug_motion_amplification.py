import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier
from angstrom.processing.phase import extract_phase, extract_amplitude, reconstruct_from_amplitude_and_phase
from angstrom.processing.filters import temporal_ideal_filter

def debug_amplification_pipeline():
    """
    Step-by-step debugging of the motion amplification pipeline.
    """
    print("=== DEBUGGING MOTION AMPLIFICATION PIPELINE ===\n")

    # Step 1: Load and process video
    print("Step 1: Loading and processing video...")
    test_video_path = "src/angstrom/data/testvideos/turtleshort.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amplifier = MotionAmplifier(device=device)

    amplifier.load_video(test_video_path)
    amplifier.process()

    print(f"✓ Video loaded: {amplifier.video.shape}")
    print(f"✓ FPS: {amplifier.video_fps}")
    print(f"✓ Number of frames: {len(amplifier.pyramid_coeffs)}\n")

    # Step 2: Extract phase and amplitude coefficients
    print("Step 2: Extracting phase and amplitude coefficients...")
    phase_coeffs_list = []
    amplitude_coeffs_list = []

    for frame_coeffs in amplifier.pyramid_coeffs:
        phase_coeffs = extract_phase(frame_coeffs)
        amplitude_coeffs = extract_amplitude(frame_coeffs)
        phase_coeffs_list.append(phase_coeffs)
        amplitude_coeffs_list.append(amplitude_coeffs)

    print(f"✓ Phase coefficients extracted for {len(phase_coeffs_list)} frames")
    print(f"✓ Amplitude coefficients extracted for {len(amplitude_coeffs_list)} frames\n")

    # Step 3: Analyze pyramid structure
    print("Step 3: Analyzing pyramid structure...")
    first_frame = phase_coeffs_list[0]
    print(f"✓ Pyramid structure: {len(first_frame)} levels")

    for i, level in enumerate(first_frame):
        if isinstance(level, list) or isinstance(level, np.ndarray):
            print(f"  Level {i}: {len(level)} bands")
        else:
            print(f"  Level {i}: {type(level)}")

    # Find a bandpass level to analyze
    level_idx = None
    band_idx = None
    for i, level in enumerate(first_frame):
        if isinstance(level, np.ndarray) and len(level) > 0:
            level_idx = i
            band_idx = 0
            break

    if level_idx is None:
        print("❌ No bandpass levels found!")
        return

    print(f"✓ Analyzing level {level_idx}, band {band_idx}\n")

    # Step 4: Extract temporal phase sequence
    print("Step 4: Extracting temporal phase sequence...")
    phase_temporal_sequence = []
    for frame_coeffs in phase_coeffs_list:
        band_data = frame_coeffs[level_idx][band_idx]
        if isinstance(band_data, torch.Tensor):
            phase_temporal_sequence.append(band_data.cpu().numpy())
        elif isinstance(band_data, np.ndarray):
            phase_temporal_sequence.append(band_data)
        else:
            phase_temporal_sequence.append(np.array(0.0))

    phase_band = np.stack(phase_temporal_sequence, axis=0)
    print(f"✓ Phase band shape: {phase_band.shape}")
    print(f"✓ Phase range: [{np.min(phase_band):.3f}, {np.max(phase_band):.3f}]\n")

    # Step 5: Calculate phase differences
    print("Step 5: Calculating phase differences...")
    num_frames = len(phase_coeffs_list)
    base_phase = phase_band[0]
    phase_differences = np.zeros_like(phase_band)

    for t in range(num_frames):
        phase_diff = phase_band[t] - base_phase
        phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
        phase_differences[t] = phase_diff

    print(f"✓ Phase differences calculated")
    print(f"✓ Phase differences range: [{np.min(phase_differences):.3f}, {np.max(phase_differences):.3f}]")

    # Check if there's any motion
    motion_magnitude = np.std(phase_differences)
    print(f"✓ Motion magnitude (std dev): {motion_magnitude:.6f}")

    if motion_magnitude < 1e-6:
        print("❌ WARNING: Very little motion detected in phase differences!")
    else:
        print("✓ Motion detected in phase differences")
    print()

    # Step 6: Apply temporal filtering
    print("Step 6: Applying temporal bandpass filter...")
    amplification_factor = 10
    low_freq, high_freq = 0.1, 2.0
    fps = float(amplifier.video_fps) if amplifier.video_fps is not None else 30.0

    print(f"✓ Filter parameters: {low_freq}-{high_freq} Hz, FPS: {fps}")

    filtered_phase_differences = temporal_ideal_filter(phase_differences, low_freq, high_freq, fps)

    print(f"✓ Filtered phase differences range: [{np.min(filtered_phase_differences):.3f}, {np.max(filtered_phase_differences):.3f}]")

    # Check if filtering removed all motion
    filtered_motion_magnitude = np.std(filtered_phase_differences)
    print(f"✓ Filtered motion magnitude (std dev): {filtered_motion_magnitude:.6f}")

    if filtered_motion_magnitude < 1e-6:
        print("❌ WARNING: Filtering removed all motion!")
        print("   This could be because:")
        print("   - Motion frequencies are outside the filter range")
        print("   - Video is too short for the frequency range")
        print("   - Motion is too slow or too fast")
    else:
        print("✓ Motion preserved after filtering")
    print()

    # Step 7: Amplify motion
    print("Step 7: Amplifying motion...")
    amplified_phase_deviations = filtered_phase_differences * amplification_factor

    print(f"✓ Amplification factor: {amplification_factor}")
    print(f"✓ Amplified deviations range: [{np.min(amplified_phase_deviations):.3f}, {np.max(amplified_phase_deviations):.3f}]")

    amplified_motion_magnitude = np.std(amplified_phase_deviations)
    print(f"✓ Amplified motion magnitude (std dev): {amplified_motion_magnitude:.6f}")

    if filtered_motion_magnitude > 0:
        actual_amplification = amplified_motion_magnitude / filtered_motion_magnitude
        print(f"✓ Actual amplification ratio: {actual_amplification:.2f}x")
    print()

    # Step 8: Reconstruct amplified phase
    print("Step 8: Reconstructing amplified phase...")
    amplified_phase = np.zeros_like(phase_band)
    for t in range(num_frames):
        amplified_phase[t] = base_phase + amplified_phase_deviations[t]
        amplified_phase[t] = np.mod(amplified_phase[t] + np.pi, 2*np.pi) - np.pi

    print(f"✓ Amplified phase range: [{np.min(amplified_phase):.3f}, {np.max(amplified_phase):.3f}]")

    # Compare original vs amplified
    original_phase_range = np.max(phase_band) - np.min(phase_band)
    amplified_phase_range = np.max(amplified_phase) - np.min(amplified_phase)
    print(f"✓ Original phase range: {original_phase_range:.3f}")
    print(f"✓ Amplified phase range: {amplified_phase_range:.3f}")

    if amplified_phase_range > original_phase_range * 1.1:
        print("✓ Phase range increased - amplification working")
    else:
        print("❌ Phase range not significantly increased")
    print()

    # Step 9: Test reconstruction
    print("Step 9: Testing reconstruction...")
    try:
        # Test reconstruction with original coefficients
        original_recombined = reconstruct_from_amplitude_and_phase(
            amplitude_coeffs_list[0],
            [phase_coeffs_list[0][level_idx][band_idx] if isinstance(phase_coeffs_list[0][level_idx], list) else phase_coeffs_list[0][level_idx]]
        )
        print("✓ Original reconstruction successful")

        # Test reconstruction with amplified coefficients
        amplified_phase_tensor = torch.from_numpy(amplified_phase[0])
        amplified_recombined = reconstruct_from_amplitude_and_phase(
            amplitude_coeffs_list[0],
            [amplified_phase_tensor if isinstance(phase_coeffs_list[0][level_idx], list) else phase_coeffs_list[0][level_idx]]
        )
        print("✓ Amplified reconstruction successful")

    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")

    # Step 10: Create debugging plots
    print("\nStep 10: Creating debugging plots...")
    create_debugging_plots(phase_band, phase_differences, filtered_phase_differences,
                          amplified_phase_deviations, amplified_phase, fps, low_freq, high_freq, amplification_factor)

    print("\n=== DEBUGGING COMPLETE ===")
    print("Check the generated plots to see where amplification might be failing.")

def create_debugging_plots(phase_band, phase_differences, filtered_phase_differences,
                          amplified_phase_deviations, amplified_phase, fps, low_freq, high_freq, amplification_factor):
    """Create comprehensive debugging plots."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Motion Amplification Pipeline Debugging', fontsize=16)

    # Choose a pixel to track (center of the image)
    h, w = phase_band.shape[1], phase_band.shape[2]
    pixel_y, pixel_x = h // 2, w // 2
    num_frames = phase_band.shape[0]

    # Plot 1: Original phase over time
    original_phase_temporal = phase_band[:, pixel_y, pixel_x]
    axes[0, 0].plot(original_phase_temporal, 'b-', linewidth=2)
    axes[0, 0].set_title('Original Phase ϕ(t)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Phase (radians)')
    axes[0, 0].grid(True)

    # Plot 2: Phase differences
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
    final_phase = amplified_phase[:, pixel_y, pixel_x]
    axes[1, 1].plot(final_phase, 'c-', linewidth=2)
    axes[1, 1].set_title('Final Amplified Phase ϕ̃(t)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Amplified Phase (radians)')
    axes[1, 1].grid(True)

    # Plot 6: Comparison
    axes[1, 2].plot(original_phase_temporal, 'b-', linewidth=2, label='Original')
    axes[1, 2].plot(final_phase, 'r-', linewidth=2, label='Amplified')
    axes[1, 2].set_title('Original vs Amplified Phase')
    axes[1, 2].set_xlabel('Frame')
    axes[1, 2].set_ylabel('Phase (radians)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Plot 7: Motion magnitude over time
    motion_magnitude_original = np.std(phase_differences, axis=(1, 2))
    motion_magnitude_filtered = np.std(filtered_phase_differences, axis=(1, 2))
    motion_magnitude_amplified = np.std(amplified_phase_deviations, axis=(1, 2))

    axes[2, 0].plot(motion_magnitude_original, 'b-', linewidth=2, label='Original')
    axes[2, 0].plot(motion_magnitude_filtered, 'g-', linewidth=2, label='Filtered')
    axes[2, 0].plot(motion_magnitude_amplified, 'r-', linewidth=2, label='Amplified')
    axes[2, 0].set_title('Motion Magnitude Over Time')
    axes[2, 0].set_xlabel('Frame')
    axes[2, 0].set_ylabel('Motion Magnitude (std dev)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Plot 8: Frequency spectrum of phase differences
    fft_original = np.fft.fft(phase_diff_temporal)
    frequencies = np.fft.fftfreq(num_frames, d=1.0/fps)
    axes[2, 1].plot(frequencies[:num_frames//2], np.abs(fft_original)[:num_frames//2], 'b-', linewidth=2)
    axes[2, 1].axvline(x=low_freq, color='r', linestyle='--', label=f'Low cutoff: {low_freq} Hz')
    axes[2, 1].axvline(x=high_freq, color='r', linestyle='--', label=f'High cutoff: {high_freq} Hz')
    axes[2, 1].set_title('Frequency Spectrum of Phase Differences')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    # Plot 9: Statistics summary
    axes[2, 2].text(0.1, 0.9, f'Original motion std: {np.std(phase_diff_temporal):.6f}', fontsize=12)
    axes[2, 2].text(0.1, 0.8, f'Filtered motion std: {np.std(filtered_diff_temporal):.6f}', fontsize=12)
    axes[2, 2].text(0.1, 0.7, f'Amplified motion std: {np.std(amplified_dev_temporal):.6f}', fontsize=12)
    axes[2, 2].text(0.1, 0.6, f'Actual amplification: {np.std(amplified_dev_temporal)/np.std(filtered_diff_temporal):.2f}x', fontsize=12)
    axes[2, 2].text(0.1, 0.5, f'Target amplification: {amplification_factor}x', fontsize=12)
    axes[2, 2].text(0.1, 0.4, f'Video length: {num_frames} frames', fontsize=12)
    axes[2, 2].text(0.1, 0.3, f'FPS: {fps}', fontsize=12)
    axes[2, 2].text(0.1, 0.2, f'Filter range: {low_freq}-{high_freq} Hz', fontsize=12)
    axes[2, 2].set_title('Statistics Summary')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('amplification_debugging.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Debugging plots saved as 'amplification_debugging.png'")

if __name__ == "__main__":
    debug_amplification_pipeline()