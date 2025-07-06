import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier
from angstrom.processing.phase import extract_phase
from angstrom.utils.visualization import (
    visualize_pyramid_phases,
    visualize_phase_temporal_evolution,
    visualize_phase_comparison,
    create_phase_video,
    visualize_pyramid_structure
)


def main():
    """
    Example script demonstrating pyramid phase visualization functions.
    """
    print("=== Pyramid Phase Visualization Demo ===\n")

    # Check if test video exists
    test_video_path = "src/angstrom/data/testvideos/vibration_test.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        print("Please ensure you have a test video file.")
        return

    print("1. Loading and processing video...")

    # Initialize motion amplifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amplifier = MotionAmplifier(device=device)

    # Load and process the video
    amplifier.load_video(test_video_path)
    amplifier.process()

    print(f"✓ Video loaded: {amplifier.video.shape}")
    print(f"✓ FPS: {amplifier.video_fps}")
    print(f"✓ Number of frames: {len(amplifier.pyramid_coeffs) if amplifier.pyramid_coeffs else 0}\n")

    # Extract phase coefficients for all frames
    print("2. Extracting phase coefficients...")
    phase_coeffs_list = []
    if amplifier.pyramid_coeffs:
        for frame_coeffs in amplifier.pyramid_coeffs:
            phase_coeffs = extract_phase(frame_coeffs)
            phase_coeffs_list.append(phase_coeffs)

    print(f"✓ Phase coefficients extracted for {len(phase_coeffs_list)} frames\n")

    # Analyze pyramid structure
    first_frame = phase_coeffs_list[0]
    print("3. Analyzing pyramid structure...")
    print(f"✓ Pyramid structure: {len(first_frame)} levels")

    for i, level in enumerate(first_frame):
        if isinstance(level, list):
            print(f"  Level {i}: {len(level)} bands (bandpass)")
        else:
            print(f"  Level {i}: {type(level)} (lowpass/highpass)")

    print("\n4. Creating visualizations...")

    # Create output directory for visualizations
    output_dir = "pyramid_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Visualization 1: Pyramid structure analysis
    print("   Creating pyramid structure visualization...")
    visualize_pyramid_structure(
        phase_coeffs_list,
        frame_idx=0,
        save_path=os.path.join(output_dir, "pyramid_structure.png")
    )

    # Visualization 2: Phase visualization for first frame
    print("   Creating phase visualization for first frame...")
    visualize_pyramid_phases(
        phase_coeffs_list,
        frame_idx=0,
        max_levels=3,
        max_bands=4,
        save_path=os.path.join(output_dir, "pyramid_phases_frame0.png")
    )

    # Find a bandpass level to analyze
    level_idx = None
    band_idx = None
    for i, level in enumerate(first_frame):
        if isinstance(level, list) and len(level) > 0:
            level_idx = i
            band_idx = 0
            break

    if level_idx is not None and band_idx is not None:
        # Visualization 3: Temporal evolution
        print(f"   Creating temporal evolution visualization (Level {level_idx}, Band {band_idx})...")
        visualize_phase_temporal_evolution(
            phase_coeffs_list,
            level_idx=level_idx,
            band_idx=band_idx,
            save_path=os.path.join(output_dir, "phase_temporal_evolution.png")
        )

        # Visualization 4: Phase video
        print("   Creating phase video...")
        create_phase_video(
            phase_coeffs_list,
            level_idx=level_idx,
            band_idx=band_idx,
            output_path=os.path.join(output_dir, "phase_evolution.mp4"),
            fps=10
        )

    # Visualization 5: Phase comparison (if we have amplified phases)
    print("5. Testing phase amplification and comparison...")
    try:
        # Run motion amplification to get amplified phases
        amplification_factor = 4
        frequency_range = (0.1, 2.0)  # (low_freq, high_freq)

        print(f"   Running motion amplification (factor={amplification_factor})...")
        amplified_video = amplifier.amplify(
            amplification_factor=amplification_factor,
            frequency_range=frequency_range
        )

        # Extract amplified phase coefficients
        amplified_phase_coeffs_list = []
        if amplifier.pyramid_coeffs:
            for frame_coeffs in amplifier.pyramid_coeffs:  # These should be updated after amplification
                phase_coeffs = extract_phase(frame_coeffs)
                amplified_phase_coeffs_list.append(phase_coeffs)

        if level_idx is not None and band_idx is not None:
            print("   Creating phase comparison visualization...")
            visualize_phase_comparison(
                phase_coeffs_list,
                amplified_phase_coeffs_list,
                frame_idx=0,
                level_idx=level_idx,
                band_idx=band_idx,
                save_path=os.path.join(output_dir, "phase_comparison.png")
            )

    except Exception as e:
        print(f"   Warning: Could not create phase comparison: {e}")

    print(f"\n=== Visualization Complete ===")
    print(f"All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - pyramid_structure.png: Overall pyramid structure analysis")
    print("  - pyramid_phases_frame0.png: Phase visualization for first frame")
    print("  - phase_temporal_evolution.png: Temporal evolution of phase")
    print("  - phase_evolution.mp4: Video showing phase evolution over time")
    if level_idx is not None:
        print("  - phase_comparison.png: Comparison of original vs amplified phases")


if __name__ == "__main__":
    main()