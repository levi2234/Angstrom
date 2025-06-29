#!/usr/bin/env python3
"""
Video Motion Amplification Example Script

This script demonstrates how to use the MotionAmplifier class to amplify
motion in videos with frequency range boosting capabilities.

Example usage:
    python video_motion_amplification.py --input input.mp4 --output output.mp4 --amplification 10 --freq-range 0.5 2.0
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from angstrom.core.motion_amplifier import MotionAmplifier


def main():
    """Main function to run video motion amplification examples."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Motion Amplification")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", required=True, help="Output video file path")
    parser.add_argument("--amplification", "-a", type=float, default=10.0,
                       help="Amplification factor (default: 10.0)")
    parser.add_argument("--freq-low", type=float, default=0.5,
                       help="Lower frequency cutoff in Hz (default: 0.5)")
    parser.add_argument("--freq-high", type=float, default=2.0,
                       help="Upper frequency cutoff in Hz (default: 2.0)")
    parser.add_argument("--filter-order", type=int, default=5,
                       help="Temporal filter order (default: 5)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="auto",
                       help="Device to use for computation (default: auto)")
    parser.add_argument("--no-frequency-filter", action="store_true",
                       help="Disable frequency filtering (amplify all frequencies)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    try:
        # Initialize the motion amplifier
        print("Initializing MotionAmplifier...")
        amplifier = MotionAmplifier(device=device)
        print(device)

        # Set frequency range
        frequency_range = None if args.no_frequency_filter else (args.freq_low, args.freq_high)

        if frequency_range:
            print(f"Frequency range: {frequency_range[0]}-{frequency_range[1]} Hz")
        else:
            print("No frequency filtering (amplifying all frequencies)")

        print(f"Amplification factor: {args.amplification}")

        # Process the video
        print(f"\nProcessing video: {args.input}")
        print(f"Output will be saved to: {args.output}")

        amplified_video = amplifier.process_video(
            input_path=args.input,
            output_path=args.output,
            amplification_factor=int(args.amplification),
            frequency_range=frequency_range,
            temporal_filter_order=args.filter_order
        )

        print(f"\n✅ Video processing completed successfully!")
        print(f"Output video saved to: {args.output}")

        # Print video statistics
        if amplified_video is not None:
            print(f"Output video shape: {amplified_video.shape}")
            print(f"Output video range: [{amplified_video.min():.3f}, {amplified_video.max():.3f}]")

        return 0

    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        return 1


def run_examples():
    """Run predefined examples to demonstrate different use cases."""

    print("🎬 Video Motion Amplification Examples")
    print("=" * 50)

    # Check if we have test videos available
    test_video_dir = Path(__file__).parent.parent / "src" / "angstrom" / "data" / "testvideos"

    if not test_video_dir.exists():
        print(f"Test video directory not found: {test_video_dir}")
        print("Please provide your own video file using the command line arguments.")
        return

    # Find available test videos
    test_videos = list(test_video_dir.glob("*.mp4"))
    if not test_videos:
        print("No test videos found in the testvideos directory.")
        return

    print(f"Found {len(test_videos)} test videos:")
    for i, video in enumerate(test_videos):
        print(f"  {i+1}. {video.name}")

    # Use the first available video for examples
    input_video = test_videos[0]
    print(f"\nUsing test video: {input_video.name}")

    # Create output directory for examples
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Example 1: Basic motion amplification (all frequencies)
    print("\n📹 Example 1: Basic motion amplification (all frequencies)")
    print("-" * 60)

    try:
        amplifier1 = MotionAmplifier()
        output1 = output_dir / f"example1_basic_{input_video.stem}.mp4"

        amplified1 = amplifier1.process_video(
            input_path=str(input_video),
            output_path=str(output1),
            amplification_factor=5,
            frequency_range=None  # No frequency filtering
        )
        print(f"✅ Example 1 completed: {output1}")

    except Exception as e:
        print(f"❌ Example 1 failed: {e}")

    # Example 2: Low-frequency motion amplification (0.1-1.0 Hz)
    print("\n📹 Example 2: Low-frequency motion amplification (0.1-1.0 Hz)")
    print("-" * 60)

    try:
        amplifier2 = MotionAmplifier()
        output2 = output_dir / f"example2_lowfreq_{input_video.stem}.mp4"

        amplified2 = amplifier2.process_video(
            input_path=str(input_video),
            output_path=str(output2),
            amplification_factor=8,
            frequency_range=(0.1, 1.0)
        )
        print(f"✅ Example 2 completed: {output2}")

    except Exception as e:
        print(f"❌ Example 2 failed: {e}")

    # Example 3: Medium-frequency motion amplification (0.5-2.0 Hz)
    print("\n📹 Example 3: Medium-frequency motion amplification (0.5-2.0 Hz)")
    print("-" * 60)

    try:
        amplifier3 = MotionAmplifier()
        output3 = output_dir / f"example3_mediumfreq_{input_video.stem}.mp4"

        amplified3 = amplifier3.process_video(
            input_path=str(input_video),
            output_path=str(output3),
            amplification_factor=10,
            frequency_range=(0.5, 2.0)
        )
        print(f"✅ Example 3 completed: {output3}")

    except Exception as e:
        print(f"❌ Example 3 failed: {e}")

    # Example 4: High-frequency motion amplification (1.0-5.0 Hz)
    print("\n📹 Example 4: High-frequency motion amplification (1.0-5.0 Hz)")
    print("-" * 60)

    try:
        amplifier4 = MotionAmplifier()
        output4 = output_dir / f"example4_highfreq_{input_video.stem}.mp4"

        amplified4 = amplifier4.process_video(
            input_path=str(input_video),
            output_path=str(output4),
            amplification_factor=15,
            frequency_range=(1.0, 5.0)
        )
        print(f"✅ Example 4 completed: {output4}")

    except Exception as e:
        print(f"❌ Example 4 failed: {e}")

    print(f"\n🎉 All examples completed! Check the output directory: {output_dir}")
    print("\n📋 Summary of frequency ranges:")
    print("  - Example 1: All frequencies (no filtering)")
    print("  - Example 2: 0.1-1.0 Hz (very slow motion)")
    print("  - Example 3: 0.5-2.0 Hz (slow to medium motion)")
    print("  - Example 4: 1.0-5.0 Hz (medium to fast motion)")


if __name__ == "__main__":
    # Check if any command line arguments were provided
    if len(sys.argv) > 1:
        # Run with command line arguments
        exit_code = main()
        sys.exit(exit_code)
    else:
        # Run predefined examples
        run_examples()