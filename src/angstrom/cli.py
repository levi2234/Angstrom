#!/usr/bin/env python3
"""
Command-line interface for Angstrom motion amplification.
"""

import argparse
import sys
from pathlib import Path

from angstrom.core.motion_amplifier import MotionAmplifier


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Angstrom: Phase-based motion amplification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic motion amplification
  angstrom input.mp4 output.mp4 --factor 10

  # Amplify specific frequency range (breathing motion)
  angstrom input.mp4 output.mp4 --factor 50 --freq-range 0.1 0.5

  # Amplify heartbeat motion
  angstrom input.mp4 output.mp4 --factor 100 --freq-range 0.8 2.0
        """,
    )

    parser.add_argument("input", type=str, help="Input video file path")

    parser.add_argument("output", type=str, help="Output video file path")

    parser.add_argument("--factor", "-f", type=float, default=10.0, help="Amplification factor (default: 10.0)")

    parser.add_argument(
        "--freq-range",
        "-r",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Frequency range in Hz (e.g., 0.1 2.0 for human motion)",
    )

    parser.add_argument(
        "--device", "-d", type=str, choices=["cpu", "cuda"], help="Device to use (default: auto-detect)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Validate output directory
    output_path = Path(args.output)
    output_dir = output_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize motion amplifier
        device = args.device
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Using CPU.", file=sys.stderr)
                device = "cpu"

        amplifier = MotionAmplifier(device=device)

        if args.verbose:
            print(f"Processing video: {input_path}")
            print(f"Output: {output_path}")
            print(f"Amplification factor: {args.factor}")
            if args.freq_range:
                print(f"Frequency range: {args.freq_range[0]}-{args.freq_range[1]} Hz")

        # Process video
        amplifier.process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            amplification_factor=args.factor,
            frequency_range=tuple(args.freq_range) if args.freq_range else None,
        )

        if args.verbose:
            print("Motion amplification completed successfully!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
