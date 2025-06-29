import os
import sys
import torch

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier

def test_motion_amplification():
    """
    Test motion amplification on the generated square motion video.
    """

    # Check if test video exists
    test_video_path = "test_square_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        print("Please run generate_test_video.py first to create the test video.")
        return

    print("Testing motion amplification on pulsing square video...")

    # Initialize motion amplifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    amplifier = MotionAmplifier(device=device)

    # Load and process the video
    print("Loading video...")
    amplifier.load_video(test_video_path)

    print("Processing video...")
    amplifier.process()

    # Apply motion amplification with more reasonable parameters
    print("Applying motion amplification...")
    amplification_factor = 20  # More reasonable amplification

    # Target the specific frequency of the pulse (2 Hz) with a narrow band
    # The video pulses at 2 Hz, so we target that frequency range
    amplified_video = amplifier.amplify(
        amplification_factor=amplification_factor,
        frequency_range=(1.8, 2.2)  # Hz - narrow band around 2 Hz pulse
    )

    # Save the amplified video
    output_path = "test_square_motion_amplified.mp4"

    # Convert tensor back to video format
    import cv2
    import numpy as np

    # Get video properties
    fps = amplifier.video_fps
    if fps is None:
        fps = 30.0  # Default FPS if not available

    # Handle the shape issue - the reconstructed video might have extra dimensions
    print(f"Amplified video shape: {amplified_video.shape}")

    # Ensure we have the correct shape [N, C, H, W]
    if amplified_video.dim() == 5:  # [N, 1, 1, H, W]
        amplified_video = amplified_video.squeeze(2)  # Remove extra dimension
    elif amplified_video.dim() == 4:  # [N, C, H, W]
        pass  # Already correct
    else:
        raise ValueError(f"Unexpected amplified video shape: {amplified_video.shape}")

    print(f"Corrected amplified video shape: {amplified_video.shape}")

    height, width = amplified_video.shape[2], amplified_video.shape[3]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print("Saving amplified video...")
    for i in range(amplified_video.shape[0]):
        # Get frame and convert to numpy
        frame = amplified_video[i, 0].cpu().numpy()  # [H, W]

        # Convert from [0, 1] to [0, 255] and to uint8
        frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # Write frame
        out.write(frame)

    out.release()

    print(f"Amplified video saved to: {output_path}")
    print(f"Amplification factor: {amplification_factor}x")
    print(f"Frequency range: 1.8-2.2 Hz (targeting 2 Hz pulse)")

    # Print some statistics
    print(f"\nVideo statistics:")
    if amplifier.video is not None:
        print(f"Original video shape: {amplifier.video.shape}")
    print(f"Amplified video shape: {amplified_video.shape}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")

if __name__ == "__main__":
    test_motion_amplification()