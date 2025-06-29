import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier

def debug_motion_amplification():
    """
    Debug motion amplification to understand why videos appear frozen.
    """

    # Check if test video exists
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== DEBUGGING MOTION AMPLIFICATION ===")

    # Initialize motion amplifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    amplifier = MotionAmplifier(device=device)

    # Load video
    print("\n1. Loading video...")
    amplifier.load_video(test_video_path)
    print(f"   Video shape: {amplifier.video.shape}")
    print(f"   Video FPS: {amplifier.video_fps}")

    # Check a few frames to see if there's actual motion
    print("\n2. Checking for motion in original video...")
    frames = amplifier.video.cpu().numpy()
    for i in [0, 25, 50, 75]:
        frame = frames[i, 0]  # [H, W]
        # Find the white square
        white_pixels = np.where(frame > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            print(f"   Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")

    # Process video
    print("\n3. Processing video...")
    amplifier.process()

    # Check if we have pyramid coefficients
    print(f"   Number of frame coefficients: {len(amplifier.pyramid_coeffs)}")
    if amplifier.pyramid_coeffs:
        print(f"   First frame coefficient structure: {type(amplifier.pyramid_coeffs[0])}")
        if isinstance(amplifier.pyramid_coeffs[0], list):
            print(f"   Number of pyramid levels: {len(amplifier.pyramid_coeffs[0])}")

    # Test amplification without temporal filtering first
    print("\n4. Testing amplification WITHOUT temporal filtering...")
    try:
        amplified_video_no_filter = amplifier.amplify(
            amplification_factor=10,
            frequency_range=None  # No temporal filtering
        )
        print(f"   Amplified video shape (no filter): {amplified_video_no_filter.shape}")

        # Check if there's motion in the amplified video
        print("   Checking for motion in amplified video (no filter)...")
        amp_frames = amplified_video_no_filter.cpu().numpy()
        for i in [0, 25, 50, 75]:
            frame = amp_frames[i, 0]  # [H, W]
            # Find the white square
            white_pixels = np.where(frame > 0.5)
            if len(white_pixels[0]) > 0:
                center_y = np.mean(white_pixels[0])
                center_x = np.mean(white_pixels[1])
                print(f"     Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")

        # Save this version
        save_video(amp_frames, "debug_no_filter.mp4", amplifier.video_fps or 30.0)
        print("   Saved: debug_no_filter.mp4")

    except Exception as e:
        print(f"   Error in amplification without filter: {e}")

    # Test with temporal filtering
    print("\n5. Testing amplification WITH temporal filtering...")
    try:
        amplified_video_with_filter = amplifier.amplify(
            amplification_factor=10,
            frequency_range=(1.8, 2.2)  # Hz
        )
        print(f"   Amplified video shape (with filter): {amplified_video_with_filter.shape}")

        # Check if there's motion in the amplified video
        print("   Checking for motion in amplified video (with filter)...")
        amp_frames = amplified_video_with_filter.cpu().numpy()
        for i in [0, 25, 50, 75]:
            frame = amp_frames[i, 0]  # [H, W]
            # Find the white square
            white_pixels = np.where(frame > 0.5)
            if len(white_pixels[0]) > 0:
                center_y = np.mean(white_pixels[0])
                center_x = np.mean(white_pixels[1])
                print(f"     Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")

        # Save this version
        save_video(amp_frames, "debug_with_filter.mp4", amplifier.video_fps or 30.0)
        print("   Saved: debug_with_filter.mp4")

    except Exception as e:
        print(f"   Error in amplification with filter: {e}")

    print("\n=== DEBUG COMPLETE ===")

def save_video(frames, output_path, fps):
    """Helper function to save video frames."""
    height, width = frames.shape[2], frames.shape[3]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    for i in range(frames.shape[0]):
        frame = frames[i, 0]  # [H, W]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        out.write(frame)

    out.release()

if __name__ == "__main__":
    debug_motion_amplification()