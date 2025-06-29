import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier

def test_motion_frames():
    """
    Test motion amplification using frames that actually have motion.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== TESTING MOTION FRAMES ===")

    # Read multiple frames
    cap = cv2.VideoCapture(test_video_path)
    frames = []
    for i in range(10):  # Read 10 frames
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = gray.astype(np.float32) / 255.0
            frames.append(img)
    cap.release()

    if len(frames) < 3:
        print("Failed to read enough frames from video.")
        return

    print(f"Read {len(frames)} frames from video")

    # Find frames with motion
    print("\n1. Analyzing frame differences...")
    motion_frames = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i] - frames[i + 1])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"   Frame {i} vs {i+1}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")

        if max_diff > 0.001:  # Significant motion threshold
            motion_frames.append((i, i+1, max_diff))

    if not motion_frames:
        print("   No frames with significant motion found!")
        return

    print(f"\nFound {len(motion_frames)} frame pairs with motion:")
    for start, end, diff in motion_frames:
        print(f"   Frames {start}-{end}: Max diff = {diff:.6f}")

    # Test motion amplification on the first pair with motion
    start_frame, end_frame, _ = motion_frames[0]
    print(f"\n2. Testing motion amplification on frames {start_frame}-{end_frame}...\n")

    orig1_center = orig2_center = amp1_center = amp2_center = None
    orig_motion = amp_motion = None
    amplification_ratio = None

    try:
        # Create motion amplifier
        amplifier = MotionAmplifier()

        # Get the two frames with motion
        frame1 = torch.from_numpy(frames[start_frame]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        frame2 = torch.from_numpy(frames[end_frame]).unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]

        # Stack frames to create a mini-video
        mini_video = torch.cat([frame1, frame2], dim=0)  # [2, 1, H, W]

        # Set the video directly
        amplifier.video = mini_video
        amplifier.video_fps = 30.0  # Assume 30 FPS
        amplifier.video_shape = mini_video.shape

        # Process the mini-video
        pyramid_coeffs = amplifier.process()

        # Amplify with frequency range
        amplified_video = amplifier.amplify(
            amplification_factor=5,
            frequency_range=(0.4, 0.6)
        )

        if amplified_video is not None and len(amplified_video) > 0:
            # Check both original and amplified frames
            orig1 = frames[start_frame]
            orig2 = frames[end_frame]
            amp1 = amplified_video[0].squeeze().cpu().numpy()
            amp2 = amplified_video[1].squeeze().cpu().numpy()

            # Find square centers
            def find_square_center(img):
                white_pixels = np.where(img > 0.5)
                if len(white_pixels[0]) > 0:
                    center_y = np.mean(white_pixels[0])
                    center_x = np.mean(white_pixels[1])
                    return (center_x, center_y)
                return None

            orig1_center = find_square_center(orig1)
            orig2_center = find_square_center(orig2)
            amp1_center = find_square_center(amp1)
            amp2_center = find_square_center(amp2)

            print(f"   Original frame {start_frame}: Square center at {orig1_center}")
            print(f"   Original frame {end_frame}: Square center at {orig2_center}")
            print(f"   Amplified frame 0: Square center at {amp1_center}")
            print(f"   Amplified frame 1: Square center at {amp2_center}")

            # Check if motion was amplified
            if orig1_center and orig2_center and amp1_center and amp2_center:
                orig_motion = np.sqrt((orig2_center[0] - orig1_center[0])**2 +
                                    (orig2_center[1] - orig1_center[1])**2)
                amp_motion = np.sqrt((amp2_center[0] - amp1_center[0])**2 +
                                   (amp2_center[1] - amp1_center[1])**2)
                amplification_ratio = amp_motion / orig_motion if orig_motion != 0 else None
                print(f"   Original motion magnitude: {orig_motion:.3f}")
                print(f"   Amplified motion magnitude: {amp_motion:.3f}")
                print(f"   Amplification ratio: {amplification_ratio:.3f}x")
            else:
                print("   Could not determine all square centers for amplification ratio.")
        else:
            print("   No amplified frames returned")

    except Exception as e:
        print(f"   Error in motion amplifier test: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== SUMMARY OF AMPLIFIED MOTION TEST ===")
    print(f"Original frame {start_frame} center: {orig1_center}")
    print(f"Original frame {end_frame} center: {orig2_center}")
    print(f"Amplified frame 0 center: {amp1_center}")
    print(f"Amplified frame 1 center: {amp2_center}")
    print(f"Original motion magnitude: {orig_motion}")
    print(f"Amplified motion magnitude: {amp_motion}")
    print(f"Amplification ratio: {amplification_ratio}")

    print("\n=== MOTION FRAMES TEST COMPLETE ===")

if __name__ == "__main__":
    test_motion_frames()