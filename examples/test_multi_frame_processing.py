import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.pyramid import ComplexSteerablePyramid
from angstrom.processing.phase import extract_phase,  extract_amplitude, reconstruct_from_amplitude_and_phase

def test_multi_frame_processing():
    """
    Test if the issue is in multi-frame processing by comparing single vs multi-frame results.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== TESTING MULTI-FRAME PROCESSING ===")

    # Read multiple frames
    cap = cv2.VideoCapture(test_video_path)
    frames = []
    for i in range(5):  # Read 5 frames
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = gray.astype(np.float32) / 255.0
            frames.append(img)
    cap.release()

    if len(frames) < 2:
        print("Failed to read enough frames from video.")
        return

    print(f"Read {len(frames)} frames from video")

    # Test 1: Process each frame individually
    print("\n1. Testing individual frame processing...")
    pyramid = ComplexSteerablePyramid()

    for i, frame in enumerate(frames):
        img_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)

        # Decompose
        coeffs = pyramid.decompose(img_tensor)

        # Extract phase and amplitude
        phase = extract_phase(coeffs)
        amplitude = extract_amplitude(coeffs)



        # Reconstruct
        recombined = reconstruct_from_amplitude_and_phase(amplitude, amplified_phase)

        # Get filters and crops
        h, w = frame.shape
        dummy_image = np.zeros((h, w))
        filters, crops = pyramid.pyr.get_filters(dummy_image, cropped=True)

        # Reconstruct frame
        recon = pyramid.pyr.reconstruct_image(recombined, filters, crops, full=False)

        # Find square center
        white_pixels = np.where(recon > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            print(f"   Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")
        else:
            print(f"   Frame {i}: No square found")

    # Test 2: Process frames as a batch (simulate multi-frame processing)
    print("\n2. Testing batch frame processing...")
    try:
        # Stack frames into a batch
        batch_tensors = []
        for frame in frames:
            img_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
            batch_tensors.append(img_tensor)

        # Process each frame in the batch
        for i, img_tensor in enumerate(batch_tensors):
            # Decompose
            coeffs = pyramid.decompose(img_tensor)

            # Extract phase and amplitude
            phase = extract_phase(coeffs)
            amplitude = extract_amplitude(coeffs)

            # Amplify phase
            amplified_phase = amplify_phase(phase, 5)

            # Reconstruct
            recombined = reconstruct_from_amplitude_and_phase(amplitude, amplified_phase)

            # Get filters and crops
            h, w = frames[i].shape
            dummy_image = np.zeros((h, w))
            filters, crops = pyramid.pyr.get_filters(dummy_image, cropped=True)

            # Reconstruct frame
            recon = pyramid.pyr.reconstruct_image(recombined, filters, crops, full=False)

            # Find square center
            white_pixels = np.where(recon > 0.5)
            if len(white_pixels[0]) > 0:
                center_y = np.mean(white_pixels[0])
                center_x = np.mean(white_pixels[1])
                print(f"   Batch Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")
            else:
                print(f"   Batch Frame {i}: No square found")

    except Exception as e:
        print(f"   Error in batch processing: {e}")

    # Test 3: Check if there are differences between consecutive frames
    print("\n3. Checking frame differences...")
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i] - frames[i + 1])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"   Frame {i} vs {i+1}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")

        # Check if there's motion in the square region
        square_region = frames[i][20:30, 20:30]  # Approximate square region
        square_region_next = frames[i+1][20:30, 20:30]
        square_diff = np.abs(square_region - square_region_next)
        square_max_diff = np.max(square_diff)
        print(f"   Square region diff: Max = {square_max_diff:.6f}")

    # Test 4: Check if the issue is in the motion amplifier's frame handling
    print("\n4. Testing motion amplifier frame handling...")
    try:
        from angstrom.core.motion_amplifier import MotionAmplifier

        # Create a simple motion amplifier
        amplifier = MotionAmplifier()

        # Process just the first two frames by creating a mini-video
        frame1 = torch.from_numpy(frames[0]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        frame2 = torch.from_numpy(frames[1]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Stack frames to create a mini-video
        mini_video = torch.cat([frame1, frame2], dim=0)  # [2, 1, H, W]

        # Set the video directly (bypassing load_video)
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
            # Check the first amplified frame
            recon = amplified_video[0].squeeze().cpu().numpy()

            # Find square center
            white_pixels = np.where(recon > 0.5)
            if len(white_pixels[0]) > 0:
                center_y = np.mean(white_pixels[0])
                center_x = np.mean(white_pixels[1])
                print(f"   Amplified frame: Square center at ({center_x:.1f}, {center_y:.1f})")
            else:
                print(f"   Amplified frame: No square found")
        else:
            print("   No amplified frames returned")

    except Exception as e:
        print(f"   Error in motion amplifier test: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== MULTI-FRAME PROCESSING TEST COMPLETE ===")

if __name__ == "__main__":
    test_multi_frame_processing()