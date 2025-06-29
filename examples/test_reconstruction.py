import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier

def test_reconstruction():
    """
    Test if the reconstruction process works correctly by reconstructing without amplification.
    """

    # Check if test video exists
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== TESTING RECONSTRUCTION PROCESS ===")

    # Initialize motion amplifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    amplifier = MotionAmplifier(device=device)

    # Load video
    print("\n1. Loading video...")
    amplifier.load_video(test_video_path)
    print(f"   Video shape: {amplifier.video.shape}")

    # Check original motion
    print("\n2. Original video motion check...")
    frames = amplifier.video.cpu().numpy()
    for i in [0, 25, 50, 75]:
        frame = frames[i, 0]
        white_pixels = np.where(frame > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            print(f"   Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")

    # Process video
    print("\n3. Processing video...")
    amplifier.process()

    # Test reconstruction WITHOUT any amplification (factor = 1)
    print("\n4. Testing reconstruction with amplification factor = 1...")
    try:
        # This should give us the original video back
        reconstructed_video = amplifier.amplify(
            amplification_factor=1,  # No amplification
            frequency_range=None  # No temporal filtering
        )
        print(f"   Reconstructed video shape: {reconstructed_video.shape}")

        # Check if motion is preserved
        print("   Checking motion in reconstructed video...")
        recon_frames = reconstructed_video.cpu().numpy()
        for i in [0, 25, 50, 75]:
            frame = recon_frames[i, 0]
            white_pixels = np.where(frame > 0.5)
            if len(white_pixels[0]) > 0:
                center_y = np.mean(white_pixels[0])
                center_x = np.mean(white_pixels[1])
                print(f"     Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")

        # Save reconstructed video
        save_video(recon_frames, "test_reconstruction.mp4", amplifier.video_fps or 30.0)
        print("   Saved: test_reconstruction.mp4")

        # Compare with original
        print("\n5. Comparing original vs reconstructed...")
        original_centers = []
        reconstructed_centers = []

        for i in range(min(10, frames.shape[0])):  # Check first 10 frames
            # Original
            frame = frames[i, 0]
            white_pixels = np.where(frame > 0.5)
            if len(white_pixels[0]) > 0:
                orig_center = (np.mean(white_pixels[1]), np.mean(white_pixels[0]))
                original_centers.append(orig_center)

            # Reconstructed
            frame = recon_frames[i, 0]
            white_pixels = np.where(frame > 0.5)
            if len(white_pixels[0]) > 0:
                recon_center = (np.mean(white_pixels[1]), np.mean(white_pixels[0]))
                reconstructed_centers.append(recon_center)

        print(f"   Original centers: {original_centers[:3]}...")
        print(f"   Reconstructed centers: {reconstructed_centers[:3]}...")

        if len(original_centers) == len(reconstructed_centers):
            differences = []
            for orig, recon in zip(original_centers, reconstructed_centers):
                diff = np.sqrt((orig[0] - recon[0])**2 + (orig[1] - recon[1])**2)
                differences.append(diff)
            avg_diff = np.mean(differences)
            print(f"   Average position difference: {avg_diff:.3f} pixels")

            if avg_diff < 1.0:
                print("   ✓ Reconstruction is working correctly!")
            else:
                print("   ✗ Reconstruction has significant errors!")
        else:
            print("   ✗ Could not compare centers - different number of frames!")

    except Exception as e:
        print(f"   Error in reconstruction test: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== RECONSTRUCTION TEST COMPLETE ===")

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
    test_reconstruction()