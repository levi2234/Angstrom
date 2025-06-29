import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.pyramid import ComplexSteerablePyramid

def test_pyramid_identity():
    """
    Decompose and immediately reconstruct a single frame to test pyramid identity.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    # Read first frame using OpenCV
    cap = cv2.VideoCapture(test_video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read frame from video.")
        return
    # Convert to grayscale and normalize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = gray.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    print(f"Original frame shape: {img_tensor.shape}")

    # Decompose and reconstruct
    pyramid = ComplexSteerablePyramid()
    coeffs = pyramid.decompose(img_tensor)
    recon = pyramid.reconstruct(coeffs)
    print(f"Reconstructed frame shape: {recon.shape}")

    # Remove batch/channel dims for comparison
    orig = img_tensor[0, 0].cpu().numpy()
    rec = recon[0, 0].cpu().numpy()

    # Compare
    diff = np.abs(orig - rec)
    print(f"Mean absolute difference: {np.mean(diff):.6f}")
    print(f"Max absolute difference: {np.max(diff):.6f}")

    # Save for visual inspection
    cv2.imwrite("pyramid_identity_original.png", (orig * 255).astype(np.uint8))
    cv2.imwrite("pyramid_identity_recon.png", (rec * 255).clip(0,255).astype(np.uint8))
    cv2.imwrite("pyramid_identity_diff.png", (diff * 255 * 10).clip(0,255).astype(np.uint8))
    print("Saved pyramid_identity_original.png, pyramid_identity_recon.png, pyramid_identity_diff.png")

if __name__ == "__main__":
    test_pyramid_identity()