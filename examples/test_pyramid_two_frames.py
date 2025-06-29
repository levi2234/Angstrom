import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.pyramid import ComplexSteerablePyramid

def test_pyramid_two_frames():
    """
    Decompose and reconstruct two consecutive frames to check if motion is preserved.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    # Read first two frames using OpenCV
    cap = cv2.VideoCapture(test_video_path)
    frames = []
    for _ in range(2):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video.")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = gray.astype(np.float32) / 255.0
        frames.append(img)
    cap.release()

    img_tensor_0 = torch.from_numpy(frames[0]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    img_tensor_1 = torch.from_numpy(frames[1]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    print(f"Frame 0 shape: {img_tensor_0.shape}, Frame 1 shape: {img_tensor_1.shape}")

    # Decompose and reconstruct both frames
    pyramid = ComplexSteerablePyramid()
    coeffs_0 = pyramid.decompose(img_tensor_0)
    recon_0 = pyramid.reconstruct(coeffs_0)
    coeffs_1 = pyramid.decompose(img_tensor_1)
    recon_1 = pyramid.reconstruct(coeffs_1)

    # Remove batch/channel dims for comparison
    orig_0 = img_tensor_0[0, 0].cpu().numpy()
    orig_1 = img_tensor_1[0, 0].cpu().numpy()
    rec_0 = recon_0[0, 0].cpu().numpy()
    rec_1 = recon_1[0, 0].cpu().numpy()

    # Compare original and reconstructed differences
    orig_diff = np.abs(orig_1 - orig_0)
    rec_diff = np.abs(rec_1 - rec_0)
    print(f"Original mean diff: {np.mean(orig_diff):.6f}, max diff: {np.max(orig_diff):.6f}")
    print(f"Reconstructed mean diff: {np.mean(rec_diff):.6f}, max diff: {np.max(rec_diff):.6f}")

    # Save for visual inspection
    cv2.imwrite("two_frames_orig_0.png", (orig_0 * 255).astype(np.uint8))
    cv2.imwrite("two_frames_orig_1.png", (orig_1 * 255).astype(np.uint8))
    cv2.imwrite("two_frames_rec_0.png", (rec_0 * 255).clip(0,255).astype(np.uint8))
    cv2.imwrite("two_frames_rec_1.png", (rec_1 * 255).clip(0,255).astype(np.uint8))
    cv2.imwrite("two_frames_orig_diff.png", (orig_diff * 255 * 10).clip(0,255).astype(np.uint8))
    cv2.imwrite("two_frames_rec_diff.png", (rec_diff * 255 * 10).clip(0,255).astype(np.uint8))
    print("Saved two_frames_orig_0.png, two_frames_orig_1.png, two_frames_rec_0.png, two_frames_rec_1.png, two_frames_orig_diff.png, two_frames_rec_diff.png")

if __name__ == "__main__":
    test_pyramid_two_frames()