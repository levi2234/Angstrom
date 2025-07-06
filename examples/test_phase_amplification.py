import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.pyramid import ComplexSteerablePyramid
from angstrom.processing.phase import extract_phase, extract_amplitude, reconstruct_from_amplitude_and_phase

def test_phase_amplification():
    """
    Test if phase amplification is causing the spatial positioning issue.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== TESTING PHASE AMPLIFICATION ===")

    # Read first frame
    cap = cv2.VideoCapture(test_video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read frame from video.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = gray.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    print(f"Original frame shape: {img_tensor.shape}")

    # Decompose
    pyramid = ComplexSteerablePyramid()
    coeffs = pyramid.decompose(img_tensor)

    # Get filters and crops for reconstruction
    h, w = img.shape
    dummy_image = np.zeros((h, w))
    filters, crops = pyramid.pyr.get_filters(dummy_image, cropped=True)

    # Extract phase and amplitude for all tests
    phase = extract_phase(coeffs)
    amplitude = extract_amplitude(coeffs)

    # Test 1: Reconstruct without any phase manipulation
    print("\n1. Testing reconstruction without phase manipulation...")
    try:
        recon_original = pyramid.pyr.reconstruct_image(coeffs, filters, crops, full=False)

        # Find square center
        white_pixels = np.where(recon_original > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            print(f"   Square center (no phase manipulation): ({center_x:.1f}, {center_y:.1f})")
        else:
            print("   No square found")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Reconstruct with phase amplification factor = 1 (should be same as original)
    print("\n2. Testing reconstruction with amplification factor = 1...")
    try:
        # Amplify phase by factor 1 (should be no change)
        amplified_phase = amplify_phase(phase, 1)

        # Reconstruct from amplitude and amplified phase
        recombined = reconstruct_from_amplitude_and_phase(amplitude, amplified_phase)

        # Reconstruct frame
        recon_amplified = pyramid.pyr.reconstruct_image(recombined, filters, crops, full=False)

        # Find square center
        white_pixels = np.where(recon_amplified > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            print(f"   Square center (amplification factor = 1): ({center_x:.1f}, {center_y:.1f})")
        else:
            print("   No square found")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Reconstruct with phase amplification factor = 10
    print("\n3. Testing reconstruction with amplification factor = 10...")
    try:
        # Amplify phase by factor 10
        amplified_phase = amplify_phase(phase, 10)

        # Reconstruct from amplitude and amplified phase
        recombined = reconstruct_from_amplitude_and_phase(amplitude, amplified_phase)

        # Reconstruct frame
        recon_amplified = pyramid.pyr.reconstruct_image(recombined, filters, crops, full=False)

        # Find square center
        white_pixels = np.where(recon_amplified > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            print(f"   Square center (amplification factor = 10): ({center_x:.1f}, {center_y:.1f})")
        else:
            print("   No square found")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Check if the issue is in the phase extraction/reconstruction process
    print("\n4. Testing phase extraction and reconstruction process...")
    try:
        # Print some info about the phase structure
        print(f"   Phase structure: {type(phase)}")
        if isinstance(phase, list):
            print(f"   Number of phase levels: {len(phase)}")
            for i, level in enumerate(phase):
                if isinstance(level, list):
                    print(f"     Level {i}: {len(level)} bands")
                else:
                    print(f"     Level {i}: {type(level)}")

        # Check if any phase values are NaN or inf
        def check_phase_values(phase_list):
            for i, level in enumerate(phase_list):
                if isinstance(level, list):
                    for j, band in enumerate(level):
                        if torch.isnan(band).any() or torch.isinf(band).any():
                            print(f"     Level {i}, Band {j}: Contains NaN or Inf values")
                elif isinstance(level, torch.Tensor):
                    if torch.isnan(level).any() or torch.isinf(level).any():
                        print(f"     Level {i}: Contains NaN or Inf values")

        check_phase_values(phase)

    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== PHASE AMPLIFICATION TEST COMPLETE ===")

if __name__ == "__main__":
    test_phase_amplification()