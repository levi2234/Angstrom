import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier
from angstrom.processing.phase import extract_phase, amplify_phase, extract_amplitude, reconstruct_from_amplitude_and_phase

def test_pipeline_steps():
    """
    Test each step of the motion amplification pipeline separately.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== TESTING PIPELINE STEPS ===")

    # Initialize motion amplifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    amplifier = MotionAmplifier(device=device)

    # Load video
    print("\n1. Loading video...")
    amplifier.load_video(test_video_path)
    print(f"   Video shape: {amplifier.video.shape}")

    # Process video (decompose all frames)
    print("\n2. Processing video (decompose)...")
    amplifier.process()
    print(f"   Number of frame coefficients: {len(amplifier.pyramid_coeffs)}")

    # Test step 3: Extract phase and amplitude from first frame
    print("\n3. Testing phase/amplitude extraction...")
    frame_coeffs = amplifier.pyramid_coeffs[0]
    phase = extract_phase(frame_coeffs)
    amplitude = extract_amplitude(frame_coeffs)
    print(f"   Phase structure: {type(phase)}")
    print(f"   Amplitude structure: {type(amplitude)}")

    # Test step 4: Amplify phase
    print("\n4. Testing phase amplification...")
    amplification_factor = 10
    amplified_phase = amplify_phase(phase, amplification_factor)
    print(f"   Amplified phase structure: {type(amplified_phase)}")

    # Test step 5: Reconstruct from amplitude and amplified phase
    print("\n5. Testing reconstruction from amplitude and phase...")
    recombined = reconstruct_from_amplitude_and_phase(amplitude, amplified_phase)
    print(f"   Recombined structure: {type(recombined)}")

    # Test step 6: Reconstruct frame
    print("\n6. Testing frame reconstruction...")
    reconstructed = amplifier.pyramid.reconstruct(recombined)
    print(f"   Reconstructed shape: {reconstructed.shape}")

    # Test step 7: Compare original vs reconstructed for first frame
    print("\n7. Comparing original vs reconstructed (first frame)...")
    original_frame = amplifier.video[0, 0].cpu().numpy()
    recon_frame = reconstructed[0, 0].cpu().numpy()

    # Find square centers
    orig_white = np.where(original_frame > 0.5)
    recon_white = np.where(recon_frame > 0.5)

    if len(orig_white[0]) > 0 and len(recon_white[0]) > 0:
        orig_center = (np.mean(orig_white[1]), np.mean(orig_white[0]))
        recon_center = (np.mean(recon_white[1]), np.mean(recon_white[0]))
        print(f"   Original square center: {orig_center}")
        print(f"   Reconstructed square center: {recon_center}")

        diff = np.sqrt((orig_center[0] - recon_center[0])**2 + (orig_center[1] - recon_center[1])**2)
        print(f"   Position difference: {diff:.3f} pixels")

        if diff < 1.0:
            print("   ✓ Single frame amplification works!")
        else:
            print("   ✗ Single frame amplification has issues!")
    else:
        print("   ✗ Could not find square in one or both frames!")

    # Test step 8: Test the full amplify method on first few frames
    print("\n8. Testing full amplify method (first 5 frames)...")
    try:
        # Create a small test video with just 5 frames
        test_video = amplifier.video[:5].clone()
        amplifier.video = test_video

        # Process just these frames
        amplifier.pyramid_coeffs = []
        for i in range(5):
            frame = test_video[i:i+1]
            coeffs = amplifier.pyramid.decompose(frame)
            amplifier.pyramid_coeffs.append(coeffs)

        # Amplify without temporal filtering
        amplified_video = amplifier.amplify(
            amplification_factor=amplification_factor,
            frequency_range=None
        )
        print(f"   Amplified video shape: {amplified_video.shape}")

        # Check motion in amplified video
        print("   Checking motion in amplified video...")
        amp_frames = amplified_video.cpu().numpy()
        for i in [0, 2, 4]:
            frame = amp_frames[i, 0]
            white_pixels = np.where(frame > 0.5)
            if len(white_pixels[0]) > 0:
                center_y = np.mean(white_pixels[0])
                center_x = np.mean(white_pixels[1])
                print(f"     Frame {i}: Square center at ({center_x:.1f}, {center_y:.1f})")

    except Exception as e:
        print(f"   Error in full amplify method: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== PIPELINE STEPS TEST COMPLETE ===")

if __name__ == "__main__":
    test_pipeline_steps()