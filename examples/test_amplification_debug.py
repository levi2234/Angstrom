import os
import sys
import torch
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.pyramid import ComplexSteerablePyramid
from angstrom.processing.phase import extract_phase, amplify_phase, extract_amplitude, reconstruct_from_amplitude_and_phase, amplify_motion_phase

def test_amplification_debug():
    """
    Debug the phase amplification process step by step.
    """
    test_video_path = "test_subtle_motion.mp4"
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return

    print("=== DEBUGGING PHASE AMPLIFICATION ===")

    # Read two consecutive frames with motion
    cap = cv2.VideoCapture(test_video_path)
    frames = []
    for i in range(5):  # Read 5 frames
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = gray.astype(np.float32) / 255.0
            frames.append(img)
    cap.release()

    # Find frames with motion
    motion_frames = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i] - frames[i + 1])
        max_diff = np.max(diff)
        if max_diff > 0.001:
            motion_frames.append((i, i+1, max_diff))

    if not motion_frames:
        print("No frames with motion found!")
        return

    start_frame, end_frame, _ = motion_frames[0]
    print(f"Testing frames {start_frame} and {end_frame} with motion")

    # Process both frames
    pyramid = ComplexSteerablePyramid()

    frame1_tensor = torch.from_numpy(frames[start_frame]).unsqueeze(0).unsqueeze(0)
    frame2_tensor = torch.from_numpy(frames[end_frame]).unsqueeze(0).unsqueeze(0)

    # Decompose both frames
    coeffs1 = pyramid.decompose(frame1_tensor)
    coeffs2 = pyramid.decompose(frame2_tensor)

    # Get filters and crops for reconstruction
    h, w = frames[start_frame].shape
    dummy_image = np.zeros((h, w))
    filters, crops = pyramid.pyr.get_filters(dummy_image, cropped=True)

    print("\n1. Testing original reconstruction...")
    # Reconstruct original frames
    recon1_orig = pyramid.pyr.reconstruct_image(coeffs1, filters, crops, full=False)
    recon2_orig = pyramid.pyr.reconstruct_image(coeffs2, filters, crops, full=False)

    # Find square centers
    def find_square_center(img):
        white_pixels = np.where(img > 0.5)
        if len(white_pixels[0]) > 0:
            center_y = np.mean(white_pixels[0])
            center_x = np.mean(white_pixels[1])
            return (center_x, center_y)
        return None

    orig1_center = find_square_center(recon1_orig)
    orig2_center = find_square_center(recon2_orig)
    print(f"   Original frame {start_frame} center: {orig1_center}")
    print(f"   Original frame {end_frame} center: {orig2_center}")

    orig_motion = 0.0  # Default value
    if orig1_center and orig2_center:
        orig_motion = np.sqrt((orig2_center[0] - orig1_center[0])**2 +
                            (orig2_center[1] - orig1_center[1])**2)
        print(f"   Original motion magnitude: {orig_motion:.3f}")

    print("\n2. Testing phase amplification...")
    # Test different amplification factors
    for factor in [1, 5, 10, 20]:
        print(f"\n   Testing amplification factor: {factor}")

        # Use frame 1 as reference, amplify motion from frame 1 to frame 2
        phase1 = extract_phase(coeffs1)
        phase2 = extract_phase(coeffs2)
        amplitude1 = extract_amplitude(coeffs1)
        amplitude2 = extract_amplitude(coeffs2)

        # Calculate phase difference (motion) from frame 1 to frame 2
        motion_phase = []
        for p1_level, p2_level in zip(phase1, phase2):
            if isinstance(p1_level, list):
                motion_level = []
                for p1_band, p2_band in zip(p1_level, p2_level):
                    if isinstance(p1_band, torch.Tensor) and isinstance(p2_band, torch.Tensor):
                        # Calculate phase difference (motion)
                        motion_band = p2_band - p1_band
                        motion_level.append(motion_band)
                    else:
                        motion_level.append(torch.zeros_like(p1_band) if isinstance(p1_band, torch.Tensor) else p1_band)
                motion_phase.append(motion_level)
            else:
                if isinstance(p1_level, torch.Tensor):
                    motion_phase.append(torch.zeros_like(p1_level))
                else:
                    motion_phase.append(p1_level)

        # Amplify the motion phase
        amplified_phase = amplify_motion_phase(phase1, motion_phase, factor)

        # Reconstruct amplified frame 2
        recombined = reconstruct_from_amplitude_and_phase(amplitude2, amplified_phase)
        recon2_amp = pyramid.pyr.reconstruct_image(recombined, filters, crops, full=False)

        # Find amplified square centers (frame 1 stays the same, frame 2 is amplified)
        amp1_center = orig1_center  # Frame 1 is the reference, doesn't change
        amp2_center = find_square_center(recon2_amp)

        print(f"     Amplified frame {start_frame} center: {amp1_center}")
        print(f"     Amplified frame {end_frame} center: {amp2_center}")

        if amp1_center and amp2_center:
            amp_motion = np.sqrt((amp2_center[0] - amp1_center[0])**2 +
                               (amp2_center[1] - amp1_center[1])**2)
            print(f"     Amplified motion magnitude: {amp_motion:.3f}")

            if orig_motion > 0:
                ratio = amp_motion / orig_motion
                print(f"     Amplification ratio: {ratio:.3f}x")
            else:
                print(f"     No original motion to amplify")

    print("\n3. Testing phase values...")
    # Check if phase values are changing
    phase1 = extract_phase(coeffs1)
    phase2 = extract_phase(coeffs2)

    print(f"   Phase structure: {type(phase1)}")
    if isinstance(phase1, list):
        print(f"   Number of phase levels: {len(phase1)}")

        # Check phase differences between frames
        for i, (p1_level, p2_level) in enumerate(zip(phase1, phase2)):
            if isinstance(p1_level, list):
                print(f"     Level {i}: {len(p1_level)} bands")
                for j, (p1_band, p2_band) in enumerate(zip(p1_level, p2_level)):
                    if isinstance(p1_band, torch.Tensor):
                        phase_diff = torch.abs(p1_band - p2_band)
                        max_diff = torch.max(phase_diff).item()
                        mean_diff = torch.mean(phase_diff).item()
                        print(f"       Band {j}: Max phase diff = {max_diff:.6f}, Mean = {mean_diff:.6f}")

    print("\n=== AMPLIFICATION DEBUG COMPLETE ===")

if __name__ == "__main__":
    test_amplification_debug()