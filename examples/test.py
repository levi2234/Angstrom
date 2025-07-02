import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

def build_gaussian_pyramid(img, levels=3):
    pyramid = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def reconstruct_from_gaussian_pyramid(pyramid):
    img = pyramid[-1]
    for lower in reversed(pyramid[:-1]):
        img = cv2.pyrUp(img)
        if img.shape != lower.shape:
            img = cv2.resize(img, (lower.shape[1], lower.shape[0]))
        img += lower
    return img

def butter_bandpass(low, high, fs, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_temporal_filter(data, low, high, fps):
    b, a = butter_bandpass(low, high, fps)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

def phase_amplify_video(video_path, low=0.4, high=1.0, amplification=10, levels=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    # 1. Read all frames into memory
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)

    frames = np.stack(frames)  # shape: (T, H, W)
    T, H, W = frames.shape

    # 2. Decompose each frame into pyramid (lowest level only, for simplicity)
    pyramid_sequence = []
    for t in range(T):
        pyr = build_gaussian_pyramid(frames[t], levels)
        pyramid_sequence.append(pyr[-1])  # smallest level

    pyramid_sequence = np.stack(pyramid_sequence)  # shape: (T, h, w)

    # 3. Compute temporal phase variation using Hilbert transform (better than FFT for local phase)
    analytic_signal = hilbert(pyramid_sequence, axis=0)
    phase_sequence = np.angle(analytic_signal)

    # 4. Bandpass filter the temporal signal
    filtered = apply_temporal_filter(phase_sequence, low, high, fps)

    # Debug: print stats and show a filtered frame
    print("Filtered stats: mean=", np.mean(filtered), "std=", np.std(filtered))
    print("Amplified stats: mean=", np.mean(filtered * amplification), "std=", np.std(filtered * amplification))
    cv2.imshow("Filtered (first frame)", np.clip((filtered[0] - filtered[0].min()) / (filtered[0].ptp() + 1e-8), 0, 1))
    cv2.waitKey(500)
    cv2.destroyWindow("Filtered (first frame)")

    # 5. Amplify phase and reconstruct
    amplified = pyramid_sequence + amplification * filtered

    # 6. Reconstruct full frame sequence from pyramid base
    output_frames = []
    for t in range(T):
        # Create dummy pyramid with only the amplified base
        dummy_pyramid = [np.zeros_like(frames[t])] * (levels - 1) + [amplified[t]]
        out = reconstruct_from_gaussian_pyramid(dummy_pyramid)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        output_frames.append(out)

    # 7. Display or save video
    for out in output_frames:
        cv2.imshow("Amplified Motion", out)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Example usage:
phase_amplify_video("C:/Users/levi2/Desktop/Projects/Angstrom/src/angstrom/data/testvideos/turtleshort.mp4", low=0.5, high=4, amplification=15, levels=4)
