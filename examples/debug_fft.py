import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.phase import amplify_temporal_fft_band

def debug_fft():
    """
    Debug the FFT amplification process step by step.
    """
    print("=== DEBUGGING FFT AMPLIFICATION ===")

    # Create a simple test signal
    T, H, W = 100, 10, 10
    fps = 30.0
    freq = 0.3  # Hz - use a frequency that matches the FFT bins

    # Create temporal signal: [T, H, W]
    temporal_signal = np.zeros((T, H, W))
    for t in range(T):
        offset = 0.5 * np.sin(2 * np.pi * freq * t / fps)
        temporal_signal[t, :, :] = offset

    print(f"Signal shape: {temporal_signal.shape}")
    print(f"Signal range: [{temporal_signal.min():.3f}, {temporal_signal.max():.3f}]")

    # Step 1: Apply FFT along temporal axis
    fft_temporal = np.fft.fft(temporal_signal, axis=0)
    print(f"FFT shape: {fft_temporal.shape}")
    print(f"FFT range: [{np.abs(fft_temporal).min():.3f}, {np.abs(fft_temporal).max():.3f}]")

    # Step 2: Calculate frequency bins
    freqs = np.fft.fftfreq(T, 1/fps)
    print(f"Frequency bins: {freqs[:10]}...")  # First 10 frequencies

    # Step 3: Create amplification mask
    amplification_factor = 5
    frequency_range = (0.25, 0.35)  # Target the 0.3 Hz frequency
    low_freq, high_freq = frequency_range

    amplification_mask = np.ones(T)
    freq_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    amplification_mask[freq_mask] = amplification_factor

    print(f"Frequency range: {frequency_range}")
    print(f"Frequencies in range: {freqs[freq_mask]}")
    print(f"Amplification mask: {amplification_mask[:10]}...")  # First 10 values

    # Step 4: Apply amplification mask
    amplification_mask = amplification_mask.reshape(T, 1, 1)
    amplified_fft = fft_temporal * amplification_mask

    print(f"Amplified FFT range: [{np.abs(amplified_fft).min():.3f}, {np.abs(amplified_fft).max():.3f}]")

    # Step 5: Apply inverse FFT
    amplified_temporal = np.real(np.fft.ifft(amplified_fft, axis=0))

    print(f"Amplified signal range: [{amplified_temporal.min():.3f}, {amplified_temporal.max():.3f}]")

    # Check amplification
    original_amplitude = np.max(np.abs(temporal_signal))
    amplified_amplitude = np.max(np.abs(amplified_temporal))
    actual_amplification = amplified_amplitude / original_amplitude

    print(f"Original amplitude: {original_amplitude:.3f}")
    print(f"Amplified amplitude: {amplified_amplitude:.3f}")
    print(f"Amplification ratio: {actual_amplification:.3f}x")

    # Check specific frequency components
    center_pixel = temporal_signal[:, H//2, W//2]
    fft_center = np.fft.fft(center_pixel)

    print(f"\nCenter pixel FFT magnitudes:")
    for i in range(min(10, len(freqs))):
        print(f"  Freq {freqs[i]:.3f} Hz: {np.abs(fft_center[i]):.3f}")

    # Check if our target frequency is being amplified
    target_freq_idx = np.argmin(np.abs(freqs - freq))
    print(f"\nTarget frequency {freq} Hz at index {target_freq_idx}")
    print(f"Original magnitude at target: {np.abs(fft_center[target_freq_idx]):.3f}")
    print(f"Amplified magnitude at target: {np.abs(fft_center[target_freq_idx] * amplification_mask[target_freq_idx, 0, 0]):.3f}")

if __name__ == "__main__":
    debug_fft()