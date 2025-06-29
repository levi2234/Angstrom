import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.processing.phase import amplify_temporal_fft_band

def test_fft_amplification():
    """
    Test the FFT-based temporal amplification with a simple sinusoidal signal.
    """
    print("=== TESTING FFT-BASED AMPLIFICATION ===")

    # Create a simple test signal: sinusoidal motion
    T, H, W = 100, 10, 10
    fps = 30.0
    freq = 0.1  # Hz
    amplitude = 0.5  # Small amplitude for subtle motion

    # Create temporal signal: [T, H, W]
    temporal_signal = np.zeros((T, H, W))
    for t in range(T):
        # Create a moving pattern
        offset = amplitude * np.sin(2 * np.pi * freq * t / fps)
        # Add the offset to create motion
        temporal_signal[t, :, :] = offset

    print(f"Original signal shape: {temporal_signal.shape}")
    print(f"Original signal range: [{temporal_signal.min():.3f}, {temporal_signal.max():.3f}]")

    # Test amplification
    amplification_factor = 5
    frequency_range = (0.05, 0.15)  # Target the motion frequency

    print(f"\nAmplifying with factor {amplification_factor} in frequency range {frequency_range}")

    # Apply FFT amplification
    amplified_signal = amplify_temporal_fft_band(
        temporal_signal, amplification_factor, frequency_range, fps
    )

    print(f"Amplified signal range: [{amplified_signal.min():.3f}, {amplified_signal.max():.3f}]")

    # Check if amplification worked
    original_amplitude = np.max(np.abs(temporal_signal))
    amplified_amplitude = np.max(np.abs(amplified_signal))
    actual_amplification = amplified_amplitude / original_amplitude

    print(f"Original amplitude: {original_amplitude:.3f}")
    print(f"Amplified amplitude: {amplified_amplitude:.3f}")
    print(f"Actual amplification ratio: {actual_amplification:.3f}x")

    # Plot the results
    center_pixel = temporal_signal[:, H//2, W//2]
    amplified_center = amplified_signal[:, H//2, W//2]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(center_pixel, label='Original')
    plt.plot(amplified_center, label='Amplified')
    plt.title('Temporal Signal at Center Pixel')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # FFT of both signals
    fft_original = np.fft.fft(center_pixel)
    fft_amplified = np.fft.fft(amplified_center)
    freqs = np.fft.fftfreq(T, 1/fps)

    plt.plot(freqs[:T//2], np.abs(fft_original)[:T//2], label='Original')
    plt.plot(freqs[:T//2], np.abs(fft_amplified)[:T//2], label='Amplified')
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fft_amplification_test.png')
    plt.show()

    print(f"\nTest completed! Amplification ratio: {actual_amplification:.3f}x")
    if actual_amplification > 1.5:
        print("✅ FFT amplification is working correctly!")
    else:
        print("❌ FFT amplification is not working as expected.")

    return actual_amplification

if __name__ == "__main__":
    test_fft_amplification()