from scipy.signal import butter, filtfilt
import numpy as np


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y


def temporal_ideal_filter(data, lowcut, highcut, fs):
    """
    Apply an ideal bandpass filter to the data along the time dimension using FFT.

    Args:
        data (np.ndarray): Input array with time as the first dimension (T, ...).
        lowcut (float): Lower frequency cutoff in Hz.
        highcut (float): Upper frequency cutoff in Hz.
        fs (float): Sampling rate (frames per second).

    Returns:
        np.ndarray: Filtered data, real part only.
    """
    fft = np.fft.fft(data, axis=0)
    frequencies = np.fft.fftfreq(data.shape[0], d=1.0 / fs)
    mask = (np.abs(frequencies) >= lowcut) & (np.abs(frequencies) <= highcut)
    fft[~mask] = 0
    filtered = np.fft.ifft(fft, axis=0)
    return np.real(filtered)
