import numpy as np
from angstrom.processing.temporal_filter import temporal_ideal_filter

def temporal_ideal_filter(data, fl, fh, fps):
    fft = np.fft.fft(data, axis=0)
    frequencies = np.fft.fftfreq(data.shape[0], d=1.0/fps)
    mask = (np.abs(frequencies) >= fl) & (np.abs(frequencies) <= fh)
    fft[~mask] = 0
    filtered = np.fft.ifft(fft, axis=0)
    return np.real(filtered)
