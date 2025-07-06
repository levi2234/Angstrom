# angstrom/processing/phase.py

import torch
import numpy as np


def extract_phase(coeffs):
    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr
    phase_coeffs = []
    for level in coeffs:
        if isinstance(level, list):
            phase_coeffs.append([np.angle(to_numpy(band)) for band in level])
        else:
            phase_coeffs.append(np.angle(to_numpy(level)))
    return phase_coeffs


def extract_amplitude(coeffs):
    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr
    amplitude_coeffs = []
    for level in coeffs:
        if isinstance(level, list):
            amplitude_coeffs.append([np.abs(to_numpy(band)) for band in level])
        else:
            amplitude_coeffs.append(np.abs(to_numpy(level)))
    return amplitude_coeffs


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return arr


def reconstruct_from_amplitude_and_phase(amplitude_coeffs, phase_coeffs):
    """
    Combine amplitude and phase back into complex steerable pyramid coefficients.
    All operations are done in numpy.
    """
    recombined = []
    for amp_level, phase_level in zip(amplitude_coeffs, phase_coeffs):
        if isinstance(amp_level, list):
            complex_level = []
            for amp, phase in zip(amp_level, phase_level):
                amp = to_numpy(amp)
                phase = to_numpy(phase)
                if np.isnan(amp).any() or np.isinf(amp).any():
                    raise ValueError(
                        "Amplitude coefficients contain NaN or infinite values")
                if np.isnan(phase).any() or np.isinf(phase).any():
                    raise ValueError(
                        "Phase coefficients contain NaN or infinite values")
                phase = np.clip(phase, -np.pi, np.pi)
                complex_coeff = amp * np.exp(1j * phase)
                complex_level.append(complex_coeff)
            recombined.append(complex_level)
        else:
            amp = to_numpy(amp_level)
            recombined.append(amp)
    return recombined
