# angstrom/processing/phase.py

import torch
import numpy as np

def extract_phase(coeffs):
    """
    Extract the phase (angle) from complex steerable pyramid coefficients.
    """
    phase_coeffs = []
    for level in coeffs:
        if isinstance(level, list):  # Bandpass filters: list of complex tensors
            phase_level = []
            for band in level:
                if isinstance(band, torch.Tensor):
                    phase_level.append(torch.angle(band))
                elif isinstance(band, np.ndarray):
                    phase_level.append(np.angle(band))
                else:
                    phase_level.append(band)
            phase_coeffs.append(phase_level)
        else:
            if isinstance(level, torch.Tensor):
                phase_coeffs.append(torch.angle(level))
            elif isinstance(level, np.ndarray):
                phase_coeffs.append(np.angle(level))
            else:
                phase_coeffs.append(level)
    return phase_coeffs



def extract_amplitude(coeffs):
    """
    Extract the amplitude (magnitude) from complex steerable pyramid coefficients.
    """
    amplitude_coeffs = []
    for level in coeffs:
        if isinstance(level, list):
            amp_level = []
            for band in level:
                if isinstance(band, torch.Tensor):
                    amp_level.append(torch.abs(band))
                elif isinstance(band, np.ndarray):
                    amp_level.append(np.abs(band))
                else:
                    amp_level.append(band)
            amplitude_coeffs.append(amp_level)
        else:
            if isinstance(level, torch.Tensor):
                amplitude_coeffs.append(torch.abs(level))
            elif isinstance(level, np.ndarray):
                amplitude_coeffs.append(np.abs(level))
            else:
                amplitude_coeffs.append(level)
    return amplitude_coeffs


def reconstruct_from_amplitude_and_phase(amplitude_coeffs, phase_coeffs):
    """
    Combine amplitude and phase back into complex steerable pyramid coefficients.
    """
    recombined = []

    for amp_level, phase_level in zip(amplitude_coeffs, phase_coeffs):
        if isinstance(amp_level, list):
            complex_level = []
            for amp, phase in zip(amp_level, phase_level):
                # Validate inputs
                if isinstance(amp, torch.Tensor):
                    if torch.isnan(amp).any() or torch.isinf(amp).any():
                        raise ValueError("Amplitude coefficients contain NaN or infinite values")
                elif isinstance(amp, np.ndarray):
                    if np.isnan(amp).any() or np.isinf(amp).any():
                        raise ValueError("Amplitude coefficients contain NaN or infinite values")

                if isinstance(phase, torch.Tensor):
                    if torch.isnan(phase).any() or torch.isinf(phase).any():
                        raise ValueError("Phase coefficients contain NaN or infinite values")
                elif isinstance(phase, np.ndarray):
                    if np.isnan(phase).any() or np.isinf(phase).any():
                        raise ValueError("Phase coefficients contain NaN or infinite values")

                # Ensure phase is within reasonable bounds and types match
                if isinstance(amp, torch.Tensor) or isinstance(phase, torch.Tensor):
                    # Convert both to torch.Tensor
                    if not isinstance(amp, torch.Tensor):
                        amp = torch.from_numpy(amp)
                    if not isinstance(phase, torch.Tensor):
                        phase = torch.from_numpy(phase)
                    phase = torch.clamp(phase, -np.pi, np.pi)
                    complex_coeff = amp * torch.exp(1j * phase)
                else:
                    # Both are numpy
                    phase = np.clip(phase, -np.pi, np.pi)
                    complex_coeff = amp * np.exp(1j * phase)

                complex_level.append(complex_coeff)
            recombined.append(complex_level)
        else:
            # lowpass/highpass unchanged
            recombined.append(amp_level)

    return recombined
