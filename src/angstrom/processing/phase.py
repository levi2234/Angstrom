# angstrom/processing/phase.py

import torch

def extract_phase(coeffs):
    """
    Extract the phase (angle) from complex steerable pyramid coefficients.

    Args:
        coeffs (list): Output from pyramid.decompose().
                       Each item is either a list of bandpass filters (complex tensors)
                       or a single lowpass/highpass tensor.

    Returns:
        list: Same structure as coeffs, but with phase (angles) for complex tensors.
    """
    phase_coeffs = []

    for level in coeffs:
        if isinstance(level, list):  # Bandpass filters: list of complex tensors
            phase_level = [torch.angle(band) for band in level]
            phase_coeffs.append(phase_level)
        else:
            phase_coeffs.append(level)  # lowpass/highpass unchanged or None

    return phase_coeffs


def amplify_phase(phase_coeffs, factor):
    """
    Amplifies the phase of steerable pyramid coefficients.

    Args:
        phase_coeffs (list): List of phase tensors.
        factor (float): Amplification factor.

    Returns:
        list: Amplified phase tensors.
    """
    amplified = []
    for level in phase_coeffs:
        if isinstance(level, list):
            amplified_level = [p * factor for p in level]
            amplified.append(amplified_level)
        else:
            amplified.append(level)
    return amplified


def extract_amplitude(coeffs):
    """
    Extract the amplitude (magnitude) from complex steerable pyramid coefficients.

    Args:
        coeffs (list): Output from pyramid.decompose().

    Returns:
        list: Same structure as coeffs, with magnitude tensors where appropriate.
    """
    amplitude_coeffs = []

    for level in coeffs:
        if isinstance(level, list):
            amp_level = [torch.abs(band) for band in level]
            amplitude_coeffs.append(amp_level)
        else:
            amplitude_coeffs.append(level)

    return amplitude_coeffs


def reconstruct_from_amplitude_and_phase(amplitude_coeffs, phase_coeffs):
    """
    Combine amplitude and phase back into complex steerable pyramid coefficients.

    Args:
        amplitude_coeffs (list): List of magnitude tensors.
        phase_coeffs (list): List of angle tensors.

    Returns:
        list: Recombined complex pyramid coefficients for reconstruction.
    """
    recombined = []

    for amp_level, phase_level in zip(amplitude_coeffs, phase_coeffs):
        if isinstance(amp_level, list):
            complex_level = [
                amp * torch.exp(1j * phase)
                for amp, phase in zip(amp_level, phase_level)
            ]
            recombined.append(complex_level)
        else:
            recombined.append(amp_level)  # lowpass/highpass unchanged

    return recombined
