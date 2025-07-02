# angstrom/processing/phase.py

import torch
import numpy as np
from angstrom.processing.filters import butter_bandpass_filter



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
            complex_level = []
            for amp, phase in zip(amp_level, phase_level):
                # Validate inputs
                if torch.isnan(amp).any() or torch.isinf(amp).any():
                    raise ValueError("Amplitude coefficients contain NaN or infinite values")
                if torch.isnan(phase).any() or torch.isinf(phase).any():
                    raise ValueError("Phase coefficients contain NaN or infinite values")

                # Ensure phase is within reasonable bounds
                phase = torch.clamp(phase, -np.pi, np.pi)

                complex_coeff = amp * torch.exp(1j * phase)
                complex_level.append(complex_coeff)
            recombined.append(complex_level)
        else:
            recombined.append(amp_level)  # lowpass/highpass unchanged

    return recombined




def amplify_phase_bandpass_motion(phase_coeffs_list, amplification_factor=10, low_freq=0.1, high_freq=2.0, fps=30.0, filter_func=None):
    """
    Amplify motion by applying a temporal FFT-based amplification to the phase differences (motion phase), amplifying the bandpassed component, and reconstructing the new phase for each frame.

    Args:
        phase_coeffs_list (list): List of phase coefficients for each frame
        amplification_factor (float): Factor to amplify the bandpassed phase
        low_freq (float): Lower frequency bound (Hz)
        high_freq (float): Upper frequency bound (Hz)
        fps (float): Frames per second
        filter_func (callable, optional): Unused, kept for compatibility
    Returns:
        list: Amplified phase coefficients for each frame
    """
    num_frames = len(phase_coeffs_list)
    if num_frames == 0:
        return []

    # Get structure from first frame
    first_frame = phase_coeffs_list[0]
    amplified_phase_coeffs_list = []

    # For each level and band
    for level_idx, level in enumerate(first_frame):
        if isinstance(level, list):
            n_bands = len(level)
            amplified_bands = []
            for band_idx in range(n_bands):
                # Extract temporal sequence for this band: [T, H, W]
                temporal_sequence = []
                for frame_coeffs in phase_coeffs_list:
                    band_data = frame_coeffs[level_idx][band_idx]
                    if isinstance(band_data, torch.Tensor):
                        temporal_sequence.append(band_data.cpu().numpy())
                    else:
                        temporal_sequence.append(np.zeros_like(band_data))
                phase_band = np.stack(temporal_sequence, axis=0)
                # Use FFT-based amplification
                amplified_phase_band = amplify_temporal_fft_band(
                    phase_band, amplification_factor, (low_freq, high_freq), fps
                )
                # Split back into frames
                for t in range(num_frames):
                    if len(amplified_bands) <= t:
                        amplified_bands.append([])
                    amplified_bands[t].append(torch.from_numpy(amplified_phase_band[t]))
            for t in range(num_frames):
                if len(amplified_phase_coeffs_list) <= t:
                    amplified_phase_coeffs_list.append([])
                amplified_phase_coeffs_list[t].append(amplified_bands[t])
        else:
            for t in range(num_frames):
                if len(amplified_phase_coeffs_list) <= t:
                    amplified_phase_coeffs_list.append([])
                amplified_phase_coeffs_list[t].append(level)
    return amplified_phase_coeffs_list
