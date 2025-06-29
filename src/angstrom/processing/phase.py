# angstrom/processing/phase.py

import torch
import numpy as np
from angstrom.processing.temporal_filter import butter_bandpass_filter

def unwrap_phase(phase_tensor):
    """
    Unwrap phase values to handle phase wrapping issues.

    Args:
        phase_tensor (torch.Tensor): Phase tensor with values in [-π, π]

    Returns:
        torch.Tensor: Unwrapped phase tensor
    """
    # Convert to numpy for unwrapping
    phase_np = phase_tensor.cpu().numpy()

    # Unwrap along each dimension
    unwrapped = np.unwrap(phase_np, axis=0)  # Unwrap along time dimension
    if unwrapped.ndim > 1:
        unwrapped = np.unwrap(unwrapped, axis=1)  # Unwrap along height dimension
    if unwrapped.ndim > 2:
        unwrapped = np.unwrap(unwrapped, axis=2)  # Unwrap along width dimension

    return torch.from_numpy(unwrapped).to(phase_tensor.device)


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

    WARNING: This function is deprecated. Use amplify_motion_phase() instead
    for proper motion amplification, as direct phase multiplication is incorrect.

    Args:
        phase_coeffs (list): List of phase tensors.
        factor (float): Amplification factor.

    Returns:
        list: Amplified phase tensors.
    """
    import warnings
    warnings.warn(
        "amplify_phase() is deprecated. Use amplify_motion_phase() for proper motion amplification.",
        DeprecationWarning,
        stacklevel=2
    )

    amplified = []
    for level in phase_coeffs:
        if isinstance(level, list):
            amplified_level = [p * factor for p in level]
            amplified.append(amplified_level)
        else:
            amplified.append(level)
    return amplified


def amplify_motion_phase(base_phase, motion_phase, factor):
    """
    Amplify motion by amplifying phase differences between frames.

    This is the correct approach for motion amplification:
    1. Take a base phase (reference frame)
    2. Take motion phase (difference from base)
    3. Amplify the motion phase by the factor
    4. Add the amplified motion to the base phase

    Args:
        base_phase (list): Phase coefficients of the base/reference frame
        motion_phase (list): Phase coefficients representing motion (difference from base)
        factor (float): Amplification factor for the motion

    Returns:
        list: Amplified phase coefficients
    """
    amplified = []

    for base_level, motion_level in zip(base_phase, motion_phase):
        if isinstance(base_level, list):
            amplified_level = []
            for base_band, motion_band in zip(base_level, motion_level):
                if isinstance(base_band, torch.Tensor) and isinstance(motion_band, torch.Tensor):
                    # Amplify the motion phase and add to base phase
                    amplified_band = base_band + (motion_band * factor)
                    amplified_level.append(amplified_band)
                else:
                    amplified_level.append(base_band)
            amplified.append(amplified_level)
        else:
            # For lowpass/highpass, just use the base
            amplified.append(base_level)

    return amplified


def amplify_phase_temporal_fft(phase_coeffs_list, amplification_factor=10, frequency_range=None, fps=30.0):
    """
    Amplify motion using temporal FFT of phase coefficients.

    This is the CORRECT approach for phase-based motion amplification:
    1. Take temporal FFT of phase coefficients across all frames
    2. Amplify specific frequency bands in the temporal domain
    3. Apply inverse FFT to get amplified motion

    Args:
        phase_coeffs_list (list): List of phase coefficients for each frame
        amplification_factor (float): Factor to amplify motion frequencies
        frequency_range (tuple): (low_freq, high_freq) in Hz to amplify
        fps (float): Frames per second of the video

    Returns:
        list: Amplified phase coefficients for each frame
    """
    if not phase_coeffs_list:
        return phase_coeffs_list

    num_frames = len(phase_coeffs_list)
    amplified_coeffs = []

    # Get the structure from the first frame
    first_frame = phase_coeffs_list[0]

    # Process each level and band
    for level_idx, level in enumerate(first_frame):
        if isinstance(level, list):  # Bandpass filters
            amplified_level = []
            for band_idx, _ in enumerate(level):
                # Extract temporal sequence for this band: [T, H, W]
                temporal_sequence = []
                for frame_coeffs in phase_coeffs_list:
                    band_data = frame_coeffs[level_idx][band_idx]
                    if isinstance(band_data, torch.Tensor):
                        temporal_sequence.append(band_data.cpu().numpy())
                    else:
                        temporal_sequence.append(np.zeros_like(band_data))

                # Stack into [T, H, W] array
                temporal_array = np.stack(temporal_sequence, axis=0)

                # Apply temporal FFT amplification
                amplified_temporal = amplify_temporal_fft_band(
                    temporal_array, amplification_factor, frequency_range, fps
                )

                # Convert back to tensor
                amplified_tensor = torch.from_numpy(amplified_temporal)
                amplified_level.append(amplified_tensor)

            amplified_coeffs.append(amplified_level)
        else:
            # For lowpass/highpass, just use the first frame's coefficients
            amplified_coeffs.append(level)

    return amplified_coeffs


def amplify_temporal_fft_band(temporal_array, amplification_factor, frequency_range, fps):
    """
    Amplify a specific frequency band in the temporal domain using FFT.

    Args:
        temporal_array (np.ndarray): Shape [T, H, W] - temporal sequence
        amplification_factor (float): Factor to amplify motion frequencies
        frequency_range (tuple): (low_freq, high_freq) in Hz to amplify
        fps (float): Frames per second

    Returns:
        np.ndarray: Amplified temporal array
    """
    # Input validation
    if temporal_array is None or temporal_array.size == 0:
        raise ValueError("Temporal array is empty or None")

    if temporal_array.shape[0] < 2:
        raise ValueError("Temporal array must have at least 2 frames")

    if amplification_factor <= 0:
        raise ValueError("Amplification factor must be positive")

    if fps <= 0:
        raise ValueError("FPS must be positive")

    if frequency_range is not None:
        low_freq, high_freq = frequency_range
        if low_freq < 0 or high_freq < 0 or low_freq >= high_freq:
            raise ValueError("Invalid frequency range: low_freq < high_freq and both >= 0")

        # Check if frequency range is within Nyquist limit
        nyquist_freq = fps / 2
        if high_freq > nyquist_freq:
            raise ValueError(f"High frequency {high_freq} Hz exceeds Nyquist frequency {nyquist_freq} Hz")

    T, H, W = temporal_array.shape

    # Apply FFT along temporal axis
    fft_temporal = np.fft.fft(temporal_array, axis=0)

    # Calculate frequency bins
    freqs = np.fft.fftfreq(T, 1/fps)

    # Create amplification mask
    amplification_mask = np.ones(T)

    if frequency_range is not None:
        low_freq, high_freq = frequency_range
        # Find frequency bins within the range
        freq_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        # Apply amplification to those frequencies
        amplification_mask[freq_mask] = amplification_factor
    else:
        # Amplify all non-DC frequencies
        amplification_mask[1:] = amplification_factor

    # Apply amplification mask
    # Reshape mask to broadcast with fft_temporal
    amplification_mask = amplification_mask.reshape(T, 1, 1)
    amplified_fft = fft_temporal * amplification_mask

    # Apply inverse FFT
    amplified_temporal = np.real(np.fft.ifft(amplified_fft, axis=0))

    return amplified_temporal


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


def amplify_phase_bandpass(phase_coeffs_list, amplification_factor=10, low_freq=0.4, high_freq=0.6, fps=30.0, filter_order=2):
    """
    Amplify motion by applying a temporal bandpass filter to the phase, amplifying the bandpassed component, and reconstructing the new phase for each frame.

    CORRECTED APPROACH:
    1. Extract motion phase (temporal differences)
    2. Apply bandpass filter to motion phase
    3. Amplify the filtered motion
    4. Add amplified motion to base phase

    Args:
        phase_coeffs_list (list): List of phase coefficients for each frame
        amplification_factor (float): Factor to amplify the bandpassed phase
        low_freq (float): Lower frequency bound (Hz)
        high_freq (float): Upper frequency bound (Hz)
        fps (float): Frames per second
        filter_order (int): Order of the Butterworth filter
    Returns:
        list: Amplified phase coefficients for each frame
    """
    num_frames = len(phase_coeffs_list)
    if num_frames == 0:
        return []

    # Get structure from first frame
    first_frame = phase_coeffs_list[0]

    # Prepare output: list of [frames][level][band][H][W]
    amplified_phase_coeffs_list = []

    # For each level and band
    for level_idx, level in enumerate(first_frame):
        if isinstance(level, list):
            # For each band
            band_shape = level[0].shape

            # Stack phase over time: [frames, bands, H, W]
            phase_stack = []
            for frame in phase_coeffs_list:
                band_stack = []
                for band in frame[level_idx]:
                    band_stack.append(band.cpu().numpy())
                phase_stack.append(band_stack)
            phase_stack = np.array(phase_stack)  # [frames, bands, H, W]
            n_bands = phase_stack.shape[1]

            # For each band, extract motion and amplify
            amplified_bands = []
            for band_idx in range(n_bands):
                # [frames, H, W]
                phase_band = phase_stack[:, band_idx, :, :]

                # Extract motion phase (temporal differences)
                motion_phase = np.zeros_like(phase_band)
                for t in range(1, num_frames):
                    # Calculate phase difference (motion)
                    motion_phase[t] = phase_band[t] - phase_band[t-1]

                # Bandpass filter the motion phase along time axis for each pixel
                filtered_motion = butter_bandpass_filter(motion_phase, low_freq, high_freq, fps, order=filter_order)

                # Amplify the filtered motion
                amplified_motion = filtered_motion * amplification_factor

                # Add amplified motion to original phase
                amplified_phase = phase_band + amplified_motion

                # Split back into frames
                for t in range(num_frames):
                    if len(amplified_bands) <= t:
                        amplified_bands.append([])
                    amplified_bands[t].append(torch.from_numpy(amplified_phase[t]))

            # Add to output
            for t in range(num_frames):
                if len(amplified_phase_coeffs_list) <= t:
                    amplified_phase_coeffs_list.append([])
                amplified_phase_coeffs_list[t].append(amplified_bands[t])
        else:
            # For lowpass/highpass, just copy
            for t in range(num_frames):
                if len(amplified_phase_coeffs_list) <= t:
                    amplified_phase_coeffs_list.append([])
                amplified_phase_coeffs_list[t].append(level)

    return amplified_phase_coeffs_list
