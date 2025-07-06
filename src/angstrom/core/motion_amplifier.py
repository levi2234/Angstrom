import os
import cv2
import numpy as np
from angstrom.io.video_io import read_video_frames, write_video_frames
from angstrom.processing.pyramid import ComplexSteerablePyramid
from angstrom.processing.phase import extract_phase,  extract_amplitude, reconstruct_from_amplitude_and_phase
from angstrom.processing.filters import butter_bandpass_filter, temporal_ideal_filter
import torch
from tqdm import tqdm


class MotionAmplifier:
    def __init__(self, device=None):
        """Initialize the MotionAmplifier for video processing.

        Args:
            device (torch.device, optional): Device to use for computation.
                Defaults to CUDA if available, otherwise CPU.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.video = None
        self.video_fps = None
        self.video_shape = None
        self.pyramid = ComplexSteerablePyramid()
        self.pyramid_coeffs = None
        self.temporal_filtered_coeffs = None

    def load_video(self, input_path):
        """Load video frames as a PyTorch tensor with video properties.

        Args:
            input_path (str): Path to the input video file.
        """
        # Get video properties
        cap = cv2.VideoCapture(input_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Load video frames
        self.video = read_video_frames(input_path).to(self.device)
        self.video_shape = self.video.shape  # [N, C, H, W]

        print(f"Loaded video: {self.video_shape[0]} frames, {self.video_fps} FPS, "
              f"Resolution: {self.video_shape[2]}x{self.video_shape[3]}")

    def process(self, input_path=None):
        """Process a video by decomposing all frames into pyramid coefficients.

        Args:
            input_path (str, optional): Path to input video. If provided, loads the video first.

        Returns:
            list: List of pyramid coefficients for each frame.
        """
        if input_path:
            self.load_video(input_path)

        if self.video is None:
            raise ValueError("No video loaded. Please load a video using 'load_video()' before processing.")

        print("Decomposing video frames into pyramid coefficients...")
        pyramid_coeffs = []

        # Process each frame
        video_shape = self.video.shape
        for i in tqdm(range(video_shape[0]), desc="Decomposing frames"):
            frame = self.video[i:i+1]  # [1, C, H, W]
            coeffs = self.pyramid.decompose(frame)
            pyramid_coeffs.append(coeffs)

        self.pyramid_coeffs = pyramid_coeffs


        return pyramid_coeffs


    def amplify(self, amplification_factor=10, frequency_range=None):
        """Amplify motion in the video using proper motion phase extraction and amplification.

        This method implements the corrected approach:
        1. Extract phase coefficients for all frames
        2. Calculate motion phase (temporal differences)
        3. Apply temporal filtering to motion phase
        4. Amplify the filtered motion
        5. Add amplified motion to base phase
        6. Reconstruct frames

        Args:
            amplification_factor (float): Factor by which to amplify the motion
            frequency_range (tuple, optional): (lowcut, highcut) frequency range in Hz to amplify.
                If None, amplifies all frequencies.

        Returns:
            torch.Tensor: Amplified video tensor of shape [N, C, H, W]
        """
        if self.pyramid_coeffs is None:
            raise ValueError("No processing has been performed yet on the video")

        if self.video is None:
            raise ValueError("No video loaded. Cannot amplify.")

        print("Amplifying motion using proper motion phase extraction...")

        # Extract phase coefficients for all frames
        phase_coeffs_list = []
        amplitude_coeffs_list = []

        for frame_coeffs in self.pyramid_coeffs:
            phase_coeffs = extract_phase(frame_coeffs)
            amplitude_coeffs = extract_amplitude(frame_coeffs)
            phase_coeffs_list.append(phase_coeffs)
            amplitude_coeffs_list.append(amplitude_coeffs)

        # Set frequency range and fps
        fps = float(self.video_fps) if self.video_fps is not None else 30.0
        if frequency_range is not None:
            low_freq, high_freq = frequency_range
        else:
            # Default to amplifying motion frequencies (0.1-2.0 Hz typical for human motion)
            low_freq, high_freq = 0.1, 2.0

        # Apply corrected motion amplification
        amplified_phase_coeffs_list = self._amplify_motion_phase(
            phase_coeffs_list,
            amplification_factor=amplification_factor,
            low_freq=low_freq,
            high_freq=high_freq,
            fps=fps
        )

        # Reconstruct frames using amplified phases
        print("Reconstructing amplified frames...")
        reconstructed_frames = []
        target_size = (self.video.shape[2], self.video.shape[3])  # (height, width)

        for i, (amplitude_coeffs, phase_coeffs) in tqdm(enumerate(zip(amplitude_coeffs_list, amplified_phase_coeffs_list)), desc="Reconstructing frames"):
            # Reconstruct from amplitude and amplified phase
            recombined = reconstruct_from_amplitude_and_phase(amplitude_coeffs, phase_coeffs)

            # Reconstruct frame using explicit target size
            reconstructed = self.pyramid.reconstruct_with_size(recombined, target_size)

            # Ensure proper shape [C, H, W]
            if reconstructed.dim() == 2:
                reconstructed = reconstructed.unsqueeze(0)

            reconstructed_frames.append(reconstructed)

        # Stack all frames into video tensor [N, C, H, W]
        amplified_video = torch.stack(reconstructed_frames, dim=0)
        return amplified_video

    def _amplify_motion_phase(self, phase_coeffs_list, amplification_factor=10, low_freq=0.1, high_freq=2.0, fps=30.0):
        """Internal method to properly amplify motion phase.

        This method implements the correct phase-based motion amplification according to theory:
        1. Extract phase differences: Δϕ(t) = ϕ(t) - ϕ(0)
        2. Apply temporal bandpass filter to phase differences: bandpass(Δϕ(t))
        3. Amplify the filtered phase deviations: α * bandpass(Δϕ(t))
        4. Add amplified phase deviations to base phase: ϕ̃(t) = ϕ(0) + α * bandpass(Δϕ(t))

        Args:
            phase_coeffs_list (list): List of phase coefficients for each frame
            amplification_factor (float): Amplification factor α
            low_freq (float): Lower frequency bound (Hz)
            high_freq (float): Upper frequency bound (Hz)
            fps (float): Frames per second

        Returns:
            list: Amplified phase coefficients for each frame
        """
        num_frames = len(phase_coeffs_list)
        if num_frames < 2:
            return phase_coeffs_list

        # Check if pyramid coefficients are available
        if self.pyramid_coeffs is None:
            raise ValueError("No pyramid coefficients available. Run process() first.")

        # Get structure from first frame
        first_frame = phase_coeffs_list[0]
        amplified_phase_coeffs_list = []

        # For each level and band
        for level_idx, level in enumerate(first_frame):
            if isinstance(level, list):
                # For each band
                n_bands = len(level)
                amplified_bands = []

                for band_idx in range(n_bands):
                    # Extract temporal sequence of PHASE coefficients: [T, H, W]
                    phase_temporal_sequence = []
                    for frame_coeffs in phase_coeffs_list:
                        band_data = frame_coeffs[level_idx][band_idx]
                        if isinstance(band_data, torch.Tensor):
                            phase_temporal_sequence.append(band_data.cpu().numpy())
                        elif isinstance(band_data, np.ndarray):
                            phase_temporal_sequence.append(band_data)
                        else:
                            # Handle scalar or other types
                            if hasattr(band_data, 'shape'):
                                phase_temporal_sequence.append(np.zeros_like(band_data))
                            else:
                                phase_temporal_sequence.append(np.array(0.0))

                    # Stack into [T, H, W] array of phase coefficients
                    phase_band = np.stack(phase_temporal_sequence, axis=0)

                    # Step 1: Calculate phase differences from base phase (frame 0)
                    # Δϕ(t) = ϕ(t) - ϕ(0)
                    base_phase = phase_band[0]  # ϕ(0)
                    phase_differences = np.zeros_like(phase_band)
                    for t in range(num_frames):
                        phase_diff = phase_band[t] - base_phase
                        # Handle phase wrapping (ensure differences are in [-π, π])
                        phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
                        phase_differences[t] = phase_diff

                    # Step 2: Apply temporal bandpass filter to phase differences
                    # bandpass(Δϕ(t))
                    filtered_phase_differences = temporal_ideal_filter(phase_differences, low_freq, high_freq, fps)

                    # Step 3: Amplify the filtered phase deviations
                    # α * bandpass(Δϕ(t))
                    amplified_phase_deviations = filtered_phase_differences * amplification_factor

                    # Step 4: Add amplified phase deviations to base phase
                    # ϕ̃(t) = ϕ(0) + α * bandpass(Δϕ(t))
                    amplified_phase = np.zeros_like(phase_band)
                    for t in range(num_frames):
                        amplified_phase[t] = base_phase + amplified_phase_deviations[t]
                        # Ensure phase stays within reasonable bounds
                        amplified_phase[t] = np.mod(amplified_phase[t] + np.pi, 2*np.pi) - np.pi

                    # Split back into frames
                    for t in range(num_frames):
                        if len(amplified_bands) <= t:
                            amplified_bands.append([])
                        phase_frame = amplified_phase[t]
                        if isinstance(phase_frame, np.ndarray):
                            amplified_bands[t].append(torch.from_numpy(phase_frame))
                        else:
                            amplified_bands[t].append(torch.tensor(phase_frame))

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

    def save_video(self, video_tensor, output_path):
        """Save the processed video tensor to a file.

        Args:
            video_tensor (torch.Tensor): Video tensor of shape [N, C, H, W]
            output_path (str): Path to save the output video
        """
        if self.video_fps is None:
            raise ValueError("Video FPS not available. Cannot save video.")

        if video_tensor is None:
            raise ValueError("Video tensor is None. Cannot save video.")

        # Convert tensor to numpy and prepare for saving
        video_np = video_tensor.cpu().numpy()

        # Convert from [N, C, H, W] to list of [H, W, C] frames
        frames = []
        for i in range(video_np.shape[0]):
            frame = video_np[i]  # [C, H, W]
            if frame is None:
                continue
            if frame.shape[0] == 1:  # Grayscale
                frame = frame[0]  # [H, W]
                frame = (frame * 255).astype(np.uint8)
            else:  # RGB
                frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
                frame = (frame * 255).astype(np.uint8)
            frames.append(frame)

        # Save video
        write_video_frames(frames, output_path, self.video_fps)
        print(f"Video saved to: {output_path}")
