import os
from angstrom.io.video_io import read_video_frames
from angstrom.processing.pyramid import ComplexSteerablePyramid
from angstrom.processing.phase import extract_phase, amplify_phase, extract_amplitude, reconstruct_from_amplitude_and_phase
import torch
from tqdm import tqdm

class MotionAmplifier:
    def __init__(self, device=torch.device):
        self.video = None
        self.pyramid_levels = 4
        self.pyramid = ComplexSteerablePyramid()
        self.pyramid_coeffs = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def load_video(self, input_path):
        # Load video frames as a PyTorch tensor
        self.video = read_video_frames(input_path).to(self.device)

    def process(self, input_path=None, amplification_factor=10):
        """
        Processes a video by loading it, decomposing its frames into pyramid coefficients,
        and optionally applying an amplification factor.
        Args:
            input_path (str, optional): The file path to the input video. If provided, the video
                will be loaded before processing. Defaults to None.
            amplification_factor (int, optional): The factor by which to amplify the motion
                in the video. Defaults to 10.
        Raises:
            ValueError: If no video is loaded prior to processing.
        Returns:
            list: A list of pyramid coefficients obtained from decomposing the video frames.
        """
        if input_path:
            self.load_video(input_path)

        # Check if self.video is loaded
        if self.video is None:
            raise ValueError("No video loaded. Please load a video using 'load_video()' before processing.")

        pyramid_coeffs = []

        # Decompose frames and extract phase
        pyramid_coeffs = self.pyramid.decompose(self.video)  # Decompose frame

        self.pyramid_coeffs = pyramid_coeffs

        return pyramid_coeffs


    #Now amplification is done on a single frame an no temporal effects are taken into account.
    # We still need to amplify
    def amplify(self, amplification_factor = 10):
        if self.pyramid_coeffs is None:
            raise ValueError("No processing has been performed yet on the video")

        reconstructed_array = []

        for coeffs in tqdm(self.pyramid_coeffs, desc="Amplifying and video frames"):
            phase = extract_phase(self.pyramid_coeffs)  # Extract phase
            amplitude = extract_amplitude(self.pyramid_coeffs)
            amplified_phase = amplify_phase(phase, amplification_factor)  # Amplify phase
            recombined = reconstruct_from_amplitude_and_phase(amplitude, amplified_phase)
            reconstructed = self.pyramid.reconstruct(recombined)
            reconstructed_array.append(reconstructed)

        return  reconstructed_array
