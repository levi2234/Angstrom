import torch
import numpy as np
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch

class ComplexSteerablePyramid:
    def __init__(self, height=5, nbands=4, scale_factor=2):
        """
        Initializes the ComplexSteerablePyramid.

        Args:
            height (int): Number of levels in the pyramid.
            nbands (int): Number of orientation bands.
            scale_factor (float): Scaling factor between levels.
            device (torch.device or str): Device to use ('cuda' or 'cpu').
        """

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the pyramid
        try:
            self.pyr = SCFpyr_PyTorch(height=height, nbands=nbands, scale_factor=scale_factor, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SCFpyr_PyTorch: {e}")

    def decompose(self, image_batch):
        """
        Decomposes an image tensor into its complex steerable pyramid representation.

        Args:
            image_batch (torch.Tensor): A 4D tensor of shape [N,C,H, W]

        Returns:
            list: Pyramid coefficients.
        """

        # Ensure the tensor is on the same device as the pyramid
        image_tensor = image_batch.to(self.device)
        print(np.shape(image_tensor), type(image_tensor))
        coeffs = self.pyr.build(image_tensor)
        return coeffs

    def reconstruct(self, coeffs):
        """
        Reconstructs an image from its pyramid coefficients.

        Args:
            coeffs (list): Pyramid coefficients.

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [H, W].
        """
        if coeffs is None:
            raise ValueError("Coefficients cannot be None for reconstruction.")

        recon = self.pyr.reconstruct(coeffs)
        if recon is None:
            raise RuntimeError("Reconstruction failed and returned None.")
        else:
            recon = recon.squeeze(0).squeeze(0)
        return recon

    def updated(self):
        pass