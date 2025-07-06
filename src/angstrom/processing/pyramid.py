import torch
import numpy as np
from angstrom.pyramids.steerable_pyramid import SuboctaveSP, SteerablePyramid


class ComplexSteerablePyramid:
    def __init__(
            self,
            height=5,
            nbands=4,
            scale_factor=2,
            pyramid_type="pyramidal"):
        """
        Initializes the ComplexSteerablePyramid.

        Args:
            height (int): Number of levels in the pyramid.
            nbands (int): Number of orientation bands.
            scale_factor (float): Scaling factor between levels.
            device (torch.device or str): Device to use ('cuda' or 'cpu').
        """

        # Set the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the pyramid using the custom implementation
        try:

            if pyramid_type == "suboctave":
                self.pyr = SuboctaveSP(
                    depth=height,
                    orientations=nbands,
                    filters_per_octave=1,
                    complex_pyr=True)
            elif pyramid_type == "pyramidal":
                self.pyr = SteerablePyramid(
                    depth=height,
                    orientations=nbands,
                    filters_per_octave=1,
                    complex_pyr=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SuboctaveSP: {e}")

        # Store original image size for reconstruction
        self.original_image_size = None

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
        # print(np.shape(image_tensor), type(image_tensor))

        # Convert to numpy for the custom implementation
        image_np = image_tensor.cpu().numpy()

        # Get the first image (assuming batch size 1 for now)
        image = image_np[0, 0]  # [H, W]

        # Store original image size for reconstruction
        self.original_image_size = image.shape

        # Get filters and crops
        filters, crops = self.pyr.get_filters(image, cropped=True)

        # Build pyramid
        coeffs = self.pyr.build_pyramid(image, filters, crops)

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

        # Use stored original image size for reconstruction
        if self.original_image_size is None:
            # Fallback: use the size of the first coefficient
            h, w = coeffs[0].shape
            print(
                f"Warning: No original image size stored, using coefficient size: {h}x{w}")
        else:
            h, w = self.original_image_size
            print(f"Reconstructing to original size: {h}x{w}")

        # Create dummy image with correct original size
        dummy_image = np.zeros((h, w))
        filters, crops = self.pyr.get_filters(dummy_image, cropped=True)

        recon = self.pyr.reconstruct_image(coeffs, filters, crops)

        if recon is None:
            raise RuntimeError("Reconstruction failed and returned None.")
        else:
            # Convert back to torch tensor
            recon_tensor = torch.from_numpy(recon).to(self.device)
            recon_tensor = recon_tensor.unsqueeze(0).unsqueeze(
                0)  # Add batch and channel dimensions
        return recon_tensor

    def reconstruct_with_size(self, coeffs, target_size):
        """
        Reconstructs an image from its pyramid coefficients with a specific target size.
        This method ensures consistent reconstruction regardless of stored state.

        Args:
            coeffs (list): Pyramid coefficients.
            target_size (tuple): Target size (height, width) for reconstruction.

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [H, W].
        """
        if coeffs is None:
            raise ValueError("Coefficients cannot be None for reconstruction.")

        h, w = target_size
        # print(f"Reconstructing to target size: {h}x{w}")

        # Create dummy image with target size
        dummy_image = np.zeros((h, w))
        filters, crops = self.pyr.get_filters(dummy_image, cropped=True)

        recon = self.pyr.reconstruct_image(coeffs, filters, crops)

        if recon is None:
            raise RuntimeError("Reconstruction failed and returned None.")
        else:
            # Convert back to torch tensor
            recon_tensor = torch.from_numpy(recon).to(self.device)
            recon_tensor = recon_tensor.unsqueeze(0).unsqueeze(
                0)  # Add batch and channel dimensions
        return recon_tensor

    def updated(self):
        pass
