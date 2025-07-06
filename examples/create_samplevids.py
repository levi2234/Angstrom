import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

class SyntheticVideoGenerator:
    """Generate synthetic videos with controlled motion for testing motion amplification."""

    def __init__(self, width=256, height=256, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

    def generate_pulse_video(self, duration=5, pulse_frequency=1.0, pulse_amplitude=0.02):
        """
        Generate a video with subtle pulsing motion (like heartbeat).

        Args:
            duration (float): Duration in seconds
            pulse_frequency (float): Pulse frequency in Hz (1.0 = 60 BPM)
            pulse_amplitude (float): Amplitude of pulsing (0.02 = 2% size change)

        Returns:
            torch.Tensor: Video tensor of shape [T, 1, H, W]
        """
        num_frames = int(duration * self.fps)
        frames = []

        # Create a simple scene with geometric shapes
        base_image = self._create_base_scene()

        for t in range(num_frames):
            time_sec = t / self.fps

            # Calculate pulse scaling factor
            pulse_phase = 2 * np.pi * pulse_frequency * time_sec
            scale_factor = 1.0 + pulse_amplitude * np.sin(pulse_phase)

            # Apply pulsing transformation
            frame = self._apply_pulse_transform(base_image, scale_factor)
            frames.append(frame)

        return torch.stack(frames).unsqueeze(1)  # [T, 1, H, W]

    def generate_breathing_video(self, duration=10, breathing_frequency=0.25, breathing_amplitude=0.03):
        """
        Generate a video with breathing-like motion.

        Args:
            duration (float): Duration in seconds
            breathing_frequency (float): Breathing frequency in Hz (0.25 = 15 BPM)
            breathing_amplitude (float): Amplitude of breathing motion

        Returns:
            torch.Tensor: Video tensor of shape [T, 1, H, W]
        """
        num_frames = int(duration * self.fps)
        frames = []

        # Create a chest-like scene
        base_image = self._create_chest_scene()

        for t in range(num_frames):
            time_sec = t / self.fps

            # Calculate breathing displacement
            breathing_phase = 2 * np.pi * breathing_frequency * time_sec
            displacement = breathing_amplitude * np.sin(breathing_phase)

            # Apply breathing transformation
            frame = self._apply_breathing_transform(base_image, displacement)
            frames.append(frame)

        return torch.stack(frames).unsqueeze(1)  # [T, 1, H, W]

    def generate_vibration_video(self, duration=3, vibration_frequency=5.0, vibration_amplitude=0.5):
        """
        Generate a video with small vibrations (easier to see when amplified).

        Args:
            duration (float): Duration in seconds
            vibration_frequency (float): Vibration frequency in Hz
            vibration_amplitude (float): Amplitude in pixels

        Returns:
            torch.Tensor: Video tensor of shape [T, 1, H, W]
        """
        num_frames = int(duration * self.fps)
        frames = []

        # Create a scene with objects that can vibrate
        base_image = self._create_vibration_scene()

        for t in range(num_frames):
            time_sec = t / self.fps

            # Calculate vibration displacement
            vibration_phase = 2 * np.pi * vibration_frequency * time_sec
            x_displacement = vibration_amplitude * np.sin(vibration_phase)
            y_displacement = vibration_amplitude * np.cos(vibration_phase * 1.3)  # Slightly different frequency

            # Apply vibration transformation
            frame = self._apply_vibration_transform(base_image, x_displacement, y_displacement)
            frames.append(frame)

        return torch.stack(frames).unsqueeze(1)  # [T, 1, H, W]

    def _create_base_scene(self):
        """Create a base scene with geometric shapes."""
        image = np.zeros((self.height, self.width), dtype=np.float32)

        # Add a circle in the center
        center = (self.width // 2, self.height // 2)
        radius = min(self.width, self.height) // 6
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = 0.8

        # Add some texture/details
        # Grid pattern
        image[::20, :] += 0.1
        image[:, ::20] += 0.1

        # Add some random noise for texture
        noise = np.random.normal(0, 0.02, image.shape)
        image += noise

        return torch.from_numpy(np.clip(image, 0, 1))

    def _create_chest_scene(self):
        """Create a chest-like scene for breathing simulation."""
        image = np.zeros((self.height, self.width), dtype=np.float32)

        # Create an elliptical "chest" shape
        center_x, center_y = self.width // 2, self.height // 2
        a, b = self.width // 3, self.height // 4  # Semi-axes

        y, x = np.ogrid[:self.height, :self.width]
        chest_mask = ((x - center_x)**2 / a**2 + (y - center_y)**2 / b**2) <= 1
        image[chest_mask] = 0.6

        # Add ribs (horizontal lines)
        for i in range(center_y - b//2, center_y + b//2, 15):
            if 0 <= i < self.height:
                image[i, center_x - a//2:center_x + a//2] = 0.9

        # Add some texture
        noise = np.random.normal(0, 0.03, image.shape)
        image += noise

        return torch.from_numpy(np.clip(image, 0, 1))

    def _create_vibration_scene(self):
        """Create a scene with objects that can vibrate."""
        image = np.zeros((self.height, self.width), dtype=np.float32)

        # Add multiple objects
        objects = [
            (self.width // 4, self.height // 4, 20),      # (x, y, size)
            (3 * self.width // 4, self.height // 4, 25),
            (self.width // 2, 3 * self.height // 4, 30),
        ]

        for x, y, size in objects:
            # Create square objects
            x1, y1 = max(0, x - size), max(0, y - size)
            x2, y2 = min(self.width, x + size), min(self.height, y + size)
            image[y1:y2, x1:x2] = 0.7

        # Add background pattern
        image[::10, :] += 0.2
        image[:, ::10] += 0.2

        return torch.from_numpy(np.clip(image, 0, 1))

    def _apply_pulse_transform(self, image, scale_factor):
        """Apply pulsing transformation to image."""
        # Convert to numpy for OpenCV operations
        img_np = image.numpy()

        # Calculate scaling transformation
        center = (self.width // 2, self.height // 2)
        M = cv2.getRotationMatrix2D(center, 0, scale_factor)

        # Apply transformation
        transformed = cv2.warpAffine(img_np, M, (self.width, self.height))

        return torch.from_numpy(transformed)

    def _apply_breathing_transform(self, image, displacement):
        """Apply breathing transformation (vertical stretching)."""
        img_np = image.numpy()

        # Create transformation matrix for vertical scaling
        scale_y = 1.0 + displacement
        center = (self.width // 2, self.height // 2)
        M = np.array([[1, 0, 0], [0, scale_y, center[1] * (1 - scale_y)]], dtype=np.float32)

        # Apply transformation
        transformed = cv2.warpAffine(img_np, M, (self.width, self.height))

        return torch.from_numpy(transformed)

    def _apply_vibration_transform(self, image, x_displacement, y_displacement):
        """Apply vibration transformation (translation)."""
        img_np = image.numpy()

        # Create translation matrix
        M = np.array([[1, 0, x_displacement], [0, 1, y_displacement]], dtype=np.float32)

        # Apply transformation
        transformed = cv2.warpAffine(img_np, M, (self.width, self.height))

        return torch.from_numpy(transformed)

    def save_video_frames(self, video_tensor, output_path="synthetic_video.mp4"):
        """Save video tensor as MP4 file."""
        import cv2

        # Get video properties
        num_frames, _, height, width = video_tensor.shape

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height), isColor=False)

        for i in range(num_frames):
            # Convert frame to uint8 format expected by OpenCV
            frame_np = video_tensor[i].squeeze().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)

            # Write frame
            out.write(frame_np)

        # Release everything
        out.release()
        print(f"Video saved to: {output_path}")

    def create_comparison_video(self, original_video, amplified_video_path, output_path="comparison.mp4"):
        """Create side-by-side comparison video."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        import cv2

        # Load amplified video from file
        cap = cv2.VideoCapture(amplified_video_path)
        amplified_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to grayscale and normalize to [0,1]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized_frame = gray_frame.astype(np.float32) / 255.0
            amplified_frames.append(normalized_frame)

        cap.release()

        # Ensure both videos have the same length
        min_frames = min(len(original_video), len(amplified_frames))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        def animate(frame):
            ax1.clear()
            ax2.clear()

            ax1.imshow(original_video[frame].squeeze().numpy(), cmap='gray')
            ax1.set_title('Original')
            ax1.axis('off')

            ax2.imshow(amplified_frames[frame], cmap='gray')
            ax2.set_title('Amplified')
            ax2.axis('off')

            return []

        anim = FuncAnimation(fig, animate, frames=min_frames,
                           interval=1000/self.fps, blit=False)

        # Save as MP4
        # writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='Motion Amplification'), bitrate=1800)

        # Use PillowWriter instead (saves as GIF, no external dependencies)
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=self.fps)

        # Change output path to .gif extension
        output_path = output_path.replace('.mp4', '.gif')
        anim.save(output_path, writer=writer)
        plt.close()

        print(f"Comparison video saved to: {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Create generator
    generator = SyntheticVideoGenerator(width=128, height=128, fps=30)

    # Generate different types of test videos
    print("Generating pulse video...")
    pulse_video = generator.generate_pulse_video(duration=3, pulse_frequency=1.0, pulse_amplitude=0.01)
    print(f"Pulse video shape: {pulse_video.shape}")

    print("Generating breathing video...")
    breathing_video = generator.generate_breathing_video(duration=5, breathing_frequency=0.3, breathing_amplitude=0.02)
    print(f"Breathing video shape: {breathing_video.shape}")

    print("Generating vibration video...")
    vibration_video = generator.generate_vibration_video(duration=2, vibration_frequency=3.0, vibration_amplitude=0.3)
    print(f"Vibration video shape: {vibration_video.shape}")

    # Save some sample frames
    # Save as MP4
    # generator.save_video_frames(pulse_video, "pulse_test.mp4")
    # generator.save_video_frames(breathing_video, "breathing_test.mp4")
    # generator.save_video_frames(vibration_video, "vibration_test.mp4")


    generator.create_comparison_video(breathing_video, "C:/Users/levi2/Desktop/Projects/Angstrom/amplified_output.mp4", "comparison.gif")

    print("Test videos generated successfully!")
    print("Use these videos to test your motion amplification pipeline:")
    print("1. Pulse video: Should show subtle pulsing that can be amplified")
    print("2. Breathing video: Should show breathing-like motion")
    print("3. Vibration video: Should show small vibrations that become visible when amplified")