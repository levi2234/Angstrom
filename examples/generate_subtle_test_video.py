import numpy as np
import cv2
import os

def generate_subtle_test_video(output_path="test_subtle_motion.mp4", fps=30):
    """
    Generate a test video with 100 frames of a 50x50 image containing a square
    that has very subtle motion - just a few pixels of movement.

    Args:
        output_path (str): Path to save the output video
        fps (int): Frames per second for the video
    """

    # Video parameters
    width, height = 50, 50
    num_frames = 100

    # Square parameters - much smaller motion
    square_size = 10  # Fixed square size
    square_center_x = width // 2
    square_center_y = height // 2

    # Subtle motion parameters
    FREQ = 0.3  # Frequency of oscillation (Hz, matches FFT bin for 100 frames at 30 FPS)
    motion_amplitude = 2.5  # Maximum movement in pixels (sub-pixel amplitude)
    blur_sigma = 0.001  # Gaussian blur sigma

    frames = []
    for frame_num in range(num_frames):
        # Calculate sub-pixel center position (sinusoidal motion)
        x_offset = motion_amplitude * np.sin(2 * np.pi * FREQ * frame_num)
        y_offset = motion_amplitude * np.cos(2 * np.pi * FREQ * frame_num)
        current_x = square_center_x + x_offset
        current_y = square_center_y + y_offset

        # Create empty image
        img = np.zeros((height, width), dtype=np.float32)

        # Calculate square boundaries
        half_size = square_size // 2
        x1 = max(0, int(current_x - half_size))
        y1 = max(0, int(current_y - half_size))
        x2 = min(width, int(current_x + half_size))
        y2 = min(height, int(current_y + half_size))

        # Draw the square (may be slightly off-grid)
        img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1.0

        # Apply Gaussian blur for smooth edges
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

        # Convert to uint8
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        # Convert to 3-channel for video
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        frames.append(img_color)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Generating {num_frames} frames of {width}x{height} video...")
    print(f"Motion frequency: {FREQ} Hz")
    print(f"Motion amplitude: ±{motion_amplitude} pixels")
    print(f"Gaussian blur sigma: {blur_sigma}")

    for frame in frames:
        out.write(frame)

    # Release video writer
    out.release()

    print(f"Subtle motion test video saved to: {output_path}")
    print(f"Video properties: {num_frames} frames, {fps} FPS, {width}x{height} resolution")
    print(f"Square moves in a circular pattern with radius {motion_amplitude} pixels")

if __name__ == "__main__":
    # Generate the subtle test video
    generate_subtle_test_video()