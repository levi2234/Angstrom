import cv2
import numpy as np
import os

def generate_test_video(output_path="test_square_motion.mp4", fps=30):
    """
    Generate a test video with 100 frames of a 50x50 image containing a square
    that pulses by growing and shrinking periodically.

    Args:
        output_path (str): Path to save the output video
        fps (int): Frames per second for the video
    """

    # Video parameters
    width, height = 50, 50
    num_frames = 100

    # Square parameters
    base_square_size = 10  # Base square size
    square_center_x = width // 2
    square_center_y = height // 2

    # Pulsing parameters
    pulse_frequency = 2.0  # Hz - how fast the square pulses
    pulse_amplitude = 8    # Maximum additional size for pulsing

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"Generating {num_frames} frames of {width}x{height} video...")
    print(f"Pulse frequency: {pulse_frequency} Hz")
    print(f"Pulse amplitude: ±{pulse_amplitude} pixels")

    current_square_size = base_square_size  # Initialize for final print

    for frame_num in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width), dtype=np.uint8)

        # Calculate time in seconds
        time_seconds = frame_num / fps

        # Calculate pulsing size using a sine wave
        # This creates a smooth pulsing motion
        pulse_offset = pulse_amplitude * np.sin(2 * np.pi * pulse_frequency * time_seconds)
        current_square_size = base_square_size + pulse_offset

        # Ensure square doesn't exceed frame boundaries
        current_square_size = max(2, min(current_square_size, min(width, height) - 2))

        # Calculate square boundaries
        half_size = int(current_square_size // 2)
        x1 = max(0, square_center_x - half_size)
        y1 = max(0, square_center_y - half_size)
        x2 = min(width, square_center_x + half_size)
        y2 = min(height, square_center_y + half_size)

        # Draw the square (white on black background)
        frame[y1:y2, x1:x2] = 255

        # Write frame to video
        out.write(frame)

        # Print progress every 10 frames
        if frame_num % 10 == 0:
            print(f"Frame {frame_num}/{num_frames}, Square size: {current_square_size:.1f}x{current_square_size:.1f}")

    # Release video writer
    out.release()

    print(f"Test video saved to: {output_path}")
    print(f"Video properties: {num_frames} frames, {fps} FPS, {width}x{height} resolution")
    print(f"Square pulses from {base_square_size - pulse_amplitude:.1f}x{base_square_size - pulse_amplitude:.1f} to {base_square_size + pulse_amplitude:.1f}x{base_square_size + pulse_amplitude:.1f}")

if __name__ == "__main__":
    # Generate the test video
    generate_test_video()