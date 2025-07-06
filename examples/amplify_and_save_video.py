import os
import sys
import numpy as np
import torch

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.core.motion_amplifier import MotionAmplifier
from angstrom.io.video_io import write_video_frames

#INPUT_VIDEO = "test_subtle_motion.mp4"
INPUT_VIDEO = "src/angstrom/data/testvideos/pulse_test.mp4"
OUTPUT_VIDEO = "amplified_output.mp4"
AMPLIFICATION_FACTOR = 10
FREQ_RANGE = (0.1, 4)  # Hz, matches the 0.3 Hz FFT bin

if not os.path.exists(INPUT_VIDEO):
    print(f"Input video not found: {INPUT_VIDEO}")
    sys.exit(1)

print(f"Loading and amplifying {INPUT_VIDEO} ...")

amplifier = MotionAmplifier()
amplifier.load_video(INPUT_VIDEO)
amplifier.process()
amplified_video = amplifier.amplify(
    amplification_factor=AMPLIFICATION_FACTOR,
    frequency_range=FREQ_RANGE
)

# Convert to numpy and save as video
if amplified_video is not None and len(amplified_video) > 0:
    # Move to CPU and convert to numpy
    video_np = amplified_video.cpu().numpy()  # [N, C, H, W] or [N, C, 1, H, W]
    print(f"Amplified video tensor shape: {amplified_video.shape}")
    print(f"Amplified video tensor dtype: {amplified_video.dtype}")

    # Remove singleton dimensions
    if video_np.ndim == 5 and video_np.shape[2] == 1:
        # Handle [N, C, 1, H, W] -> [N, C, H, W]
        video_np = video_np[:, :, 0, :, :]
        print(f"Numpy video shape after removing singleton dim: {video_np.shape}")

    # Remove channel dimension if single channel
    if video_np.shape[1] == 1:
        video_np = video_np[:, 0]  # [N, H, W]
    print(f"Numpy video shape after channel removal: {video_np.shape}")

    # Rescale to 0-255 for saving
    video_np = (video_np * 255.0).clip(0, 255).astype('uint8')
    print(f"Numpy video shape after rescaling: {video_np.shape}")
    print(f"Numpy video dtype after rescaling: {video_np.dtype}")
    print(f"First frame min/max: {video_np[0].min()}/{video_np[0].max()}")

    # Convert grayscale to color (3 channels) as expected by write_video_frames
    if video_np.ndim == 3:  # [N, H, W] grayscale
        # Convert to color by repeating the grayscale channel 3 times
        video_np = np.stack([video_np] * 3, axis=-1)  # [N, H, W, 3]
        print(f"Final video shape: {video_np.shape}")
        print(f"Final video dtype: {video_np.dtype}")
        print(f"Final first frame min/max: {video_np[0].min()}/{video_np[0].max()}")

    print(f"Saving amplified video to {OUTPUT_VIDEO} ...")
    write_video_frames(video_np, OUTPUT_VIDEO, fps=amplifier.video_fps)
    print(f"Done! Output saved as {OUTPUT_VIDEO}")
else:
    print("No amplified video generated.")

