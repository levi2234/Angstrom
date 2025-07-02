import cv2
import torch

def read_video_frames(video_path):
    """Read video frames and return them as a PyTorch tensor.

    Args:
        video_path (str): Path to the video file.

    Returns:
        torch.Tensor: A tensor of shape [N, C, H, W], where N is the number of frames,
                      C is the number of channels (1 for grayscale),
                      and H and W are the height and width of the frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Shape: [H, W]
        # Add a channel dimension (C=1) and normalize to [0, 1]
        frame_tensor = torch.from_numpy(gray_frame).float().unsqueeze(0) / 255.0  # Shape: [1, H, W]
        frames.append(frame_tensor)
    cap.release()

    # Stack frames into a single tensor: [N, C, H, W]
    video_tensor = torch.stack(frames, dim=0)
    return video_tensor

def write_video_frames(frames, output_path, fps):
    """Write frames to a video file.

    Args:
        frames (list): List of frames, each of shape [H, W, C] or [H, W]
        output_path (str): Path to save the output video
        fps (float): Frames per second for the output video
    """
    if not frames:
        raise ValueError("No frames provided")

    # Get dimensions from first frame
    first_frame = frames[0]
    if len(first_frame.shape) == 2:  # Grayscale
        height, width = first_frame.shape
    else:  # Color
        height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
