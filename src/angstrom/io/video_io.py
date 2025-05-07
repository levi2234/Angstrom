import cv2
import torch

def read_video_frames(video_path):
    """
    Reads video frames and returns them as a PyTorch tensor.

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
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
