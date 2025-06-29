import os
import sys
import numpy as np
import cv2

# Add the src directory to the path so we can import angstrom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from angstrom.io.video_io import write_video_frames

def test_video_writing():
    """
    Test if video writing works correctly with a simple test video.
    """
    print("=== TESTING VIDEO WRITING ===")

    # Create a simple test video: 10 frames with a moving square
    frames = []
    for i in range(10):
        # Create a 50x50 frame
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        # Add a white square that moves
        x = 20 + i * 2  # Square moves right
        y = 20 + (i % 3) * 5  # Square moves up and down
        frame[y:y+10, x:x+10] = [255, 255, 255]  # White square

        frames.append(frame)

    frames = np.array(frames)
    print(f"Created test video with shape: {frames.shape}")
    print(f"First frame shape: {frames[0].shape}")
    print(f"First frame min/max: {frames[0].min()}/{frames[0].max()}")

    # Try to save using write_video_frames
    output_path = "test_video_writing.mp4"
    try:
        print(f"Saving test video to {output_path}...")
        write_video_frames(frames, output_path, fps=30.0)

        # Check if file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"File created successfully! Size: {file_size} bytes")

            if file_size > 0:
                print("✅ Video writing works correctly!")
            else:
                print("❌ File is empty - video writing failed")
        else:
            print("❌ File was not created")

    except Exception as e:
        print(f"❌ Error writing video: {e}")
        import traceback
        traceback.print_exc()

    # Also try using OpenCV directly as a comparison
    output_path2 = "test_video_opencv.mp4"
    try:
        print(f"\nTrying OpenCV directly to {output_path2}...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path2, fourcc, 30.0, (50, 50))

        for frame in frames:
            out.write(frame)
        out.release()

        if os.path.exists(output_path2):
            file_size = os.path.getsize(output_path2)
            print(f"OpenCV file size: {file_size} bytes")
        else:
            print("OpenCV file was not created")

    except Exception as e:
        print(f"OpenCV error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== VIDEO WRITING TEST COMPLETE ===")

if __name__ == "__main__":
    test_video_writing()