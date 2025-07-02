"""Unit tests for the video I/O module."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import cv2

from angstrom.io.video_io import read_video_frames, write_video_frames


class TestVideoIO:
    """Test cases for video I/O functions."""

    @pytest.fixture
    def mock_video_frames(self):
        """Create mock video frames for testing."""
        return [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)
        ]

    @pytest.fixture
    def mock_grayscale_frames(self):
        """Create mock grayscale frames for testing."""
        return [
            np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(5)
        ]

    @patch('cv2.VideoCapture')
    def test_read_video_frames_success(self, mock_video_capture, mock_video_frames):
        """Test successful video frame reading."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame) for frame in mock_video_frames] + [(False, None)]
        mock_video_capture.return_value = mock_cap

        # Test reading
        result = read_video_frames("test_video.mp4")

        # Verify video capture was called
        mock_video_capture.assert_called_once_with("test_video.mp4")
        mock_cap.release.assert_called_once()

        # Verify result
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 1, 64, 64)  # 5 frames, 1 channel, 64x64
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)  # Normalized to [0, 1]

    @patch('cv2.VideoCapture')
    def test_read_video_frames_empty_video(self, mock_video_capture):
        """Test reading empty video."""
        # Mock video capture with no frames
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        # Test reading - should raise error due to empty tensor list
        with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
            read_video_frames("empty_video.mp4")

    @patch('cv2.VideoCapture')
    def test_read_video_frames_file_not_found(self, mock_video_capture):
        """Test reading non-existent video file."""
        mock_video_capture.side_effect = Exception("File not found")

        with pytest.raises(Exception):
            read_video_frames("nonexistent_video.mp4")

    @patch('cv2.VideoCapture')
    def test_read_video_frames_different_sizes(self, mock_video_capture):
        """Test reading video with frames of different sizes."""
        # Create frames of different sizes
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        ]

        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        mock_video_capture.return_value = mock_cap

        # Test reading - should raise error due to different sizes
        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            read_video_frames("test_video.mp4")

    @patch('cv2.VideoWriter')
    def test_write_video_frames_success(self, mock_video_writer, mock_video_frames):
        """Test successful video frame writing."""
        output_path = "test_output.mp4"
        fps = 30.0

        # Mock video writer
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer

        # Test writing
        write_video_frames(mock_video_frames, output_path, fps)

        # Verify video writer was called
        mock_video_writer.assert_called_once()
        assert mock_writer.write.call_count == 5  # 5 frames
        mock_writer.release.assert_called_once()

    @patch('cv2.VideoWriter')
    def test_write_video_frames_grayscale(self, mock_video_writer, mock_grayscale_frames):
        """Test writing grayscale video frames."""
        output_path = "test_output.mp4"
        fps = 30.0

        # Mock video writer
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer

        # Test writing
        write_video_frames(mock_grayscale_frames, output_path, fps)

        # Verify video writer was called
        mock_video_writer.assert_called_once()
        assert mock_writer.write.call_count == 5  # 5 frames
        mock_writer.release.assert_called_once()

    def test_write_video_frames_empty_list(self):
        """Test writing empty frame list."""
        with pytest.raises(ValueError, match="No frames provided"):
            write_video_frames([], "test_output.mp4", 30.0)

    def test_write_video_frames_none_frames(self):
        """Test writing None frames."""
        with pytest.raises(ValueError, match="No frames provided"):
            write_video_frames(None, "test_output.mp4", 30.0)

    @patch('cv2.VideoWriter')
    def test_write_video_frames_different_formats(self, mock_video_writer):
        """Test writing frames in different formats."""
        # Test with numpy arrays
        numpy_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

        # Test with torch tensors
        torch_frames = [torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8).numpy() for _ in range(3)]

        # Mock video writer
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer

        # Test numpy frames
        write_video_frames(numpy_frames, "numpy_output.mp4", 30.0)
        assert mock_writer.write.call_count == 3

        # Reset mock
        mock_writer.reset_mock()

        # Test torch frames
        write_video_frames(torch_frames, "torch_output.mp4", 30.0)
        assert mock_writer.write.call_count == 3

    @patch('cv2.VideoWriter')
    def test_write_video_frames_invalid_fps(self, mock_video_writer, mock_video_frames):
        """Test writing video with invalid FPS."""
        output_path = "test_output.mp4"

        # Mock video writer
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer

        # Test with zero FPS
        write_video_frames(mock_video_frames, output_path, 0.0)
        # Should not raise an error, but FPS should be passed to VideoWriter

        # Test with negative FPS
        write_video_frames(mock_video_frames, output_path, -1.0)
        # Should not raise an error, but FPS should be passed to VideoWriter

    def test_read_video_frames_normalization(self):
        """Test that video frames are properly normalized."""
        # Create a frame with known values
        test_frame = np.full((64, 64, 3), 128, dtype=np.uint8)

        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.side_effect = [(True, test_frame), (False, None)]
            mock_video_capture.return_value = mock_cap

            result = read_video_frames("test_video.mp4")

            # Check normalization: 128/255 ≈ 0.502
            expected_value = 128.0 / 255.0
            assert torch.allclose(result[0, 0], torch.full((64, 64), expected_value), atol=1e-6)

