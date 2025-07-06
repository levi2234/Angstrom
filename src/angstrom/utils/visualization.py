import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Union, Optional, Tuple
import cv2


def visualize_pyramid_phases(phase_coeffs_list: List,
                           frame_idx: int = 0,
                           max_levels: int = 5,
                           max_bands: int = 4,
                           figsize: Tuple[int, int] = (20, 15),
                           save_path: Optional[str] = None) -> None:
    """
    Visualize pyramid phases from motion amplification output.

    Args:
        phase_coeffs_list: List of phase coefficients for each frame
        frame_idx: Index of frame to visualize (default: 0)
        max_levels: Maximum number of pyramid levels to display
        max_bands: Maximum number of orientation bands to display per level
        figsize: Figure size for the plot
        save_path: Optional path to save the visualization
    """
    if frame_idx >= len(phase_coeffs_list):
        print(f"Frame index {frame_idx} out of range. Available frames: {len(phase_coeffs_list)}")
        return

    frame_coeffs = phase_coeffs_list[frame_idx]

    # Analyze pyramid structure
    num_levels = len(frame_coeffs)
    print(f"Pyramid structure: {num_levels} levels")

    # Count total bands for subplot layout
    total_bands = 0
    level_info = []

    for i, level in enumerate(frame_coeffs[:max_levels]):
        if isinstance(level, list):
            num_bands = min(len(level), max_bands)
            level_info.append((i, num_bands, 'bandpass'))
            total_bands += num_bands
        else:
            level_info.append((i, 1, 'lowpass/highpass'))
            total_bands += 1

    # Create subplot grid
    cols = max_bands
    rows = (total_bands + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Pyramid Phase Visualization - Frame {frame_idx}', fontsize=16)

    plot_idx = 0
    for level_idx, num_bands, level_type in level_info:
        level = frame_coeffs[level_idx]

        if isinstance(level, list):  # Bandpass filters
            for band_idx in range(num_bands):
                row = plot_idx // cols
                col = plot_idx % cols

                band_data = level[band_idx]
                if isinstance(band_data, torch.Tensor):
                    band_data = band_data.cpu().numpy()

                # Normalize phase to [-π, π] for visualization
                phase_normalized = np.mod(band_data + np.pi, 2*np.pi) - np.pi

                im = axes[row, col].imshow(phase_normalized, cmap='twilight', aspect='auto')
                axes[row, col].set_title(f'Level {level_idx}, Band {band_idx}\n({level_type})')
                axes[row, col].set_xlabel('X')
                axes[row, col].set_ylabel('Y')

                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[row, col], shrink=0.8)
                cbar.set_label('Phase (radians)')

                plot_idx += 1
        else:  # Lowpass/Highpass filter
            row = plot_idx // cols
            col = plot_idx % cols

            if isinstance(level, torch.Tensor):
                level_data = level.cpu().numpy()
            else:
                level_data = level

            # For lowpass/highpass, we might have real values instead of phase
            if np.iscomplexobj(level_data):
                phase_data = np.angle(level_data)
            else:
                phase_data = level_data

            im = axes[row, col].imshow(phase_data, cmap='viridis', aspect='auto')
            axes[row, col].set_title(f'Level {level_idx}\n({level_type})')
            axes[row, col].set_xlabel('X')
            axes[row, col].set_ylabel('Y')

            cbar = plt.colorbar(im, ax=axes[row, col], shrink=0.8)
            cbar.set_label('Value')

            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def visualize_phase_temporal_evolution(phase_coeffs_list: List,
                                     level_idx: int = 0,
                                     band_idx: int = 0,
                                     pixel_pos: Optional[Tuple[int, int]] = None,
                                     figsize: Tuple[int, int] = (15, 10),
                                     save_path: Optional[str] = None) -> None:
    """
    Visualize temporal evolution of phase at a specific pyramid level and band.

    Args:
        phase_coeffs_list: List of phase coefficients for each frame
        level_idx: Pyramid level to analyze
        band_idx: Orientation band to analyze
        pixel_pos: Specific pixel position (y, x) to track. If None, uses center pixel
        figsize: Figure size for the plot
        save_path: Optional path to save the visualization
    """
    if not phase_coeffs_list:
        print("No phase coefficients provided")
        return

    # Extract temporal sequence for the specified level and band
    temporal_sequence = []
    valid_frames = []

    for frame_idx, frame_coeffs in enumerate(phase_coeffs_list):
        if level_idx < len(frame_coeffs):
            level = frame_coeffs[level_idx]

            if isinstance(level, list) and band_idx < len(level):
                band_data = level[band_idx]
                if isinstance(band_data, torch.Tensor):
                    temporal_sequence.append(band_data.cpu().numpy())
                else:
                    temporal_sequence.append(band_data)
                valid_frames.append(frame_idx)

    if not temporal_sequence:
        print(f"No valid data found for level {level_idx}, band {band_idx}")
        return

    # Stack into [T, H, W] array
    phase_band = np.stack(temporal_sequence, axis=0)

    # Determine pixel position
    if pixel_pos is None:
        h, w = phase_band.shape[1], phase_band.shape[2]
        pixel_y, pixel_x = h // 2, w // 2
    else:
        pixel_y, pixel_x = pixel_pos

    # Extract temporal signal for the pixel
    temporal_signal = phase_band[:, pixel_y, pixel_x]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Phase Temporal Evolution - Level {level_idx}, Band {band_idx}', fontsize=16)

    # Plot 1: Temporal phase evolution at specific pixel
    axes[0, 0].plot(valid_frames, temporal_signal, 'b-', linewidth=2)
    axes[0, 0].set_title(f'Phase Evolution at Pixel ({pixel_x}, {pixel_y})')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Phase (radians)')
    axes[0, 0].grid(True)

    # Plot 2: Phase differences from first frame
    base_phase = temporal_signal[0]
    phase_differences = temporal_signal - base_phase
    # Handle phase wrapping
    phase_differences = np.mod(phase_differences + np.pi, 2*np.pi) - np.pi

    axes[0, 1].plot(valid_frames, phase_differences, 'r-', linewidth=2)
    axes[0, 1].set_title('Phase Differences from First Frame')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Phase Difference (radians)')
    axes[0, 1].grid(True)

    # Plot 3: Spatial phase distribution at middle frame
    mid_frame_idx = len(phase_band) // 2
    mid_frame_phase = phase_band[mid_frame_idx]

    im1 = axes[1, 0].imshow(mid_frame_phase, cmap='twilight', aspect='auto')
    axes[1, 0].set_title(f'Spatial Phase Distribution (Frame {valid_frames[mid_frame_idx]})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[1, 0])

    # Mark the tracked pixel
    axes[1, 0].plot(pixel_x, pixel_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)

    # Plot 4: Phase statistics over time
    phase_std = np.std(phase_band, axis=(1, 2))
    phase_mean = np.mean(phase_band, axis=(1, 2))

    axes[1, 1].plot(valid_frames, phase_mean, 'g-', linewidth=2, label='Mean')
    axes[1, 1].plot(valid_frames, phase_std, 'm-', linewidth=2, label='Std Dev')
    axes[1, 1].set_title('Phase Statistics Over Time')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal evolution visualization saved to {save_path}")

    plt.show()


def visualize_phase_comparison(original_phases: List,
                             amplified_phases: List,
                             frame_idx: int = 0,
                             level_idx: int = 0,
                             band_idx: int = 0,
                             figsize: Tuple[int, int] = (20, 10),
                             save_path: Optional[str] = None) -> None:
    """
    Compare original and amplified phases side by side.

    Args:
        original_phases: List of original phase coefficients
        amplified_phases: List of amplified phase coefficients
        frame_idx: Frame index to compare
        level_idx: Pyramid level to compare
        band_idx: Orientation band to compare
        figsize: Figure size for the plot
        save_path: Optional path to save the visualization
    """
    if frame_idx >= len(original_phases) or frame_idx >= len(amplified_phases):
        print("Frame index out of range")
        return

    # Extract phase data
    orig_frame = original_phases[frame_idx]
    amp_frame = amplified_phases[frame_idx]

    if level_idx >= len(orig_frame) or level_idx >= len(amp_frame):
        print("Level index out of range")
        return

    orig_level = orig_frame[level_idx]
    amp_level = amp_frame[level_idx]

    if isinstance(orig_level, list):
        if band_idx >= len(orig_level) or band_idx >= len(amp_level):
            print("Band index out of range")
            return

        orig_band = orig_level[band_idx]
        amp_band = amp_level[band_idx]
    else:
        orig_band = orig_level
        amp_band = amp_level

    # Convert to numpy if needed
    if isinstance(orig_band, torch.Tensor):
        orig_data = orig_band.cpu().numpy()
    else:
        orig_data = orig_band

    if isinstance(amp_band, torch.Tensor):
        amp_data = amp_band.cpu().numpy()
    else:
        amp_data = amp_band

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Phase Comparison - Frame {frame_idx}, Level {level_idx}, Band {band_idx}', fontsize=16)

    # Normalize phases for visualization
    orig_normalized = np.mod(orig_data + np.pi, 2*np.pi) - np.pi
    amp_normalized = np.mod(amp_data + np.pi, 2*np.pi) - np.pi

    # Plot 1: Original phase
    im1 = axes[0, 0].imshow(orig_normalized, cmap='twilight', aspect='auto')
    axes[0, 0].set_title('Original Phase')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Amplified phase
    im2 = axes[0, 1].imshow(amp_normalized, cmap='twilight', aspect='auto')
    axes[0, 1].set_title('Amplified Phase')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot 3: Phase difference
    phase_diff = amp_normalized - orig_normalized
    # Handle phase wrapping for difference
    phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi

    im3 = axes[0, 2].imshow(phase_diff, cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('Phase Difference (Amplified - Original)')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[0, 2])

    # Plot 4: Histogram comparison
    axes[1, 0].hist(orig_normalized.flatten(), bins=50, alpha=0.7, label='Original', color='blue')
    axes[1, 0].hist(amp_normalized.flatten(), bins=50, alpha=0.7, label='Amplified', color='red')
    axes[1, 0].set_title('Phase Distribution')
    axes[1, 0].set_xlabel('Phase (radians)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 5: Line plot through center
    h, w = orig_normalized.shape
    center_y = h // 2

    axes[1, 1].plot(orig_normalized[center_y, :], 'b-', linewidth=2, label='Original')
    axes[1, 1].plot(amp_normalized[center_y, :], 'r-', linewidth=2, label='Amplified')
    axes[1, 1].set_title(f'Phase Along Center Row (y={center_y})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot 6: Statistics comparison
    stats_data = {
        'Mean': [np.mean(orig_normalized), np.mean(amp_normalized)],
        'Std': [np.std(orig_normalized), np.std(amp_normalized)],
        'Min': [np.min(orig_normalized), np.min(amp_normalized)],
        'Max': [np.max(orig_normalized), np.max(amp_normalized)]
    }

    x = np.arange(len(stats_data))
    width = 0.35

    axes[1, 2].bar(x - width/2, [stats_data[key][0] for key in stats_data], width, label='Original', alpha=0.7)
    axes[1, 2].bar(x + width/2, [stats_data[key][1] for key in stats_data], width, label='Amplified', alpha=0.7)
    axes[1, 2].set_title('Phase Statistics Comparison')
    axes[1, 2].set_xlabel('Statistic')
    axes[1, 2].set_ylabel('Value (radians)')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(stats_data.keys())
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase comparison visualization saved to {save_path}")

    plt.show()


def create_phase_video(phase_coeffs_list: List,
                      level_idx: int = 0,
                      band_idx: int = 0,
                      output_path: str = "phase_visualization.mp4",
                      fps: int = 30,
                      colormap: str = 'twilight') -> None:
    """
    Create a video visualization of phase evolution over time.

    Args:
        phase_coeffs_list: List of phase coefficients for each frame
        level_idx: Pyramid level to visualize
        band_idx: Orientation band to visualize
        output_path: Path to save the output video
        fps: Frames per second for the output video
        colormap: Matplotlib colormap to use
    """
    if not phase_coeffs_list:
        print("No phase coefficients provided")
        return

    # Extract temporal sequence
    temporal_sequence = []
    for frame_coeffs in phase_coeffs_list:
        if level_idx < len(frame_coeffs):
            level = frame_coeffs[level_idx]

            if isinstance(level, list) and band_idx < len(level):
                band_data = level[band_idx]
                if isinstance(band_data, torch.Tensor):
                    temporal_sequence.append(band_data.cpu().numpy())
                else:
                    temporal_sequence.append(band_data)

    if not temporal_sequence:
        print(f"No valid data found for level {level_idx}, band {band_idx}")
        return

    # Stack into [T, H, W] array
    phase_band = np.stack(temporal_sequence, axis=0)

    # Normalize phases for visualization
    phase_normalized = np.mod(phase_band + np.pi, 2*np.pi) - np.pi

    # Convert to 0-255 range for video
    phase_min, phase_max = np.min(phase_normalized), np.max(phase_normalized)
    phase_scaled = ((phase_normalized - phase_min) / (phase_max - phase_min) * 255).astype(np.uint8)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_frames = []

    for frame in phase_scaled:
        colored_frame = cmap(frame)[:, :, :3]  # Remove alpha channel
        colored_frame = (colored_frame * 255).astype(np.uint8)
        colored_frames.append(colored_frame)

    # Create video writer
    h, w = colored_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Write frames
    for i, frame in enumerate(colored_frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add frame number text
        cv2.putText(frame_bgr, f'Frame {i}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f'Level {level_idx}, Band {band_idx}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame_bgr)

    out.release()
    print(f"Phase video saved to {output_path}")


def visualize_pyramid_structure(phase_coeffs_list: List,
                              frame_idx: int = 0,
                              figsize: Tuple[int, int] = (15, 10),
                              save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive visualization of the pyramid structure.

    Args:
        phase_coeffs_list: List of phase coefficients for each frame
        frame_idx: Frame index to visualize
        figsize: Figure size for the plot
        save_path: Optional path to save the visualization
    """
    if frame_idx >= len(phase_coeffs_list):
        print("Frame index out of range")
        return

    frame_coeffs = phase_coeffs_list[frame_idx]

    # Analyze structure
    structure_info = []
    for i, level in enumerate(frame_coeffs):
        if isinstance(level, list):
            structure_info.append({
                'level': i,
                'type': 'bandpass',
                'num_bands': len(level),
                'shapes': [band.shape if hasattr(band, 'shape') else 'scalar' for band in level]
            })
        else:
            structure_info.append({
                'level': i,
                'type': 'lowpass/highpass',
                'num_bands': 1,
                'shapes': [level.shape if hasattr(level, 'shape') else 'scalar']
            })

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Pyramid Structure Analysis - Frame {frame_idx}', fontsize=16)

    # Plot 1: Level types
    level_types = [info['type'] for info in structure_info]
    level_counts = [level_types.count('bandpass'), level_types.count('lowpass/highpass')]

    axes[0, 0].pie(level_counts, labels=['Bandpass', 'Lowpass/Highpass'], autopct='%1.1f%%')
    axes[0, 0].set_title('Level Type Distribution')

    # Plot 2: Number of bands per level
    levels = [info['level'] for info in structure_info]
    num_bands = [info['num_bands'] for info in structure_info]

    axes[0, 1].bar(levels, num_bands, color='skyblue')
    axes[0, 1].set_title('Number of Bands per Level')
    axes[0, 1].set_xlabel('Level')
    axes[0, 1].set_ylabel('Number of Bands')
    axes[0, 1].grid(True)

    # Plot 3: Level type by level
    colors = ['red' if info['type'] == 'bandpass' else 'blue' for info in structure_info]
    axes[1, 0].bar(levels, [1] * len(levels), color=colors)
    axes[1, 0].set_title('Level Types')
    axes[1, 0].set_xlabel('Level')
    axes[1, 0].set_ylabel('Type')
    axes[1, 0].set_yticks([])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Bandpass'),
                      Patch(facecolor='blue', label='Lowpass/Highpass')]
    axes[1, 0].legend(handles=legend_elements)

    # Plot 4: Structure summary
    axes[1, 1].axis('off')
    summary_text = f"Pyramid Structure Summary:\n\n"
    summary_text += f"Total Levels: {len(structure_info)}\n"
    summary_text += f"Bandpass Levels: {level_counts[0]}\n"
    summary_text += f"Lowpass/Highpass Levels: {level_counts[1]}\n"
    summary_text += f"Total Bands: {sum(num_bands)}\n\n"

    summary_text += "Level Details:\n"
    for info in structure_info:
        summary_text += f"Level {info['level']}: {info['type']} ({info['num_bands']} bands)\n"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pyramid structure visualization saved to {save_path}")

    plt.show()