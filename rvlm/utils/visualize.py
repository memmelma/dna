import io
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_full_paths_on_first_frame(
    video: np.ndarray,
    language_instruction: str,
    paths_out: dict,
    plot_legend: bool = True,
):
    """
    Visualize full tracked paths overlaid on the first video frame.

    Args:
        video: (T, H, W, 3) uint8 array.
        language_instruction: Task description to put in the title.
        paths_out: dict[label] -> (T, 2) array of (x_norm, y_norm), each in [0, 1000].
    """
    first_frame = video[0]
    H, W = first_frame.shape[:2]

    plt.figure()
    plt.imshow(first_frame)
    plt.axis("off")

    for label, traj_norm in paths_out.items():
        traj_px = np.empty_like(traj_norm, dtype=np.float32)
        traj_px[:, 0] = traj_norm[:, 0]
        traj_px[:, 1] = traj_norm[:, 1]

        plt.plot(
            traj_px[:, 0].astype(np.int32),
            traj_px[:, 1].astype(np.int32),
            label=label,
        )

        start_xy = traj_px[0]
        end_xy = traj_px[-1]
        plt.scatter(start_xy[0], start_xy[1], marker="x", s=100, color="red")
        plt.scatter(end_xy[0], end_xy[1], marker="x", s=100, color="green")

    plt.title(language_instruction)
    if plot_legend:
        plt.legend()
    plt.show()

def plot_costs(tracking_rewards):

    costs = {k:[] for k in tracking_rewards[0].keys()}
    for tr in tracking_rewards:
        for k,v in tr.items():
            # subtract initial tracking reward (baseline)
            costs[k].append(v - tracking_rewards[0][k])

    dpi = 100
    width_px = 256
    height_px = 256

    fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)

    for k,v in costs.items():
        plt.plot(v, label=k)
    plt.plot(np.sum(list(costs.values()), axis=0), label="total")
    plt.legend(fontsize=8)
    plt.tight_layout()
    # smaller x and y ticks size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.xlim(0, 500)
    fig.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).resize((256, 256), Image.LANCZOS).convert("RGB")
    img_array = np.array(img)

    buf.close()
    plt.close('all')
    return img_array

def plot_alignment(alignment):
    dpi = 100
    width_px = 256
    height_px = 256

    fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)

    alignment.plot(type="threeway")

    fig.canvas.draw()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    # Convert to numpy array
    img = Image.open(buf).resize((256, 256), Image.LANCZOS).convert("RGB")
    img_array = np.array(img)

    buf.close()
    plt.close('all')
    return img_array

def plot_tracking_debug(img, paths, tracking_series, tracking_rewards, rewards=None, save_dir=None, unique_id=None, colors=None):
    """
    Plot all tracked paths against reference series on a single image.
    
    Args:
        img: The current image frame (H, W, C)
        paths: Dict of tracked paths from cotracker {label: (T, 2)}
        tracking_series: Dict of best-matching reference series {label: (T, 2)}
        tracking_rewards: Dict of rewards per object {label: float}
        rewards: Optional list of rewards to display
        save_dir: Optional directory to save the plot
        unique_id: Optional unique identifier for the saved file
    
    Returns:
        numpy.ndarray: RGB image array with same shape as input img (H, W, 3) with dtype uint8
    """
    
    # Get input image dimensions
    img_height, img_width = img.shape[:2]
    
    # Calculate figure size to match input image dimensions
    # DPI is set to 100, so figsize in inches = pixels / dpi
    dpi = 100
    fig_width = img_width / dpi
    fig_height = img_height / dpi
    
    # Create single figure with exact size
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Show image background
    ax.imshow(img)
    
    # Define colors for different objects
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths))) if colors is None else colors
    
    # Plot all tracks on the same image
    for idx, (k, color) in enumerate(zip(paths.keys(), colors)):
        # Plot tracked path (from cotracker)
        tracked = paths[k].astype(int)
        ax.plot(tracked[:, 0], tracked[:, 1], 
                color=color, linewidth=2, label=f'{k} (tracked)', alpha=0.8)
        
        # Plot reference series (best match)
        ref = tracking_series[k].astype(int)
        ax.plot(ref[:, 0], ref[:, 1], 
                color=color, linewidth=2, linestyle="--", 
                label=f'{k} (ref, R={tracking_rewards[k]:.3f})', alpha=0.8)
        
        # Mark start points
        ax.scatter(tracked[0, 0], tracked[0, 1], 
                  c=[color], marker='o', s=10, 
                  linewidth=2, zorder=5)
        ax.scatter(ref[0, 0], ref[0, 1], 
                  c=[color], marker='x', s=10, linewidth=2, zorder=5)
    
    # Add legend inside the image (top-right corner)
    # ax.legend(loc='upper right', fontsize=8, framealpha=0.7, 
    #          edgecolor='white', fancybox=True)
    
    # Add reward text inside the image (top-left corner)
    if rewards is not None:
        ax.text(0.02, 0.98, f"R: {rewards[-1]:.2f}", 
               transform=ax.transAxes, fontsize=5, 
               color='red', weight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.text(0.02, 0.94, f"R mean: {np.mean(rewards):.2f}", 
               transform=ax.transAxes, fontsize=5, 
               color='green', weight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add tracking rewards summary (below total reward)
    tracking_rewards_str = json.dumps({k: np.around(np.mean(v) if isinstance(v, (list, np.ndarray)) else v, 3) 
                                      for k, v in tracking_rewards.items()})
    ax.text(0.02, 0.90, f"R track: {tracking_rewards_str}", 
           transform=ax.transAxes, fontsize=5, 
           color='blue', weight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Remove axis ticks and labels
    ax.axis('off')
    
    # Remove all padding and margins to match exact input size
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert y-axis to match image coordinates
    
    # Save to file if requested
    if save_dir is not None and unique_id is not None:
        import os
        save_path = os.path.join(save_dir, f'tracking_debug_{unique_id}.png')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # Render to buffer and convert to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Convert buffer to numpy array
    pil_img = Image.open(buf).convert("RGB")
    img_array = np.array(pil_img)
    
    # Close buffer and figure
    buf.close()
    plt.close('all')
    
    # Ensure RGB format (remove alpha channel if present)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    # Resize to exact input dimensions if needed (bbox_inches='tight' might change size slightly)
    if img_array.shape[:2] != (img_height, img_width):
        pil_resize = Image.fromarray(img_array)
        pil_resize = pil_resize.resize((img_width, img_height), Image.LANCZOS)
        img_array = np.array(pil_resize)
    
    return img_array