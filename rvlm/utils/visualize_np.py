import matplotlib
# matplotlib.use("Agg")  # non-interactive backend, no display overhead
import matplotlib.pyplot as plt
import numpy as np

def plot_full_paths_on_first_frame(
    video: np.ndarray,
    language_instruction: str,
    paths_out: dict,
    plot_legend: bool = True,
) -> np.ndarray:
    first_frame = video[0]

    fig, ax = plt.subplots()
    ax.imshow(first_frame)
    ax.axis("off")

    for label, traj_norm in paths_out.items():
        ax.plot(
            traj_norm[:, 0].astype(np.int32),
            traj_norm[:, 1].astype(np.int32),
            label=label,
        )
        ax.scatter(traj_norm[0, 0], traj_norm[0, 1], marker="x", s=100, color="red")
        ax.scatter(traj_norm[-1, 0], traj_norm[-1, 1], marker="x", s=100, color="green")

    ax.set_title(language_instruction)
    if plot_legend:
        ax.legend()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4) RGBA
    plt.close(fig)
    return img

def plot_rewards(
    tracking_rewards: np.ndarray,
    subtask_rewards: np.ndarray,
) -> np.ndarray:
    fig, ax = plt.subplots()
    ax.plot(tracking_rewards, label="tracking rewards")
    ax.plot(subtask_rewards, label="subtask rewards")
    ax.plot(subtask_rewards + tracking_rewards, label="total rewards")
    ax.legend()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img
