import numpy as np
import matplotlib.pyplot as plt


def show_em_status(iteration, x, obj_hist, Y, true_angles, est_angles):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    axes[0, 0].imshow(x, cmap="gray")
    axes[0, 0].set_title(f"EM reconstruction at iter {iteration}")
    axes[0, 0].axis("off")

    axes[0, 1].plot(obj_hist)
    axes[0, 1].set_title("Objective")
    axes[0, 1].set_xlabel("iteration")

    axes[1, 0].imshow(Y.T, cmap="gray", aspect="auto", origin="lower")
    axes[1, 0].set_title("Observed projections")
    axes[1, 0].set_xlabel("projection index")
    axes[1, 0].set_ylabel("detector bin")

    axes[1, 1].scatter(np.arange(len(true_angles)), true_angles, s=10, label="true")
    axes[1, 1].scatter(np.arange(len(est_angles)), est_angles, s=10, label="estimated")
    axes[1, 1].set_title("Angles")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()