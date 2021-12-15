""" plot.py """
from typing import List

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS


def main(epochs: int) -> None:
    """show plots"""
    activations = np.load(f"plot/activations_epoch_{epochs}.npz")
    layer_sizes = activations["layer_dims"]
    transformed_activations = MDS(n_components=3).fit_transform(
        np.array(activations["activs"]).astype(np.double)
    )
    fig = pyplot.figure()
    axis = Axes3D(fig, auto_add_to_figure=False)
    for i in range(1, len(layer_sizes) + 1):
        layer_activations = transformed_activations[
            sum(layer_sizes[: i - 1]) : sum(layer_sizes[:i]), :
        ]
        axis.scatter(
            layer_activations[:, 0],
            layer_activations[:, 1],
            layer_activations[:, 2],
        )
    fig.add_axes(axis)
    pyplot.show()


if __name__ == "__main__":
    main(1)
