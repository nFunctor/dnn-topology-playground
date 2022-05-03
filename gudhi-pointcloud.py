from sklearn.manifold import MDS
import matplotlib.pyplot as plot
import numpy as np
import gudhi


epoch = 101


activs = np.load(f"plot/activations_epoch_{epoch}.npz")
dist = np.load(f"plot/distances_epoch_{epoch}.npy")
t = np.array(activs["activs"]).astype(np.double)

#t = np.random.randn(192, 1024)

#embedding=MDS(n_components=3)
#t = embedding.fit_transform(t)

# Only if you are willing to center the activations

#t = t - t.mean(1).reshape(t.shape[0], -1) ## substract averages
#t = np.nan_to_num(t/t.std(1).reshape(t.shape[0], -1))

#t = t[0:128, ]

rips = gudhi.RipsComplex(points=t, max_edge_length=42)

#rips = gudhi.RipsComplex(distance_matrix=dist, max_edge_length=100)

simplex_tree = rips.create_simplex_tree(max_dimension=2)


diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
print("diag=", diag)

gudhi.plot_persistence_diagram(diag)
plot.show()