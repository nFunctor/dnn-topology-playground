# This requires installing gudhi, to be added to requirements.txt later
# To install manually, do pip install gudhi
from sklearn.manifold import MDS
import matplotlib.pyplot as plot
import numpy as np
import gudhi
from graph import spl2metric_rough, spl2metricmax

print("#####################################################################")
print("Computing Bottleneck distance between two diagrams")

epoch1 = 101
epoch2 = 102

activs1 = np.load(f"plot/activations_epoch_{epoch1}.npz")
activs2 = np.load(f"plot/activations_epoch_{epoch2}.npz")

t1 = np.array(activs1["activs"]).astype(np.double)
t2 = np.array(activs2["activs"]).astype(np.double)

dist1 = np.load(f"plot/distances_epoch_{epoch1}.npy")
dist2 = np.load(f"plot/distances_epoch_{epoch2}.npy")

#if willing to compute any other distance "sur place", use
#dist1 = spl2metricmax(t1)
#dist2 = spl2metricmax(t2)






#t = np.random.randn(192, 1024)

#embedding=MDS(n_components=3)
#t = embedding.fit_transform(t)

# Only if you are willing to center the activations

#t1 = t1 - t1.mean(1).reshape(t1.shape[0], -1) ## substract averages
#t1 = np.nan_to_num(t1/t1.std(1).reshape(t1.shape[0], -1))

#t2 = t2 - t2.mean(1).reshape(t2.shape[0], -1) ## substract averages
#t2 = np.nan_to_num(t2/t2.std(1).reshape(t2.shape[0], -1))


#t = t[0:128, ]

rips1 = gudhi.RipsComplex(points=t1, max_edge_length=100)
rips2 = gudhi.RipsComplex(points=t2, max_edge_length=100)

#rips1 = gudhi.RipsComplex(distance_matrix=dist1, max_edge_length=100)
#rips2 = gudhi.RipsComplex(distance_matrix=dist2, max_edge_length=100)


simplex_tree1 = rips1.create_simplex_tree(max_dimension=2)
simplex_tree2 = rips2.create_simplex_tree(max_dimension=2)

simplex_tree1.persistence(homology_coeff_field=2, min_persistence=0)
simplex_tree2.persistence(homology_coeff_field=2, min_persistence=0)

diag1 = simplex_tree1.persistence_intervals_in_dimension(1)
diag2 = simplex_tree2.persistence_intervals_in_dimension(1)

#print("diag1=", diag1)

message = "Bottleneck distance value = " + '%.2f' % gudhi.bottleneck_distance(diag1, diag2, e=0)
print(message)

gudhi.plot_persistence_diagram(diag1)
gudhi.plot_persistence_diagram(diag2)
plot.show()