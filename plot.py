from sklearn.manifold import MDS
import numpy as np

#%%

ncomp = 3 ##dimension of MDS, for now only 3 is implemented
epoch1 = 30
epoch2 = 50
#%%

embedding=MDS(n_components=3)

#%%

activs1 = np.load(f"plot/activations_epoch_{epoch1}.npy")
activs2 = np.load(f"plot/activations_epoch_{epoch2}.npy")
c=activs1.shape[0]


#%%


t = np.concatenate((np.array(activs1).astype(np.double),np.array(activs2).astype(np.double)),axis=0)

#Only if you are willing to center the activations

#t = t - t.mean(1).reshape(t.shape[0], -1) ## substract averages
#t = np.nan_to_num(t/t.std(1).reshape(t.shape[0], -1))


#%%

t = embedding.fit_transform(t)
t.shape

#%%

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

#%%

fig=pyplot.figure()
ax=Axes3D(fig)
x1 = t[0:c,0]
y1 = t[0:c,1]
z1 = t[0:c,2]

x2 = t[c:,0]
y2 = t[c:,1]
z2 = t[c:,2]
ax.scatter(x1,y1,z1)
ax.scatter(x2,y2,z2)

pyplot.show()
