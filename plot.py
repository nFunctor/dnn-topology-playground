#%% md



#%%

from sklearn.manifold import MDS
import numpy as np

#%%

ncomp = 3 ## the dimension of MDS, for now only 3 is implemented
epoch1 = 201
#epoch2 = 800

#%%

embedding=MDS(n_components=3)

#%%

activs1 = np.load(f"plot/activations_epoch_{epoch1}.npy")
#activs2 = np.load(f"plot/activations_epoch_{epoch2}.npy")
c_1=activs1.shape[0]

# Currently trying to draw multiple-layer networks, here is a constant that separates layers
c_1 = 64
c_2 = 64
#c_3 = 128

c_2 = c_2 + c_1
#c_3 = c_3 + c_2

#%%


# If you are comparing different epochs
#t = np.concatenate((np.array(activs1).astype(np.double),np.array(activs2).astype(np.double)),axis=0)
t = np.array(activs1).astype(np.double)


# Only if you are willing to center the activations

#t = t - t.mean(1).reshape(t.shape[0], -1) ## substract averages
#t = np.nan_to_num(t/t.std(1).reshape(t.shape[0], -1))

#%%
# To add a coordinate perturbation, do this
#pert = np.random.randn([3,*np.transpose(t).shape[0]])
#print([ncomp,t.shape[1]])
pert = np.random.randn(t.shape[1],ncomp)
epsilon =  np.matmul(t,pert)


#%%


#t = t - t.mean(1).reshape(t.shape[0], -1) ## substract averages
#t = np.nan_to_num(t/t.std(1).reshape(t.shape[0], -1))

t = embedding.fit_transform(t)
#t = t + epsilon


#%%

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

#%%

fig=pyplot.figure()
ax=Axes3D(fig)
x1 = t[0:c_1,0]
y1 = t[0:c_1,1]
z1 = t[0:c_1,2]

x2 = t[c_1:c_2, 0]
y2 = t[c_1:c_2, 1]
z2 = t[c_1:c_2, 2]

x3 = t[c_2:, 0]
y3 = t[c_2:, 1]
z3 = t[c_2:, 2]


#x4 = t[c_3:, 0]
#y4 = t[c_3:, 1]
#z4 = t[c_3:, 2]

ax.scatter(x1,y1,z1)
ax.scatter(x2,y2,z2)
ax.scatter(x3,y3,z3)
#ax.scatter(x4,y4,z4)


pyplot.show()
