from context import modest as md
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

mybkg = md.signals.UniformNoiseXRaySource(1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

myVecs=mybkg.generateUniformArrivalVectors(1000, FOV=np.pi)
ax.scatter(myVecs[:,0], myVecs[:,1], myVecs[:,2], color='red', marker='.')

# myVecs=mybkg.generateUniformArrivalVectors(100, FOV=np.pi/2)
# ax.scatter(myVecs[:,0], myVecs[:,1], myVecs[:,2], color='blue')

# myVecs=mybkg.generateUniformArrivalVectors(100, FOV=np.pi/4)
# ax.scatter(myVecs[:,0], myVecs[:,1], myVecs[:,2], color='green')
plt.show(block=False)
