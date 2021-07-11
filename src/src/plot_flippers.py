import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import euler_matrix
import utils
import sys


args = sys.argv
print(args)
path = "/home/barinale/Documents/bachelorproject/barinale_ws/data/flippersX1_2021-05-14_00_13_21_180197_4x4.npz"

sample = 4

if len(args)>1:
    sample = int(args[1])

if len(args)>2:
    path = args[2]

# more poses on one map

data = np.load(path, allow_pickle=True)
base_links = data['base_links']
front_lefts = data['front_lefts']
front_rights = data['front_rights']
rear_lefts = data['rear_lefts']
rear_rights = data['rear_rights']
map_cuts = data['points']
#heightmaps = data['heightmaps']
#print(heightmaps.shape)
#h = utils.heightmap_1d_to_2d(heightmaps[sample])



#raise NotImplementedError

print(map_cuts.shape)
points = map_cuts[sample]
x = points[:,0]
y = points[:,1]
z = points[:,2]

# show points in 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z, color='k')

linear = base_links[sample][0:3]
angular = base_links[sample][3:]
ax.scatter(linear[0], linear[1], linear[2],color='r')

linear = front_lefts[sample][0:3]
ax.scatter(linear[0], linear[1], linear[2],color='b')
linear = front_rights[sample][0:3]
ax.scatter(linear[0], linear[1], linear[2],color='b')
linear = rear_lefts[sample][0:3]
ax.scatter(linear[0], linear[1], linear[2],color='r')
linear = rear_rights[sample][0:3]
ax.scatter(linear[0], linear[1], linear[2],color='r')




ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('points')

'''
#show hight map in 2d
plt.figure()
ax = plt.gca()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.title('z as 2d heat map')
p = plt.imshow(h, extent=[-2,2,-2,2])
plt.colorbar(p)
'''

plt.show()

