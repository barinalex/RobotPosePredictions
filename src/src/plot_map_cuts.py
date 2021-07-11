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
path = "/home/barinale/Documents/bachelorproject/barinale_ws/map_pose_cmd_X2_23b48289_12x12_odom_cmd_v2.npz"
if len(args)>1:
    path = args[1]

sample = 30

if len(args)>2:
    sample = int(args[2])

# more poses on one map

data = np.load(path, allow_pickle=True)
poses = data['robot_pose']
map_cuts = np.array(data['points'])
heightmaps = data['heightmaps']
#cmd = data['cmd_vel']
print(poses.shape)
print(map_cuts.shape)
print(heightmaps.shape)
#print(cmd.shape)
#print(cmd[sample*50:(sample+1)*50])

poses_by_map = poses.shape[0] / map_cuts.shape[0]

h = utils.heightmap_1d_to_2d(heightmaps[sample])

#raise NotImplementedError

points = map_cuts[sample]
x = points[:,0]
y = points[:,1]
z = points[:,2]

l_tail = [-0.1,0.1,0]
r_tail = [-0.1,-0.1,0]
l_nose = [0.1,0.1,0]
r_nose = [0.1,-0.1,0]
robot_viz = np.array([l_tail,r_tail,l_nose,r_nose])

# show points in 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z, color='k')
print(range(sample * poses_by_map,(sample + 1) * poses_by_map))
for i in range(sample * poses_by_map,(1 + sample) * poses_by_map):
    pose = poses[i,:]
    linear = pose[0:3]
    angular = pose[3:]
    R = euler_matrix(angular[0], angular[1], angular[2])[:3,:3]
    robot_pose = np.transpose(robot_viz + np.matmul(R,linear))
    robot_pose = np.matmul(R,robot_pose)
    #print(robot_pose)
    #ax.scatter(robot_pose[0,:], robot_pose[1,:], robot_pose[2,:],color='r')
    ax.scatter(linear[0], linear[1], linear[2],color='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('points')

#show hight map in 2d
plt.figure()
ax = plt.gca()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.title('z as 2d heat map')
p = plt.imshow(h, extent=[-1,1,-1,1])
plt.colorbar(p)

plt.show()

