import utils
import numpy as np

ws_path = "/home/barinale/Documents/bachelorproject/barinale_ws/data/"
filename = "JointStatesX1_2021-05-19_11_44_48_015796_4x4" 
data = np.load(ws_path + filename + '.npz', allow_pickle=True) 
base_links = data['base_links']
joint_states = data['joint_states']
map_cuts = data['points']


heightmaps = []
for points in map_cuts:
    heightmap = utils.points_to_height_map(points, -2, -2, 2*2.0/10, 10)
    heightmaps.append(heightmap)

np.savez_compressed(ws_path+filename+"heightmap", points=map_cuts,
                    base_links=base_links,
                    joint_states=joint_states,
                    heightmaps=heightmaps
                    )

