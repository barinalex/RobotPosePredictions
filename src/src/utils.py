import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import euler_matrix


def get_mean(points,x0,x1,y0,y1):
    cut = []
    mean = None
    for p in points:
        if p[0] <= x1 and p[0] >= x0 and p[1] <= y1 and p[1] >= y0:
            cut.append(p) 
    cut = np.array(cut)
    cut = np.transpose(cut)
    if cut.size != 0:
        mean = np.mean(cut[2,:])
    return mean

def get_mean_around_index(array, x, y, scale):
    array = np.array(array)
    mask = [-1,0,1]
    data = []
    mean = 0
    for yshift in mask:
        for xshift in mask:
            if xshift == 0 and yshift == 0:
                continue
            y_ = y + yshift
            x_ = x + xshift
            if y_ < 0 or y_ >= scale:
                continue
            if x_ < 0 or x_ >= scale:
                continue
            value = array[y_*scale + x_]
            if value == value:
                data.append(value)
    if data:
        mean = np.mean(np.array(data))
    return mean

def get_rid_of_nan(h):
    scale = int(math.sqrt(h.size))
    for y in range(scale):
        for x in range(scale):
            if h[y * scale + x] != h[y*scale + x]:
                h[y*scale + x] = get_mean_around_index(h, x, y, scale)
    return h



def points_to_height_map(points,xbase,ybase,step,scale=10):
    h = np.zeros(scale * scale) 
    for ym in range(scale):
        for xm in range(scale):
            x0 = xbase + step * xm
            x1 = xbase + step * (xm + 1)
            y0 = ybase + step * ym
            y1 = ybase + step * (ym + 1)
            h[(scale - ym - 1)*scale + xm] = get_mean(points, x0, x1, y0, y1)
    return h

def maps_to_heightmaps(maps):
    heightmaps = []
    for m in maps:
        hm = points_to_height_map(m)
        heightmaps.append(hm)
    heightmaps = np.array(heightmaps)
    return heightmaps

def heightmap_1d_to_2d(heightmap):
    scale = int(math.sqrt(heightmap.size))
    return np.reshape(heightmap, (scale,scale))
    '''
    h = np.zeros((scale,scale))
    for ym in range(scale):
        for xm in range(scale):
            h[scale - ym - 1][xm] = heightmap[(scale - ym - 1)*scale + xm]
    return h
    '''

