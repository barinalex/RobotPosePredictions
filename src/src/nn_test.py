# -*- coding: utf-8 -*-
"""
    ML04-1.py
    ~~~~~~~~~

    Model evaluation and diagnostics
    B(E)3M3UI - Artificial Intelligence

    :author: Petr Posik, Jiri Spilka, 2019

    FEE CTU in Prague
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn import linear_model, pipeline
import math


def get_rid_of_nan(h):
    for y in range(h.shape[0]):
        for x in range(h.shape[1]):
            if h[y][x] != h[y][x]:
                h[y][x] = 0
    return h

def get_base(y,p):
    if y*p < 0:
        return 2 * abs(y) + abs(p) 
    return max(abs(y), abs(p))

def evaluate(model,X,y):
    print("X shape:",X.shape)
    print("y shape:",y.shape)
    pred = model.predict(X)
    err = y - pred
    max_value = np.maximum(np.abs(y), np.abs(pred))
    base_value = np.zeros(y.shape)
    for i, y_, p_ in zip(range(y.shape[0]), y, pred):
        roll = get_base(y_[0],p_[0])
        pitch = get_base(y_[1],p_[1])
        base_value[i] = [roll, pitch]

    rel_err = np.abs(err/base_value)
    rel_err_mean = np.mean(rel_err, axis=0)
    print("rel_err_mean:",rel_err_mean) # mean(err / max(abs(y),abs(y_pred)))
    err_mean = np.mean(np.abs(err),axis=0)
    value_mean = np.mean(np.abs(y),axis=0)
    print("errors mean:",np.round(np.rad2deg(err_mean), decimals=2))
    print("values mean:",np.round(np.rad2deg(value_mean), decimals=2))
    print("relative err:",err_mean/value_mean)
    print(regr.score(X, y))
    return err, pred

ws_path = "/home/barinale/Documents/bachelorproject/barinale_ws/"

# Load data
data = np.load(ws_path+'src/map_cuts/src/heighmaps.npz', allow_pickle=True)
X = data['heightmaps']
y = data['roll_pitch']
#roll = y[0,:]
#pitch = y[1,:]

data = np.load(ws_path+'map_pose_cmd_X1_04_01_2x2_centered_base_footprint.npz', allow_pickle=True)
X = (data['heightmaps'])
X = get_rid_of_nan(X)
y = (data['robot_pose'])[:,3:5]

print('\nThe dataset (X and y) size:')
print('X: ', X.shape)
print('y: ', y.shape)

Xtr,Xts,ytr,yts = train_test_split(X,y, train_size=0.66)

regr = MLPRegressor(max_iter=100, learning_rate_init=0.01, batch_size=8)
regr.fit(Xtr,ytr)
print("\npredictions on the same data set (test set)")
evaluate(regr,Xtr,ytr)


print("\npredictions on the same data set (test set)")
evaluate(regr,Xts,yts)

dataX1 = np.load(ws_path+'map_pose_cmd_X2_23b48289_2x2_centered_base_footprint.npz', allow_pickle=True)
X1 = (dataX1['heightmaps'])
X1 = get_rid_of_nan(X1)
y1 = (dataX1['robot_pose'])[:,3:5]
print("\ndata set from different run")
evaluate(regr, X1, y1)

raise NotImplementedError
angle_threshold = 2
y1_deg = np.round(np.rad2deg(y1), decimals=2)
mask = [abs(y_[0]) > angle_threshold and abs(y_[1]) > angle_threshold for y_ in y1_deg]
X2 = X1[mask]
y2 = y1[mask]
print("\ndata set from different run with roll and pitch >",angle_threshold,"degrees")
evaluate(regr, X2, y2)

mask = [abs(y_[0]) < angle_threshold and abs(y_[1]) < angle_threshold for y_ in y1_deg]
X3 = X1[mask]
y3 = y1[mask]
print("\ndata set from different run with roll and pitch <",angle_threshold,"degrees")
evaluate(regr, X3, y3)

