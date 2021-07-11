#!/usr/bin/env python2
import rospy
from ros_numpy import msgify, numpify
import numpy as np
import math
import matplotlib.pyplot as plt
import geometry_msgs.msg
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import tf2_ros as tf
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from tf.transformations import quaternion_from_matrix
from tf.transformations import quaternion_multiply 
from tf.transformations import euler_matrix
from tf.transformations import rotation_matrix
import utils
import message_filters
from datetime import datetime

class ScanCollector():

    def __init__(self):
        rospy.init_node('map_cuts')

        #limits 
        self.xy_threshold = 6 
        self.map_scale = self.xy_threshold * 10
        self.height_threshold = 0.5 
        self.poses_limit = 5
        self.cmd_limit = self.poses_limit * 10

        #strings
        self.data_path = rospy.get_param("data_path", "/home/barinale/Documents/bachelorproject/barinale_ws/data")
        self.tag = rospy.get_param("tag")
        datetime_stamp = (str(datetime.now())).replace(' ','_').replace(':','_').replace('.','_')
        xy_str = str(self.xy_threshold)
        self.name = "map_pose_cmd_" + self.tag + "_" + datetime_stamp + "_" + xy_str + "x" + xy_str 
        self.frame_id = 'subt' 

        #temporary data
        self.shift = None
        self.coordinates = None
        self.poses_count = self.poses_limit + 1 
        self.cmd_count = self.cmd_limit + 1
        self.R = None
        #self.pointcloud = None

        #buffers
        self.map_cuts = [] 
        self.poses = []
        self.cmd_buffer = []

        #pubs subs
        self.tfBuffer = tf.Buffer()
        self.tflistener = tf.TransformListener(self.tfBuffer)
        self.maplistener = rospy.Subscriber("/robot_data/"+self.tag+"/local_map_slow", PointCloud2, self.scan_callback)
        self.publisher = rospy.Publisher('map_cuts', PointCloud2, queue_size=10)
        self.odomlistener = rospy.Subscriber('/robot_data/'+self.tag+'/odom', Odometry, self.odom_callback)
        self.cmdlistener = rospy.Subscriber('/robot_data/'+self.tag+'/cmd_vel', Twist, self.cmd_vel_callback)

        '''
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.pointcloud is not None:
                msg = msgify(PointCloud2, self.pointcloud)
                msg.header.frame_id = self.frame_id
                self.publisher.publish(msg)
            rate.sleep()
        '''


    def cmd_vel_callback(self, msg):
        if self.cmd_count < self.cmd_limit:
            lin = msg.linear
            ang = msg.angular
            t = [lin.x, lin.y, lin.z]
            r = [ang.x, ang.y, ang.z]
            v =  np.hstack((t,r))
            self.cmd_buffer.append(v)
            self.cmd_count += 1

    def odom_callback(self, msg):
        #time = msg.header.stamp.to_sec()
        ts = self.get_base_footprint_tf(msg)
        if ts is None:
            print("transformation not found")
            return
        tl = ts.transform.translation
        tr = ts.transform.rotation
        t = [tl.x, tl.y, tl.z]
        r = euler_from_quaternion((tr.x,tr.y,tr.z,tr.w))
        pose = None

        if self.poses_count < self.poses_limit:
            xyz = t - self.shift[:3]
            xyz = np.matmul(self.R,xyz)
            pose = np.hstack((xyz,r-self.shift[3:]))
            self.poses_count += 1

        if self.coordinates is not None:
            self.poses_count = 0
            self.cmd_count = 0
            self.shift = np.hstack((t,[0,0,r[2]]))

            self.R = euler_matrix(0,0,-r[2])[:3,:3]
            pose = np.hstack(([0,0,0],[r[0],r[1],0]))
            mask = np.zeros(self.coordinates.shape[0], dtype=bool)

            for i in range(len(self.coordinates)):
                xyz = self.coordinates[i] - t
                xyz = np.matmul(self.R,xyz)
                self.coordinates[i] = xyz
                mask[i] = abs(xyz[0]) <= self.xy_threshold and abs(xyz[1]) <= self.xy_threshold and abs(xyz[2]) <= self.height_threshold
            self.coordinates = self.coordinates[mask]
            self.map_cuts.append(self.coordinates)
            #points = points[mask]
            #self.pointcloud = points 
            self.coordinates = None

        if pose is not None:
            self.poses.append(pose)


    def scan_callback(self, msg):
        if self.map_cuts:
            np.savez_compressed(self.data_path + self.name, points=self.map_cuts,
                                robot_pose=self.poses,
                                cmd_vel=np.array(self.cmd_buffer))

        points = numpify(msg).ravel()
        self.coordinates = np.transpose(np.stack([points[f] for f in ('x','y','z')]))


    def get_base_footprint_tf(self, msg):
        rate = rospy.Rate(1.0)
        error_limit = 5 
        while not rospy.is_shutdown() and error_limit > 0:
            try:
                t = self.tfBuffer.lookup_transform('subt',self.tag+"/base_footprint", msg.header.stamp, timeout=rospy.Duration(1))
                return t
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("exception")
                error_limit -= 1
                rate.sleep()
                continue
        return None 


if __name__ == '__main__':
    sc = ScanCollector()
    rospy.spin()

