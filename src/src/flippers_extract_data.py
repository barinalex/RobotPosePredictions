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
from sensor_msgs.msg import JointState
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
        self.xy_threshold = 2 
        self.map_scale = self.xy_threshold * 10
        self.height_threshold = 0.5 
        self.poses_limit = 5
        self.cmd_limit = self.poses_limit * 10

        #strings
        self.data_path = rospy.get_param("data_path", "/home/barinale/Documents/bachelorproject/barinale_ws/data")
        self.tag = rospy.get_param("tag", "X1")
        datetime_stamp = (str(datetime.now())).replace(' ','_').replace(':','_').replace('.','_')
        xy_str = str(self.xy_threshold * 2)
        self.name = "JointStates" + self.tag + "_" + datetime_stamp + "_" + xy_str + "x" + xy_str 

        #temporary data
        self.shift = None
        self.coordinates = None
        self.poses_count = self.poses_limit + 1 
        self.cmd_count = self.cmd_limit + 1
        self.R = None
        #self.pointcloud = None
        self.last_joint_time = 0

        #buffers
        self.map_cuts = [] 
        self.base_links = []
        self.joint_states = []
        self.cmd_buffer = []

        '''
        self.front_lefts = []
        self.front_rights = []
        self.rear_lefts = []
        self.rear_rights = []
        '''

        #pubs subs
        self.tfBuffer = tf.Buffer()
        self.tflistener = tf.TransformListener(self.tfBuffer)
        #self.publisher = rospy.Publisher('map_cuts', PointCloud2, queue_size=10)
        #self.odomlistener = rospy.Subscriber('/'+self.tag+'/odom', Odometry, self.odom_callback)
        #self.cmdlistener = rospy.Subscriber('/'+self.tag+'/cmd_vel', Twist, self.cmd_vel_callback)
        #self.maplistener = rospy.Subscriber('/laser_cloud_map', PointCloud2, self.cloud_callback)
        self.flipperslistener = rospy.Subscriber('/'+self.tag+'/joint_state', JointState, self.joint_callback)

        self.map_sync_listener = message_filters.Subscriber('/laser_cloud_map', PointCloud2)
        self.flippers_sync_listener = message_filters.Subscriber('/'+self.tag+'/joint_state', JointState)


        self.ts = message_filters.ApproximateTimeSynchronizer([self.flippers_sync_listener,  self.map_sync_listener], 10, 0.5)
        self.ts.registerCallback(self.callback)


    def callback(self, joint_state_msg, point_cloud_msg):
        #print("joint:",joint_state_msg.header)
        print("cloud:",point_cloud_msg.header)
        self.extract_map_cut_and_pose(point_cloud_msg)
        self.extract_joint_state(joint_state_msg)
        #self.save_data()


    def joint_callback(self, msg):
        if not self.map_cuts:
            return
        if msg.header.stamp.to_sec() - self.last_joint_time >= 0.4:
            print("joint:",msg.header.stamp.to_sec())
            self.last_joint_time = msg.header.stamp.to_sec()


    def cmd_vel_callback(self, msg):
        #print("cmd:")
        if self.cmd_count < self.cmd_limit:
            lin = msg.linear
            ang = msg.angular
            t = [lin.x, lin.y, lin.z]
            r = [ang.x, ang.y, ang.z]
            v =  np.hstack((t,r))
            self.cmd_buffer.append(v)
            self.cmd_count += 1


    def cloud_callback(self, msg):
        print("POINTCLOUD:",msg.header.stamp.to_sec())
        print(msg.header)
        self.extract_map_cut_and_pose(msg)
        #self.save_data()


    def extract_joint_state(self, msg):
        self.joint_states.append(msg.position)


    def extract_map_cut_and_pose(self, msg):
        points = numpify(msg).ravel()
        P = np.transpose(np.stack([points[f] for f in ('x','y','z')]))
        t_base = self.get_tf(msg.header.stamp, "/base_link")
        '''
        t_flep = self.get_tf(msg.header.stamp, "/front_left_flipper_end_point")
        t_frep = self.get_tf(msg.header.stamp, "/front_right_flipper_end_point")
        t_rlep = self.get_tf(msg.header.stamp, "/rear_left_flipper_end_point")
        t_rrep = self.get_tf(msg.header.stamp, "/rear_right_flipper_end_point")
        '''

        lin = t_base.transform.translation
        q = t_base.transform.rotation
        t = np.array([lin.x, lin.y, lin.z])
        r = euler_from_quaternion((q.x,q.y,q.z,q.w))

        shift = np.hstack((t,[0,0,r[2]]))
        R = euler_matrix(0,0,-r[2])[:3,:3]
        pose = np.hstack(([0,0,0],[r[0],r[1],0]))
        T = np.column_stack((R,-t))

        N = P.shape[0]
        P = np.transpose(P)
        P = np.vstack((P,np.ones(N))) 
        P = np.matmul(T,P)
        P = np.transpose(P)

        mask = np.zeros(N, dtype=bool)
        for i in range(N):
            p = P[i]
            mask[i] = abs(p[0]) <= self.xy_threshold and abs(p[1]) <= self.xy_threshold and abs(p[2]) <= self.height_threshold

        P = P[mask]
        self.map_cuts.append(P)

        self.base_links.append(pose)
        '''
        self.front_lefts.append(self.shift_frame(t_flep, R, shift))
        self.front_rights.append(self.shift_frame(t_frep, R, shift))
        self.rear_lefts.append(self.shift_frame(t_rlep, R, shift))
        self.rear_rights.append(self.shift_frame(t_rrep, R, shift))
        '''


    def save_data(self):
        if self.map_cuts:
            np.savez_compressed(self.data_path + self.name, points=np.array(self.map_cuts),
                                base_links=self.base_links,
                                joint_states=self.joint_states
                                #front_lefts=self.front_lefts,
                                #front_rights=self.front_rights,
                                #rear_lefts=self.rear_lefts,
                                #rear_rights=self.rear_rights
                                )


    def shift_frame(self, tf, R, shift):
        lin = tf.transform.translation
        q = tf.transform.rotation
        t = [lin.x, lin.y, lin.z]
        r = euler_from_quaternion((q.x,q.y,q.z,q.w))

        xyz = np.matmul(R, t - shift[:3])
        pose = np.hstack((xyz, r - shift[3:]))
        return pose


    def get_tf(self, stamp, frame):
        error_limit = 5 
        while not rospy.is_shutdown() and error_limit > 0:
            try:
                t = self.tfBuffer.lookup_transform('camera_init',self.tag+frame, stamp, timeout=rospy.Duration(2))
                return t
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("exception")
                error_limit -= 1
                continue
        return None 


    def odom_callback(self, msg):
        #print("odom:",msg.header.stamp.to_sec())
        '''
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
        '''



if __name__ == '__main__':
    sc = ScanCollector()
    rospy.spin()

