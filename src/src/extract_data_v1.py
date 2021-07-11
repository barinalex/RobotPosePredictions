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

class ScanCollector():

    def __init__(self):
        rospy.init_node('map_cuts')
        self.tag = "X2"
        self.code = "23b48289"
        self.name = "12x12_odom_cmd_v2"
        self.frame_id = 'subt' 

        #limits 
        self.xy_threshold = 6 
        self.map_scale = 60
        self.height_threshold = 0.5 
        self.poses_limit = 5
        self.cmd_limit = 50

        #temporary data
        self.shift = None
        self.coordinates = None
        self.poses_count = self.poses_limit + 1 
        self.cmd_count = self.cmd_limit + 1
        self.pointcloud_time = None
        self.R = None
        #self.pointcloud = None

        #save data
        self.map_cuts = [] 
        self.poses = []
        self.heightmaps = []
        self.cmd_buffer = []

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
        time = msg.header.stamp.to_sec()
        print("odom:",time)
        ts = self.get_baselink_tf(msg)
        tl = ts.transform.translation
        tr = ts.transform.rotation
        t = [tl.x[0], tl.y[0], tl.z[0]]
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
            #points = points[mask]
            #self.pointcloud = points 
            self.map_cuts.append(self.coordinates)
            #self.heightmaps.append(self.points_to_height_map(self.coordinates, self.map_scale))
            self.coordinates = None

        if pose is not None:
            self.poses.append(pose)

    def scan_callback(self, msg):
        print("pointcloud:",msg.header.stamp.to_sec())
        self.pointcloud_time = msg.header.stamp.to_sec()
        if self.map_cuts:
            np.savez_compressed("map_pose_cmd_"+self.tag+"_"+self.code+"_"+self.name, points=self.map_cuts,
                                robot_pose=self.poses,
                                heightmaps=self.heightmaps, 
                                cmd_vel=np.array(self.cmd_buffer))

        points = numpify(msg).ravel()
        self.coordinates = np.transpose(np.stack([points[f] for f in ('x','y','z')]))



    def points_to_height_map(self,points,scale=10):
        h = np.zeros(scale * scale) 
        xbase = -self.xy_threshold
        ybase = -self.xy_threshold
        step = self.xy_threshold*2.0/scale 
        for ym in range(scale):
            for xm in range(scale):
                x0 = xbase + step * xm
                x1 = xbase + step * (xm + 1)
                y0 = ybase + step * ym
                y1 = ybase + step * (ym + 1)
                h[(scale - ym - 1)*scale + xm] = utils.get_mean(points, x0, x1, y0, y1)
        return h


    def get_baselink_tf(self, msg):
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            try:
                t_subt_map = self.tfBuffer.lookup_transform('subt',self.tag+"/map", rospy.Time(), timeout=rospy.Duration(1))
                t_map_bl = self.tfBuffer.lookup_transform(self.tag+"/map",self.tag+"/base_link", rospy.Time(), timeout=rospy.Duration(1))
                t_bl_bfp = self.tfBuffer.lookup_transform(self.tag+"/base_link",self.tag+"/base_footprint", rospy.Time(), timeout=rospy.Duration(1))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("base except")
                rate.sleep()
                continue
            rate.sleep()
        Tsm = numpify(t_subt_map.transform)
        tsm = Tsm[:3, 3:]
        Tmb = numpify(t_map_bl.transform)
        tmb = Tmb[:3, 3:]
        Tbb = numpify(t_bl_bfp.transform)
        T = np.matmul(Tsm, np.matmul(Tmb, Tbb))
        tbb = Tbb[:3, 3:]
        t = tsm + tmb + tbb
        t = T[:3,3:]
        q = quaternion_from_matrix(T)

        #q1 = t_map_bl.transform.rotation
        #q2 = t_subt_map.transform.rotation
        #q3 = t_bl_bfp.transform.rotation
        #q = quaternion_multiply([q1.x, q1.y, q1.z, q1.w], [q2.x, q2.y, q2.z, q2.w])
        #q = quaternion_multiply([q.x, q.y, q.z, q.w], [q3.x, q3.y, q3.z, q3.w])

        ts = geometry_msgs.msg.TransformStamped()
        ts.header.stamp = msg.header.stamp
        ts.header.frame_id = "subt"
        ts.child_frame_id = self.tag+"/base_footprint"
        ts.transform.translation.x = t[0]
        ts.transform.translation.y = t[1]
        ts.transform.translation.z = t[2]
        ts.transform.rotation.x = q[0]
        ts.transform.rotation.y = q[1]
        ts.transform.rotation.z = q[2]
        ts.transform.rotation.w = q[3]
        return ts


if __name__ == '__main__':
    sc = ScanCollector()
    rospy.spin()

