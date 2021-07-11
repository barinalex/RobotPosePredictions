#!/usr/bin/env python  
import rospy
import tf_conversions
from ros_numpy import msgify, numpify
import numpy as np
import tf2_ros as tf
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_multiply
from tf.transformations import quaternion_from_matrix

tag = ""

def handle_odom_subt_tf(msg):
    br = tf.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    tfBuffer = tf.Buffer()
    tflistener = tf.TransformListener(tfBuffer)
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        try:
            t_subt_map = tfBuffer.lookup_transform('subt',tag+"/map", rospy.Time(), timeout=rospy.Duration(1))
            t_map_bl = tfBuffer.lookup_transform(tag+"/map",tag+"/base_link", rospy.Time(), timeout=rospy.Duration(1))
            t_bl_bfp = tfBuffer.lookup_transform(tag+"/base_link",tag+"/base_footprint", rospy.Time(), timeout=rospy.Duration(1))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("exception")
            rate.sleep()
            continue
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

    ts = geometry_msgs.msg.TransformStamped()
    ts.header.stamp = msg.header.stamp
    ts.header.frame_id = "subt"
    ts.child_frame_id = tag+"/base_footprint"
    ts.transform.translation.x = t[0]
    ts.transform.translation.y = t[1]
    ts.transform.translation.z = t[2]
    ts.transform.rotation.x = q[0]
    ts.transform.rotation.y = q[1]
    ts.transform.rotation.z = q[2]
    ts.transform.rotation.w = q[3]

    br.sendTransform(ts)

if __name__ == '__main__':
    tag = rospy.get_param("tag")
    rospy.init_node('tf2_subt__base_footprint_broadcaster')
    rospy.Subscriber('/robot_data/X2/odom', Odometry, handle_odom_subt_tf)
    rospy.spin()
