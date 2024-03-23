#!/usr/bin/python3
import rospy, numpy as np
import rospkg

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32

import time

import jax.numpy as jnp

# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

class my_visualisation:
    def __init__(self):
        self.freq = 200

        ## Publish reference path
            
        self.publish_ref_path = rospy.Publisher('/ref_array', MarkerArray, queue_size=1)
        self.publish_data_path = rospy.Publisher('/data_array', MarkerArray, queue_size=1)

    def dip_pos_callback(self, data):
        self.dip_pos = jnp.array([data.x, data.y, data.z]).reshape((-1,1))

    def dirt_pos_callback(self, data):
        self.dirt_pos = jnp.array([data.x, data.y, data.z]).reshape((-1,1))

    def control_t_callback(self, data):
        self.control_t = int(data.data)


def create_marker_arr(traj, dirt_pos, color_a):
    markerArray = MarkerArray()
    for i in range(traj.shape[0]):

        marker_i = Marker()
        marker_i.header.frame_id = "panda_link0"
        marker_i.id = i
        marker_i.type = marker_i.SPHERE
        marker_i.action = marker_i.ADD
        marker_i.scale.x = 0.01
        marker_i.scale.y = 0.01
        marker_i.scale.z = 0.01
        marker_i.color.a = color_a
        marker_i.color.r = 0.0
        marker_i.color.g = 1.0
        marker_i.color.b = 1.0
        marker_i.pose.orientation.w = 1.0
        marker_i.pose.position.x = traj[i, 0]
        marker_i.pose.position.y = traj[i, 1]
        marker_i.pose.position.z = traj[i, 2]

        markerArray.markers.append(marker_i)
    
    return markerArray

def main():

    ## Initialize object

    visualize_obj = my_visualisation()

    ## Load data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    data_file = curr_path + '/config/Data_Trajs/trajectory_data_scooping_orientation_orig_1.npy'
    with open(data_file, 'rb') as f:
        traj_all_combine_process = jnp.load(f) # nD x nsamples x 2*dim
        ts_new = jnp.load(f)
        scaler_all_t_combine = jnp.load(f) # 1/T (end time) nD x ntrajs (split) x 1

    ref_file = curr_path + '/config/Data_Trajs/trajectory_ref.npy'
    with open(ref_file, 'rb') as f:
        traj_ref = jnp.load(f)
        # ts = jnp.load(f)
        # scaler_all_t = jnp.load(f)

    train_indx = 0
    split_indx = 0

    # traj_load = traj_all_combine_process[train_indx, split_indx]
    traj_load = traj_all_combine_process[train_indx]
    
    ## Initialize node

    rospy.init_node('my_visuals', anonymous=True)
    rate = rospy.Rate(visualize_obj.freq) # Hz

    markerArray = MarkerArray()
    markerArray_ref = MarkerArray()

    i = 0

    for i in range(traj_load.shape[0]):

        ## Demonstration

        marker_i = Marker()
        marker_i.header.frame_id = "panda_link0"
        marker_i.id = i
        marker_i.type = marker_i.SPHERE
        marker_i.action = marker_i.ADD
        marker_i.scale.x = 0.01
        marker_i.scale.y = 0.01
        marker_i.scale.z = 0.01
        marker_i.color.a = 1.0
        marker_i.color.r = 0.0
        marker_i.color.g = 0.0
        marker_i.color.b = 1.0
        marker_i.pose.orientation.w = 1.0
        marker_i.pose.position.x = traj_load[i, 0]
        marker_i.pose.position.y = traj_load[i, 1]
        marker_i.pose.position.z = traj_load[i, 2]

        markerArray.markers.append(marker_i)

        ## Reference/target path

        marker_r_i = Marker()
        marker_r_i.header.frame_id = "panda_link0"
        marker_r_i.id = i
        marker_r_i.type = marker_i.SPHERE
        marker_r_i.action = marker_i.ADD
        marker_r_i.scale.x = 0.01
        marker_r_i.scale.y = 0.01
        marker_r_i.scale.z = 0.01
        marker_r_i.color.a = 1.0
        marker_r_i.color.r = 1.0
        marker_r_i.color.g = 0.0
        marker_r_i.color.b = 1.0
        marker_r_i.pose.orientation.w = 1.0
        marker_r_i.pose.position.x = traj_ref[i, 0]
        marker_r_i.pose.position.y = traj_ref[i, 1]
        marker_r_i.pose.position.z = traj_ref[i, 2]

        markerArray_ref.markers.append(marker_r_i)    

    while not rospy.is_shutdown():

        visualize_obj.publish_data_path.publish(markerArray)
        visualize_obj.publish_ref_path.publish(markerArray_ref)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass