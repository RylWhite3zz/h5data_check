#coding=utf-8
import os
import sys
import cv2
import copy
import h5py
import rospy
import argparse
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import Header
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist


def array_to_stamped_pose(ee, array):
    ee.x, ee.y, ee.z, ee.roll, ee.pitch, ee.yaw, ee.gripper = array[:7]

def ee_control(args):
    rospy.init_node("rospy_publish_test")

    master_pose_left_publisher = rospy.Publisher(args.master_pose_left_topic, PosCmd, queue_size=10)
    master_pose_right_publisher = rospy.Publisher(args.master_pose_right_topic, PosCmd, queue_size=10)

    ee_state_msg = PosCmd()

    action0 = [0.029966, -0.046755, 0.216642, -0.1490685714128357, 1.4796552332557529, -1.1503116034044227, 0.0009] # leftmost
    action1 = [0.051167, 0.021585, 0.216642, -0.15048228810695108, 1.4796377799632328, 0.24829053938871334, 0.0009] # rightmost 
    last_action = action0 + action0  # reference action
    # counter-oclock movement from action 0 to action 1
    N = 20 # coarse actions
    x = np.linspace(action0[0], action1[0], N)
    y = np.linspace(action0[1], action1[1], N)
    yaw = np.linspace(action0[5], action1[5], N)

    actions = [] # construct coarse actions
    for x_, y_, yaw_ in zip(x, y, yaw):
        actions.append(copy.deepcopy(last_action))
        actions[-1][0] = actions[-1][7] = x_
        actions[-1][1] = actions[-1][8] = y_
        actions[-1][5] = actions[-1][12] = yaw_

    rate = rospy.Rate(100)
    for action in actions:
        if(rospy.is_shutdown()):
                break 
        new_actions = np.linspace(last_action, action, 20) # 插值
        last_action = action
        for act in new_actions:
            print(np.round(act[:7], 4))

            array_to_stamped_pose(ee_state_msg, act[:7])
            master_pose_left_publisher.publish(ee_state_msg)

            array_to_stamped_pose(ee_state_msg, act[7:])
            master_pose_right_publisher.publish(ee_state_msg)   

            if(rospy.is_shutdown()):
                break
            rate.sleep() 


def main(args):
    rospy.init_node("rospy_publish_test")

    master_arm_left_publisher = rospy.Publisher(args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(args.master_arm_right_topic, JointState, queue_size=10)

    joint_state_msg = JointState()
    joint_state_msg.header =  Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    # last_action = [-0.0057,-0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    # last_action = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    last_action = [-0.0057,-0.031, -0.0122, -0.032, 0.0099, 0.0179, -0.001, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, -0.001] #0.2279]

    actions = [] # construct coarse actions
    for joint_0 in np.arange(-1., .5, 0.1):
        actions.append(copy.deepcopy(last_action))
        actions[-1][0] = actions[-1][7] = joint_0 

    #actions = [last_action] # debug: joint state <-> ee euler

    rate = rospy.Rate(100)
    for action in actions[:]:
        if(rospy.is_shutdown()):
                break 
        new_actions = np.linspace(last_action, action, 20) # 插值
        last_action = action
        for act in new_actions:
            print(np.round(act[:7], 4))
            cur_timestamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.header.stamp = cur_timestamp 
            
            joint_state_msg.position = act[:7]
            master_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = act[7:]
            master_arm_right_publisher.publish(joint_state_msg)   

            if(rospy.is_shutdown()):
                break
            rate.sleep() 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_ee_control', action='store_true', help='with_ee_control',required=False)
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--master_pose_left_topic', action='store', type=str, help='master_pose_left_topic',
                        default='/euler_left', required=False)
    parser.add_argument('--master_pose_right_topic', action='store', type=str, help='master_pose_right_topic',
                        default='/euler_right', required=False)
#    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
#                        default='/puppet/joint_left', required=False)
#    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
#                        default='/puppet/joint_right', required=False)

    args = parser.parse_args()
    if args.with_ee_control:
        ee_control(args)
    else:
        main(args)
    # python collect_data.py --max_timesteps 500 --is_compress --episode_idx 0 
