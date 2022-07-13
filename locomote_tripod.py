#!/usr/bin/env python
# coding: utf-8

import pybullet as p
import pybullet_data
import os
import numpy as np
import time

physicsClient = p.connect(p.GUI)

def wrap_to_pi(angle):
    return np.remainder(angle + np.pi, np.pi * 2) - np.pi


pi = np.pi
## Initialize the environment
urdf_file = 'urdf/llllll.urdf'
if not (os.path.exists(urdf_file)):
    # in_name = os.path.join(cwd,  folder, name + '.xacro')
    in_name = 'urdf/llllll.xacro'
    os.system('rosrun xacro xacro --inorder --xacro-ns '
              + in_name + ' > ' + urdf_file)

p.resetSimulation()  # remove all objects from the world and reset the world to initial conditions. (not needed here but kept for example)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physicsClient)
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(2, 0, -25, [0, 0, 0], physicsClientId=physicsClient)  # I like this view
planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),
                                  "plane100.urdf"))
startPosition = [0, 0, 0.3]  # high enough that nothing touches the ground
# startPosition=[0,0,1] # high enough that nothing touches the ground
startOrientationRPY = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler(startOrientationRPY)
robotID = p.loadURDF(
    urdf_file,
    basePosition=startPosition, baseOrientation=startOrientation,
    flags=(p.URDF_MAINTAIN_LINK_ORDER
           | p.URDF_USE_SELF_COLLISION
           | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES))

# ## Add a box for a test for the collisions
# box_id = p.createCollisionShape(
#     p.GEOM_BOX,
#     halfExtents=[.05, .05, 0.1])
# visual_id = p.createVisualShape(
#     p.GEOM_BOX,
#     rgbaColor = [0, 1, 0, 1],
#     halfExtents=[.05, .05, 0.1])
# block_ID = p.createMultiBody(
#     baseMass=0,
#     baseCollisionShapeIndex=box_id,
#     baseVisualShapeIndex = visual_id,
#     basePosition=[0.1, 0., 0.])
# # do not need to compute collisions between these
# p.setCollisionFilterPair(planeId, block_ID, -1, -1, 0)
# # Each body is part of a group. It collides with other bodies if their group
# # matches the mask, and vise versa.
# collisionFilterGroup = 1
# collisionFilterMask = 1
# for i in range(-1,num_joints_total):
#     p.setCollisionFilterGroupMask(robotID, i,collisionFilterGroup,collisionFilterMask)
# rayFromPositions = [[0,0,3], [0.1,0,3], [0,0.1,3]]
# rayToPositions = [[0,0,-3], [0.1,0,-3], [0,0.1,-3]]
# out = p.rayTestBatch(rayFromPositions=rayFromPositions,
#     rayToPositions=rayToPositions,
#     collisionFilterMask=0)
#     # collisionFilterMask=2)
# # only test hits if the bitwise and between
# # collisionFilterMask and body collision filter group is
# # non-zero. See setCollisionFilterGroupMask on how
# # to modify the body filter mask/group.
# print('Ray hit:')
# print([out[i][3][-1] for i in range(len(rayFromPositions))])
# print([out[i][2] for i in range(len(rayFromPositions))])
## Get some settings from the robot
# count all joints, including fixed ones
num_joints_total = p.getNumJoints(robotID,
                                  physicsClientId=physicsClient)
moving_joint_names = []
moving_joint_inds = []
moving_joint_types = []
moving_joint_limits = []
moving_joint_centers = []
moving_joint_max_torques = []
moving_joint_max_velocities = []
for j_ind in range(num_joints_total):
    j_info = p.getJointInfo(robotID,
                            j_ind)
    if j_info[2] != (p.JOINT_FIXED):
        moving_joint_inds.append(j_ind)
        moving_joint_names.append(j_info[1])
        moving_joint_types.append(j_info[2])
        j_limits = [j_info[8], j_info[9]]
        j_center = (j_info[8] + j_info[9]) / 2
        if j_limits[1] <= j_limits[0]:
            j_limits = [-np.inf, np.inf]
            j_center = 0
        moving_joint_limits.append(j_limits)
        moving_joint_centers.append(j_center)
        moving_joint_max_torques.append(j_info[10])
        moving_joint_max_velocities.append(j_info[11])
moving_joint_centers = np.array(moving_joint_centers)
print('moving_joint_types: ' + str(moving_joint_types))
print(moving_joint_centers)
# reset to the joint center from the urdf (may not be zeros)
num_joints = len(moving_joint_names)
for i in range(num_joints):
    center = moving_joint_centers[i]
    jind = moving_joint_inds[i]
    p.resetJointState(bodyUniqueId=robotID,
                      jointIndex=jind,
                      targetValue=center,
                      targetVelocity=0)
# parameters for alternating tripod
amp_max = pi / 12
amps = amp_max * np.ones([3, 6])
# amps[1,:] = amps[1,:] + pi/8 # boost step height
# amps[2,:] = amps[1,:] + pi/8 # boost step height
period = 1
const_offsets = np.array([[1, 0, -1, 1, 0, -1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]) * np.pi / 8  # offset front and leg base angles a little
phase_offsets = np.array([
    [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
    [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]) * np.pi
dt = 1. / 240.  # default time step
# command the robot forward
forward_cmd = 1
turn_cmd = 0
step = 0

vid_path = ("./test.mp4")

# if not os.path.exists(vid_path):
logID = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
fileName=vid_path)

while True:
    linkWorldPosition, linkWorldOrientationQuat = p.getBasePositionAndOrientation(
        bodyUniqueId=robotID, physicsClientId=physicsClient)
    linkWorldOrientation = p.getEulerFromQuaternion(linkWorldOrientationQuat)

    t = step * dt
    # # command the robot forward and turn
    # if t > period * 2:
    #     turn_cmd = 1
    #     forward_cmd = 1
    #
    # # command the robot to turn
    # if t > period * 3:
    #     turn_cmd = 1
    #     forward_cmd = 0
    #
    # # Proportional control over yaw
    # if t > 4 * period:
    #     turn_cmd = -linkWorldOrientation[2] / (pi / 2)
    #     forward_cmd = 1

    # cap at 1
    turn_cmd = min(turn_cmd, 1)
    forward_cmd = min(forward_cmd, 1)

    # only advance phase if command is positve
    if (np.abs(forward_cmd) + np.abs(turn_cmd)) > 0:
        step += 1

    # Create alt tripod leg angles
    joint_pos_command = []
    amps[0, 0:3] = amp_max * np.ones(3) * (-forward_cmd + turn_cmd)
    amps[0, 3:6] = amp_max * np.ones(3) * (forward_cmd + turn_cmd)
    amps[0, :] = np.clip(amps[0, :], -amp_max, amp_max)
    for i in range(6):
        leg_angles_i = amps[:, i] * np.sin(t * 2 * pi / period - phase_offsets[:, i])
        leg_angles_i[1:] = np.clip(leg_angles_i[2:], 0, np.inf)  # convert up-down motion to up-flat motion
        leg_angles_i += const_offsets[:, i]
        joint_pos_command.append(leg_angles_i)
    joint_pos_command = np.concatenate(joint_pos_command) + moving_joint_centers
    # Send commands to joints position control
    p.setJointMotorControlArray(
        bodyUniqueId=robotID,
        jointIndices=moving_joint_inds,
        controlMode=p.POSITION_CONTROL,
        targetPositions=joint_pos_command,
        forces=moving_joint_max_torques)
    print(joint_pos_command)
    p.stepSimulation()

    ## Misc collision detection stuff
    # out = p.rayTestBatch(rayFromPositions=rayFromPositions,
    #     rayToPositions=rayToPositions,
    #     collisionFilterMask=2)
    # # only test hits if the bitwise and between
    # # collisionFilterMask and body collision filter group is
    # # non-zero. See setCollisionFilterGroupMask on how
    # # to modify the body filter mask/group.
    # # print('Ray hit:')
    # # print([out[i][3][-1] for i in range(len(rayFromPositions))])
    # # print([out[i][2] for i in range(len(rayFromPositions))])

    time.sleep(1. / 240)
