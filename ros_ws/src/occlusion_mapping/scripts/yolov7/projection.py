import numpy as np
import cv2
import os
import random

import rospy
import struct
from sensor_msgs.msg import PointCloud2,PointField
import sensor_msgs.point_cloud2 as pc2

def read_mappoints(filename, scale, percentage):
    if(percentage < 0):
        percentage = 0
    if(percentage > 100):
        percentage = 100

    # Each line in file contains 3 values: x y z
    # Read the file and create a 2D Array
    points = []
    
    with open(filename, "r") as file:
        for line in file:
            x,y,z = [float(x)*scale for x in line.split()]
            points.append([x,y,z])
    
    size = int(len(points) * (percentage / 100.0))
    points = random.sample(points, size)
    points.sort(key = lambda x: x[2])
    points = np.array(points)

    return points

def read_trajectory(filename):
    # Each line contains 8 values: ID x y z qx qy qz qw
    trajectory = []

    with open(filename, "r") as file:
        for line in file:
            kf_id = int(line.split()[0])
            tx,ty,tz,qx,qy,qz,qw = [float(x) for x in line.split()[1:]]        
            trajectory.append([kf_id,np.array([tx,ty,tz]),np.array([qx,qy,qz,qw])])

    return trajectory
    
def get_rotation_vector(q):
    # Normalize quaternion
    q_norm = q / np.linalg.norm(q)

    # Compute rotation matrix from quaternion
    r_matrix = np.zeros((3, 3))
    
    r_matrix[0, 0] = 1 - 2*q_norm[2]**2 - 2*q_norm[3]**2
    r_matrix[0, 1] = 2*q_norm[1]*q_norm[2] - 2*q_norm[0]*q_norm[3]
    r_matrix[0, 2] = 2*q_norm[0]*q_norm[2] + 2*q_norm[1]*q_norm[3]
    r_matrix[1, 0] = 2*q_norm[1]*q_norm[2] + 2*q_norm[0]*q_norm[3]
    r_matrix[1, 1] = 1 - 2*q_norm[1]**2 - 2*q_norm[3]**2
    r_matrix[1, 2] = 2*q_norm[2]*q_norm[3] - 2*q_norm[0]*q_norm[1]
    r_matrix[2, 0] = 2*q_norm[1]*q_norm[3] - 2*q_norm[0]*q_norm[2]
    r_matrix[2, 1] = 2*q_norm[0]*q_norm[1] + 2*q_norm[2]*q_norm[3]
    r_matrix[2, 2] = 1 - 2*q_norm[1]**2 - 2*q_norm[2]**2
    
    r_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    r_matrix = np.matmul(r_matrix, r_z)

    r_vector, _ = cv2.Rodrigues(r_matrix)
    return r_vector

def publish_mappoints(points, mask, publisher):    
    pointcloud = PointCloud2()

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1)]
            
    data = []
    for mask_ind, (x, y, z) in enumerate(points):
        r,g,b = 0,0,0

        if(mask[mask_ind]==0): # Outside FOV
            r,g,b = 255, 0, 0
        elif(mask[mask_ind]==1 or mask[mask_ind]==2): # Inside FOV
            r,g,b = 0, 255, 0

        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
        data.append([x, y, z, rgb])
    
    pointcloud = pc2.create_cloud(rospy.Header(frame_id='map'), fields, data)
    publisher.publish(pointcloud)
    
def publish_path(poses, publisher):    
    path = Path()
    path.header.frame_id = 'map'

    for t,q in poses:
        pose = PoseStamped()
        pose.pose.position.x = t[0]
        pose.pose.position.y = t[1]
        pose.pose.position.z = t[2]
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        path.poses.append(pose)

    publisher.publish(path)
    
def invert_pose(rvec,tvec):
    # Convert rotation vector to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rvec)

    # Calculate inverse rotation matrix
    inv_rot_mat = np.transpose(rot_mat)

    # Calculate inverse translation vector
    inv_tvec = -np.dot(inv_rot_mat, tvec)

    # Print inverse rotation vector and translation vector
    inv_rvec, _ = cv2.Rodrigues(inv_rot_mat)
    return inv_rvec, inv_tvec
    
def project_points(points, rvec, tvec, K, D):    
    # Project the points onto the image plane
    projected_points, _ = cv2.projectPoints(points, rvec, tvec, K, D)

    # Check whether each point is in front of the camera view
    R, _ = cv2.Rodrigues(rvec)
    t = np.reshape(tvec, (3, 1))
    P = np.hstack((R, t))
    P = np.matmul(K, P)
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    projected_hom_points = np.matmul(P, hom_points.T).T
    valid_mask = projected_hom_points[:, 2] > 0
    
    return projected_points[:, 0, :], valid_mask.astype(int)
    
def find_features_to_project(points_3d, points_2d, mask, image):
    feature_size = 3
    done_size = 6

    features3d = [] # List of projectable features with their 3d coordinate and 2d image position- Each elements is [[3d],[2d]], i.e [[x,y,z],[u,v]]
    features2d = [] # List of projectable features with their 3d coordinate and 2d image position- Each elements is [[3d],[2d]], i.e [[x,y,z],[u,v]]
    height, width, channels = image.shape
    done_areas = np.zeros((height, width), np.uint8)

    # Filter out map points outside the image field of view, and project the rest onto the image
    for i,point in enumerate(points_2d):
        x,y = point.ravel()
        x,y = int(x), int(y)
        if(mask[i] != 1):
            continue
        if(x < done_size or x > width-done_size or y < done_size or y > height-done_size):
            continue # Remove points outside the image
        box = done_areas[y-done_size:y+done_size, x-done_size:x+done_size]
        if(np.any(box!=0)):
            continue # Already projected a feature there so skipping
        cv2.rectangle(image, (x-feature_size, y-feature_size), (x+feature_size, y+feature_size), (0, 255, 0), thickness=1)
        cv2.rectangle(done_areas, (x-done_size, y-done_size), (x+done_size, y+done_size), (255, 255, 255), thickness=-1)
        features3d.append(points_3d[i])
        features2d.append([x,y])
        mask[i] = 2
    return features3d, features2d, image, mask
