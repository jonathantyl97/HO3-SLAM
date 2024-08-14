import numpy as np
import cv2
import argparse

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from detectron2.utils.visualizer import ColorMode
import torch
import random

import warnings
warnings.filterwarnings("ignore")

import rospy
import struct
import projection
from sensor_msgs.msg import PointCloud2,PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point as RosPoint
from occlusion_mapping.msg import TrackedFrame
from visualization_msgs.msg import Marker
import math 

import matplotlib.pyplot as plt
from math import ceil
from scipy.spatial.transform import Rotation

from deep_sort import DeepSort
from deepsort_util import draw_bboxes
import sys
sys.path.append('/usr/local/python')
from openpose import pyopenpose as op


print("Is cuda available: ",torch.cuda.is_available())

print("Setting up openpose")
params = dict()
params["model_folder"] = "/home/vishwas/orbslam/openpose/models"    
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
print("Openpose setup complete")

# tracker constants
REID_MODEL = ""
MAX_COS_DIST = 0.5
MAX_TRACK_AGE = 100

frame_count = 0
obstacle_coors = [[], [], [], []]
human_track_lines = []

# min and max height to remove floor and ceiling points
min_obstacle_height = -0.2
max_obstacle_height = 0.45

map_size = 1000 # Number of pixels in occupancy map
map_resolution = 0.01 # Size of 1 pixel in m

depth_resolution = 0.05
human_width = 0.3

free_probability = 5
occupied_probability = 5

# Set the parameters using ORBSLAM3 config file
width = 2556
height = 1179
fx = 1709.6
fy = 1710.5
cx = 947.7
cy = 489.8

k1 = 0.20921694
k2 = -0.30599282
p1 = -0.01183278
p2 = -0.00315412
k3 = -0.83206377

camera_height = 0.95 # In metres
person_height_min = 1.5
person_height_max = 2.0

torso_height_min = 2.0
torso_height_max = 2.3

# A - orbslam
# B - human height
# C - Feature point

# Create an OccupancyGrid message
a_map_msg = OccupancyGrid()
a_map_msg.header.frame_id = 'map'
a_map_msg.info.width = map_size
a_map_msg.info.height = map_size
a_map_msg.info.resolution = map_resolution
a_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
a_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
a_map_msg.data = [0] * (map_size * map_size)

b_map_msg = OccupancyGrid()
b_map_msg.header.frame_id = 'map'
b_map_msg.info.width = map_size
b_map_msg.info.height = map_size
b_map_msg.info.resolution = map_resolution
b_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
b_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
b_map_msg.data = [100] * (map_size * map_size)

c_map_msg = OccupancyGrid()
c_map_msg.header.frame_id = 'map'
c_map_msg.info.width = map_size
c_map_msg.info.height = map_size
c_map_msg.info.resolution = map_resolution
c_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
c_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
c_map_msg.data = [100] * (map_size * map_size)

ab_map_msg = OccupancyGrid()
ab_map_msg.header.frame_id = 'map'
ab_map_msg.info.width = map_size
ab_map_msg.info.height = map_size
ab_map_msg.info.resolution = map_resolution
ab_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
ab_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
ab_map_msg.data = [-1] * (map_size * map_size)

ac_map_msg = OccupancyGrid()
ac_map_msg.header.frame_id = 'map'
ac_map_msg.info.width = map_size
ac_map_msg.info.height = map_size
ac_map_msg.info.resolution = map_resolution
ac_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
ac_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
ac_map_msg.data = [-1] * (map_size * map_size)

bc_map_msg = OccupancyGrid()
bc_map_msg.header.frame_id = 'map'
bc_map_msg.info.width = map_size
bc_map_msg.info.height = map_size
bc_map_msg.info.resolution = map_resolution
bc_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
bc_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
bc_map_msg.data = [100] * (map_size * map_size)

abc_map_msg = OccupancyGrid()
abc_map_msg.header.frame_id = 'map'
abc_map_msg.info.width = map_size
abc_map_msg.info.height = map_size
abc_map_msg.info.resolution = map_resolution
abc_map_msg.info.origin.position.x = -(map_size*map_resolution)/2
abc_map_msg.info.origin.position.y = -(map_size*map_resolution)/2
abc_map_msg.data = [-1] * (map_size * map_size)

abc_map_publisher = None
ab_map_publisher = None
ac_map_publisher = None
bc_map_publisher = None
a_map_publisher = None
b_map_publisher = None
c_map_publisher = None

frame_subscriber = None
pointcloud_publisher = None
pose_publisher = None
marker_publisher = None

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='source', default='test1.mp4')
parser.add_argument('--map', type=str, help='map', default='map_points.txt')
opt = parser.parse_args()

cfg = get_cfg()
cfg.MODEL.DEVICE='cuda' #cuda or cpu
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_names = metadata.thing_classes

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7", use_cuda=True)

def get_rectangle_points(near_point, far_point, length):
    # Step 1: Calculate the slope of the line joining p and the origin
    slope = (far_point[1] - near_point[1]) / (far_point[0] - near_point[0])  # Assuming p is a tuple (x, y)

    # Step 2: Calculate the negative reciprocal of the slope
    perp_slope = -1 / slope if slope != 0 else math.inf  # Handle division by zero case

    # Step 3: Normalize the slope vector
    magnitude = math.sqrt(1 + perp_slope**2)
    norm_slope = (1 / magnitude, perp_slope / magnitude)

    # Step 4: Calculate the coordinates of the adjacent points
    near_point1 = (near_point[0] + length * norm_slope[0], near_point[1] + length * norm_slope[1])
    near_point2 = (near_point[0] - length * norm_slope[0], near_point[1] - length * norm_slope[1])

    far_point1 = (far_point[0] + length * norm_slope[0], far_point[1] + length * norm_slope[1])
    far_point2 = (far_point[0] - length * norm_slope[0], far_point[1] - length * norm_slope[1])
    
    return near_point1, near_point2, far_point2, far_point1
    
def mark_map_obstacles(points, masks):
    global a_map_msg, abc_map_msg, ab_map_msg, ac_map_msg, map_resolution

    for i in range(len(points)):
        point = points[i]
        mask = masks[i]
        if(np.isnan(point).any()):
            continue
        
        if(mask == 2): # Point can be considered an obstacle
             # Calculate the cell index based on the given point
             cell_x = int((point[0] + map_size * map_resolution / 2) / map_resolution)
             cell_y = int((point[1] + map_size * map_resolution / 2) / map_resolution)

             # Mark the cell as occupied
             if 0 <= cell_x < map_size and 0 <= cell_y < map_size:
                index = cell_x + cell_y * map_size
                a_map_msg.data[index] = 100
                
                if(abc_map_msg.data[index] == -1):
                    abc_map_msg.data[index] = 50
                abc_map_msg.data[index] += occupied_probability  # Mark the cell as occupied
                if(abc_map_msg.data[index] > 100):
                    abc_map_msg.data[index] = 100

                if(ab_map_msg.data[index] == -1):
                    ab_map_msg.data[index] = 50
                ab_map_msg.data[index] += occupied_probability  # Mark the cell as occupied
                if(ab_map_msg.data[index] > 100):
                    ab_map_msg.data[index] = 100

                if(ac_map_msg.data[index] == -1):
                    ac_map_msg.data[index] = 50
                ac_map_msg.data[index] += occupied_probability  # Mark the cell as occupied
                if(ac_map_msg.data[index] > 100):
                    ac_map_msg.data[index] = 100
                    
def transfrom_point(point, translation_matrix, rotation_matrix):
    # Homogeneous coordinates of the point
    point_homogeneous = np.array([[point[0]], [point[1]], [1]])

    # Apply the transformation
    transformed_point_homogeneous = np.dot(translation_matrix, np.dot(rotation_matrix, point_homogeneous))

    # Convert back to Cartesian coordinates
    transformed_point = np.array([transformed_point_homogeneous[0, 0], transformed_point_homogeneous[1, 0]])
                                          
    return [transformed_point[0],transformed_point[1]]

def update_occupancy_map(points, masks,  labels, camera_pose, person_coordinate, openpose_coordinate, person_id):
    global abc_map_msg, ab_map_msg, ac_map_msg, bc_map_msg, a_map_msg, b_map_msg, c_map_msg, map_resolution, marker_publisher, fy, cy, person_height_min, person_height_max, camera_height

    depths = []
    
    # Project camera pose to 2d
    camera_x = camera_pose[0][0]
    camera_y = camera_pose[0][1]
           
    r = Rotation.from_quat(camera_pose[1])
    rotation_matrix = r.as_matrix()
    camera_yaw = r.as_euler('zyx', degrees=False)[0]
    translation_matrix = np.array([[1, 0, camera_x],
                                   [0, 1, camera_y],
                                   [0, 0, 1]])
    rotation_matrix = np.array([[np.cos(camera_yaw), -np.sin(camera_yaw), 0],
                                [np.sin(camera_yaw), np.cos(camera_yaw), 0],
                                [0, 0, 1]])
                           
    translation_3d = np.array(camera_pose[0])
    for i in range(len(points)):
        point = points[i]
        mask = masks[i]
        label = labels[i]
        
        if(np.isnan(point).any()):
            continue
        
        # Transform the point with camera pose
        translated_point = point - translation_3d
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)
        transformed_point = np.dot(inv_rotation_matrix, translated_point)
        
        depth = transformed_point[0] # Depth does not consider the y or z direction.
        depths.append(depth)
    
    if(depths == []):
        depths.append(0)

    max_depth = max(depths)+1
    bins = ceil(max_depth/depth_resolution)+1
    near_bins = [0]*bins
    far_bins = [0]*bins
    near_point_count = 0
    far_point_count = 0
    
    near_min_y = 10**5
    near_max_y = -10**5
    far_min_y = 10**5
    far_max_y = -10**5

    for i,depth in enumerate(depths):
        bin_index = int(depth/depth_resolution)
        if(labels[i] == -1):
            near_bins[bin_index] += 1
            near_point_count += 1
            near_min_y = min(near_min_y, points[i][1])
            near_max_y = max(near_max_y, points[i][1])
        elif(labels[i] == 1 and masks[i] == 2):
            far_bins[bin_index] += 1
            far_point_count += 1
            far_min_y = min(far_min_y, points[i][1])
            far_max_y = max(far_max_y, points[i][1])
        elif(labels[i] == 1):
            far_min_y = min(far_min_y, points[i][1])
            far_max_y = max(far_max_y, points[i][1])
            

    final_bins = []
    last_near_bin = -1
    first_far_bin = len(far_bins)+2

    for i in range(len(near_bins)):
        near_bin = near_bins[i]
        far_bin = far_bins[i]
        if(near_bin == 0 and far_bin == 0):
            final_bins.append([0,0])
        elif(near_bin >= far_bin):
            final_bins.append([near_bin,-1])
            last_near_bin = max(last_near_bin, i)
            first_far_bin = len(far_bins)+2 # First far bin needs to be equal or greater than near bins
        else:
            final_bins.append([far_bin,1])
            first_far_bin = min(first_far_bin, i)

    if(True):
      
        old_start_depth = (last_near_bin+1)*depth_resolution
        old_end_depth = first_far_bin*depth_resolution
        
        # Based on height assumption
        y_scale = 12.0
        x_scale = -1.0
        person_x, person_y = person_coordinate
        person_pixel_height = (cy - person_y)*y_scale
        
        if(openpose_coordinate[0][0] == 0.0 and openpose_coordinate[0][1] == 0.0):
            return
                    
        openpose_x , openpose_y = openpose_coordinate[0][0], openpose_coordinate[0][1]
        openpose_pixel_height = ((openpose_coordinate[0][0]-openpose_coordinate[1][0])**2 + (openpose_coordinate[0][1]-openpose_coordinate[1][1])**2)**0.5
        openpose_pixel_height = openpose_pixel_height*y_scale
        
        if(person_pixel_height < 0):
            return

        #start_depth = ((person_height_min-camera_height) * fy) / person_pixel_height
        #end_depth = ((person_height_max-camera_height) * fy)/ person_pixel_height
            
        start_depth = (torso_height_min * fy) / openpose_pixel_height
        end_depth = (torso_height_max * fy) / openpose_pixel_height
        person_x, person_y = openpose_y, openpose_x

        #Find bounding box based on depth
        bbox_coor = [ [start_depth,  (person_x - cx) * (start_depth / (fx*x_scale))], [end_depth, (person_x - cx) * (end_depth / (fx*x_scale))]]
        old_bbox_coor = [ [old_start_depth,  (person_x - cx) * (old_start_depth / (fx*x_scale))], [old_end_depth, (person_x - cx) * (old_end_depth / (fx*x_scale))]]
        
        transformed_bbox_coor = [transfrom_point(point, translation_matrix, rotation_matrix) for point in bbox_coor]
        old_transformed_bbox_coor = [transfrom_point(point, translation_matrix, rotation_matrix) for point in old_bbox_coor]
        
        rectangle_coors = get_rectangle_points(transformed_bbox_coor[0], transformed_bbox_coor[1], human_width/2)
        old_rectangle_coors = get_rectangle_points(old_transformed_bbox_coor[0], old_transformed_bbox_coor[1], human_width/2)
        rectangle_points = [RosPoint(i[0],i[1],0.0) for i in rectangle_coors]
        old_rectangle_points = [RosPoint(i[0],i[1],0.0) for i in old_rectangle_coors]

        rectangle_marker = Marker()
        rectangle_marker.header.frame_id = "map"
        rectangle_marker.header.stamp = rospy.Time.now()
        rectangle_marker.ns = "human_height"
        rectangle_marker.id = person_id
        rectangle_marker.type = Marker.LINE_LIST
        rectangle_marker.action = Marker.ADD
        rectangle_marker.scale.x = 0.02  # Line width
        rectangle_marker.color.r = 1.0
        rectangle_marker.color.g = 0.0
        rectangle_marker.color.b = 0.0
        rectangle_marker.color.a = 1.0
        rectangle_marker.lifetime = rospy.Duration(3.0)
        
        old_rectangle_marker = Marker()
        old_rectangle_marker.header.frame_id = "map"
        old_rectangle_marker.header.stamp = rospy.Time.now()
        old_rectangle_marker.ns = "feature_points"
        old_rectangle_marker.id = person_id
        old_rectangle_marker.type = Marker.LINE_LIST
        old_rectangle_marker.action = Marker.ADD
        old_rectangle_marker.scale.x = 0.02  # Line width
        old_rectangle_marker.color.r = 0.0
        old_rectangle_marker.color.g = 0.0
        old_rectangle_marker.color.b = 1.0
        old_rectangle_marker.color.a = 1.0
        old_rectangle_marker.lifetime = rospy.Duration(3.0)

        # Define the rectangle's corner coordinates
        corner_points = [
            rectangle_points[0], rectangle_points[1],
            rectangle_points[1], rectangle_points[2],
            rectangle_points[2], rectangle_points[3],
            rectangle_points[3], rectangle_points[0],
        ]
        
        # Define the rectangle's corner coordinates
        old_corner_points = [
            old_rectangle_points[0], old_rectangle_points[1],
            old_rectangle_points[1], old_rectangle_points[2],
            old_rectangle_points[2], old_rectangle_points[3],
            old_rectangle_points[3], old_rectangle_points[0],
        ]

        if(end_depth-start_depth < 5.0):
            rectangle_marker.points = corner_points
            marker_publisher.publish(rectangle_marker)

            min_x = min([i[0] for i in rectangle_coors])
            max_x = max([i[0] for i in rectangle_coors])
            min_y = min([i[1] for i in rectangle_coors])
            max_y = max([i[1] for i in rectangle_coors])
            polygon = Polygon(rectangle_coors)
        
            for x in frange(min_x, max_x, map_resolution):
                for y in frange(min_y, max_y, map_resolution):
                    point = Point(x, y)
                    if polygon.contains(point):
                        # Calculate the cell index based on the given point
                        cell_x = int((x + map_size * map_resolution / 2) / map_resolution)
                        cell_y = int((y + map_size * map_resolution / 2) / map_resolution)
    
                        if 0 <= cell_x < map_size and 0 <= cell_y < map_size:
                            index = cell_x + cell_y * map_size
                            b_map_msg.data[index] = 0
                            bc_map_msg.data[index] = 0
                            
                            if(abc_map_msg.data[index] == -1):
                                abc_map_msg.data[index] = 50
                            abc_map_msg.data[index] -= free_probability  # Mark the cell as occupied
                            if(abc_map_msg.data[index] < 0):
                                abc_map_msg.data[index] = 0

                            if(ab_map_msg.data[index] == -1):
                                ab_map_msg.data[index] = 50
                            ab_map_msg.data[index] -= free_probability  # Mark the cell as occupied
                            if(ab_map_msg.data[index] < 0):
                                ab_map_msg.data[index] = 0

        if(last_near_bin != -1 and old_end_depth-old_start_depth < 3.0 and not np.isnan(old_rectangle_coors).any()):
            old_rectangle_marker.points = old_corner_points
            marker_publisher.publish(old_rectangle_marker)
            
            min_x = min([i[0] for i in old_rectangle_coors])
            max_x = max([i[0] for i in old_rectangle_coors])
            min_y = min([i[1] for i in old_rectangle_coors])
            max_y = max([i[1] for i in old_rectangle_coors])
            polygon = Polygon(old_rectangle_coors)
        
            for x in frange(min_x, max_x, map_resolution):
                for y in frange(min_y, max_y, map_resolution):
                    point = Point(x, y)
                    if polygon.contains(point):
                        # Calculate the cell index based on the given point
                        cell_x = int((x + map_size * map_resolution / 2) / map_resolution)
                        cell_y = int((y + map_size * map_resolution / 2) / map_resolution)
    
                        if 0 <= cell_x < map_size and 0 <= cell_y < map_size:
                            index = cell_x + cell_y * map_size
                            c_map_msg.data[index] = 0
                            bc_map_msg.data[index] = 0
                            
                            if(abc_map_msg.data[index] == -1):
                                abc_map_msg.data[index] = 50
                            abc_map_msg.data[index] -= free_probability  # Mark the cell as occupied
                            if(abc_map_msg.data[index] < 0):
                                abc_map_msg.data[index] = 0

                            if(ac_map_msg.data[index] == -1):
                                ac_map_msg.data[index] = 50
                            ac_map_msg.data[index] -= free_probability  # Mark the cell as occupied
                            if(ac_map_msg.data[index] < 0):
                                ac_map_msg.data[index] = 0

        #indices = range(len(near_bins))
    
        # Clear the previous plot
        #plt.close()
        #fig, ax = plt.subplots()
    
        # Set the labels and title
        #ax.set_xlabel('Depth')
        #ax.set_ylabel('Points')
        #ax.set_title('Bar Chart with number of near/far points')
    
        # Plot the new bars    
        # Set the lower value in the front
        #for i in indices:
        #    if(near_bins[i] <= far_bins[i]):
        #        ax.bar(i*depth_resolution, far_bins[i], color='orange')
        #        ax.bar(i*depth_resolution, near_bins[i], color='blue')
        #    else:
        #        ax.bar(i*depth_resolution, near_bins[i], color='blue')
        #        ax.bar(i*depth_resolution, far_bins[i], color='orange')
    
        # Show the updated plot
        #plt.pause(1)    

# Helper function to generate a range with floating point step size
def frange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

def check_depth(img, x_min, y_min, x_max, y_max, x_feat, y_feat):
    y_max = y_max + 10 if (y_max + 30) < img.shape[0] else y_max
    if x_max > x_feat > x_min:
        if y_feat > y_max:
            d = -1
        else:
            d = 1
    else:
        d = 0

    return d

def relative_person_obstacle(img, person_coor, obstacle_coors):
    labels = []
    for i,obstacle in enumerate(obstacle_coors):
        d = 0
        x_feat, y_feat = obstacle[0], obstacle[1]
        if person_coor!=[]:
            d = check_depth(img, min(np.array(person_coor)[:, 0]), min(np.array(person_coor)[:, 1]),
                            max(np.array(person_coor)[:, 0]), max(np.array(person_coor)[:, 1]), x_feat, y_feat)
        labels.append(d)
    return labels

def frame_callback(msg):
    global frame_count, metadata, class_names, deepsort, human_track_lines
    global abc_map_publisher, ab_map_publisher, ac_map_publisher, bc_map_publisher, a_map_publisher, b_map_publisher, c_map_publisher
    global abc_map_msg, ab_map_msg, ac_map_msg, bc_map_msg, a_map_msg, b_map_msg, c_map_msg
    rospy.loginfo("Received frame")

    try:
        # Convert the ROS Image message to an OpenCV image
        frame = np.frombuffer(msg.image.data, dtype=np.uint8).reshape(msg.image.height, msg.image.width, 3)
    except Exception as e:
        rospy.logerr("Failed to convert image: %s", str(e))
        return

    # Process Image
    print("Running openpose")
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    
    pts_2d = [[int(i.x),int(i.y)] for i in msg.points_2d]
    pts_3d = [[i.x,i.y,i.z] for i in msg.points_3d]
    
    # 0 - Outside FOV, 1 - Inside FOV, 2 - Selcted as feature
    map_mask = np.array([2  if (point[2] >= min_obstacle_height and point[2] <= max_obstacle_height) else 0 for point in pts_3d])
    frame_count += 1

    outputs = predictor(frame)
    v = Visualizer(frame[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)

    person_indices = outputs["instances"].pred_classes == 0
    person_masks = outputs["instances"].get("pred_masks")[person_indices].to("cpu")
    bbox_xyxy = outputs["instances"].pred_boxes.tensor[person_indices].to("cpu")
    conf = outputs["instances"].scores[person_indices].to("cpu")
    
    """bbox_xcycwh = []
    for bbox in bbox_xyxy:
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = (xmax - xmin)*1.2
        height = (ymax - ymin)*1.2
        bbox_xcycwh.append([x_center, y_center, width, height])

    sort_outputs = deepsort.update(np.array(bbox_xcycwh), conf, frame)
    
    identities = []
    sort_frame_output = frame.copy()
    if len(sort_outputs) > 0:
        bbox_xyxy = sort_outputs[:, :4]
        identities = sort_outputs[:, -1]
        for i in range(len(identities)):
            xmin, ymin, xmax, ymax = bbox_xyxy[i]
            identity = identities[i]
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            while(len(human_track_lines)<identity+1):
                human_track_lines.append([])
            human_track_lines[identity].append([x_center, y_center])
        sort_frame_output = draw_bboxes(frame, bbox_xyxy, identities)

    # Visualize the detection results with the specified confidence threshold
    v.draw_instance_predictions(outputs["instances"].to("cpu"))
    frame_output = v.get_output().get_image()[:, :, ::-1].copy()

    for identity in identities:
        track_line = human_track_lines[identity]
        if(track_line != []):
            # Set a random seed based on the id_number
            random.seed(identity)

            # Generate a random color in the range [0, 255] for each channel (B, G, R)
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw lines on the output image for the specified pixel list with the random color
            for i in range(1, len(track_line)):
                x1, y1 = track_line[i - 1]
                x2, y2 = track_line[i]
                cv2.line(sort_frame_output, (int(x1),int(y1)), (int(x2),int(y2)), random_color, 4)"""

    frame_output = frame.copy()
    sort_frame_output = frame.copy()
    masks = person_masks.numpy().astype(int)*255
    masks = masks.astype(np.uint8)

    feature_size = 5
    #feature_image = frame.copy()
    height, width, channels = frame.shape
    feature_image = np.zeros((height, width, channels), dtype=frame.dtype)
    combined_mask = np.zeros((height, width, 1), dtype=np.uint8)

    for mask in masks:
        combined_mask = cv2.add(combined_mask,mask)
    mask_coor = [[] for i in range(len(masks))]
    person_coor = []

    if len(masks)>0:
        for i,point in enumerate(pts_2d):
            x,y = point
            good = True
            for mask in masks:
                if(mask[y,x] != 0):
                    good = False
                    map_mask[i] = 0
            if(good):
                cv2.rectangle(feature_image, (x-feature_size, y-feature_size), (x+feature_size, y+feature_size), (0, 255, 0), thickness=3)

        for i in range(len(masks)):
            contours, _ = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            person_x = -1
            person_y = 1e5
            for object in contours:
                for point in object:
                    mask_coor[i].append((int(point[0][0]), int(point[0][1])))
                    y = point[0][1]
                    if(y < person_y):
                        person_x = point[0][0]
                        person_y = y
                break
            person_x, person_y = int(person_x), int(person_y)
            person_coor.append([person_x, person_y])
            #if(person_x != -1):
            #    cv2.rectangle(feature_image, (person_x-feature_size, person_y-feature_size), (person_x+feature_size, person_y+feature_size), (0, 0, 255), thickness=6)

    else:    
        for x,y in pts_2d:
            cv2.rectangle(feature_image, (x-feature_size, y-feature_size), (x+feature_size, y+feature_size), (0, 255, 0), thickness=3)
    
    openpose_coor = None
    if(datum.poseKeypoints is not None):
        openpose_coor = [datum.poseKeypoints[0][0],datum.poseKeypoints[0][8]]
        for point in openpose_coor:
             print(point)
             cv2.rectangle(feature_image, (int(point[0])-feature_size, int(point[1])-feature_size), (int(point[0])+feature_size, int(point[1])+feature_size), (0, 0, 255), thickness=6)

    frame_output = cv2.add(frame_output,feature_image)
    frame = cv2.add(frame,feature_image)
    obstacle_coors[0] = obstacle_coors[1]
    obstacle_coors[1] = obstacle_coors[2]
    obstacle_coors[2] = obstacle_coors[3]
    obstacle_coors[3] = pts_2d
    
    res = (700,500)
    frame_output = cv2.resize(frame_output, res)
    #sort_frame_output = cv2.resize(sort_frame_output, res)
    combined_mask = cv2.resize(combined_mask, res)
    cv2.imshow('detectron_output', frame_output)
    #cv2.imshow('sort_output', sort_frame_output)
    cv2.imshow('person_mask', combined_mask)
    cv2.waitKey(1)
    
    if obstacle_coors[0]==[]:
        return
    
    camera_pose = [ [msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z],
                    [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]]
    mark_map_obstacles(pts_3d, map_mask)
    for i in range(len(masks)):
        feature_label = relative_person_obstacle(frame_output, mask_coor[i], obstacle_coors[3])
        update_occupancy_map(pts_3d, map_mask, feature_label, camera_pose, person_coor[i], openpose_coor, i)
    
    pose_publisher.publish(msg.pose)
    projection.publish_mappoints(pts_3d, map_mask, pointcloud_publisher)
    abc_map_publisher.publish(abc_map_msg)
    ab_map_publisher.publish(ab_map_msg)
    ac_map_publisher.publish(ac_map_msg)
    bc_map_publisher.publish(bc_map_msg)
    a_map_publisher.publish(a_map_msg)
    b_map_publisher.publish(b_map_msg)
    c_map_publisher.publish(c_map_msg)

def main():
    # Initialize ros
    global pointcloud_publisher, frame_subscriber, pose_publisher, abc_map_publisher, ab_map_publisher, ac_map_publisher, bc_map_publisher, a_map_publisher, b_map_publisher, c_map_publisher, marker_publisher

    rospy.init_node('human_occlusion_publisher')
    pointcloud_publisher = rospy.Publisher('near_far_points', PointCloud2, queue_size=1)
    pose_publisher = rospy.Publisher('frame_pose', PoseStamped, queue_size=10)
    abc_map_publisher = rospy.Publisher('abc_map', OccupancyGrid, latch=True, queue_size=1)
    ab_map_publisher = rospy.Publisher('ab_map', OccupancyGrid, latch=True, queue_size=1)
    ac_map_publisher = rospy.Publisher('ac_map', OccupancyGrid, latch=True, queue_size=1)
    bc_map_publisher = rospy.Publisher('bc_map', OccupancyGrid, latch=True, queue_size=1)
    a_map_publisher = rospy.Publisher('a_map', OccupancyGrid, latch=True, queue_size=1)
    b_map_publisher = rospy.Publisher('b_map', OccupancyGrid, latch=True, queue_size=1)
    c_map_publisher = rospy.Publisher('c_map', OccupancyGrid, latch=True, queue_size=1)
    marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    rospy.loginfo("Setup Complete")
    frame_subscriber = rospy.Subscriber('tracked_frame', TrackedFrame, frame_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
