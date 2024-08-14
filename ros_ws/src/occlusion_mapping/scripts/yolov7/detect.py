import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import rospy
from sensor_msgs.msg import PointCloud2,PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import struct
from occlusion_mapping.msg import TrackedFrame

import warnings
warnings.filterwarnings("ignore")

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Set the path to the files and keyframe folder
mappoints_file = "map_points.txt"
trajectory_file = "keyframes.txt"
keyframes_dir = "keyframes"
projected_keyframes_dir = "projected_keyframes"

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

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def read_mappoints(filename):
    # Each line in file contains 3 values: x y z
    # Read the file and create a 2D Array
    points = []
    
    with open(filename, "r") as file:
        for line in file:
            row = [float(x) for x in line.split()]
            points.append(row)
    
    points = np.array(points)
    return points
    
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
    
    r_vector, _ = cv2.Rodrigues(r_matrix)
    return r_vector

def publish_mappoints(points, mask, publisher):    
    pointcloud = PointCloud2()

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1)]
            
    data = []
    for i, (x, y, z) in enumerate(points):
        g = 0
        r = 0
        if(mask[i]==1):
            g = 1
        else:
            r = 1
        rgb = struct.unpack('I', struct.pack('BBBB', 0, g*255, r*255, 255))[0]
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

    return projected_points[:, 0, :], valid_mask

def check_depth(img, x_min, y_min, x_max, y_max, x_feat, y_feat):
    y_max = y_max + 10 if (y_max + 30) < img.shape[0] else y_max
    person_region = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    # if person_region.contains(Point(x_feat, y_feat))==True:
    if x_max > x_feat > x_min:
        if y_feat > y_max:
            d = -1
        else:
            d = 1
    else:
        d = 0
    # else:
    #     d=0
    cv2.putText(img, str(d), (int(x_feat), int(y_feat)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, 1)
    return d

def relative_person_obstacle(img, person_coor, obstacle_coors):
    count = 0
    result = "Behind"
    person_region = Polygon([(person_coor[0], person_coor[1]), (person_coor[0], person_coor[3]), (person_coor[2], person_coor[3]), (person_coor[2], person_coor[1])])
    x_min, y_min, x_max, y_max = person_coor[0], person_coor[1], person_coor[2], person_coor[3]
    for obstacle in obstacle_coors:
        x_feat, y_feat = obstacle[0], obstacle[1]
        d = check_depth(img, x_min, y_min, x_max, y_max, x_feat, y_feat)
        if count > 5:
            result = "In front of"
        obstacle_coor = Point(obstacle[0], obstacle[1])
        if person_region.contains(obstacle_coor)==True:
            count+=1
    return result

def extract_obstacle_point(image):
    obstacle_points = []

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_color1 = np.asarray([36, 50, 70])  
    hsv_color2 = np.asarray([89, 255, 255]) 
    mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

    _, thresh=cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, (kernel)) 

    contours, _=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        obstacle_points.append([x, y])
    return obstacle_points
    
def setup():
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    obstacle_coors = [[], []]

def callback(msg):
    rospy.loginfo("Received Frame")
    
    global idx_frame

    # im0s = original image
    # img = letterboc image
    
    #for idx_frame, (path, img, im0s, vid_cap) in enumerate(dataset):
    if idx_frame==0:
        obstacle_coors[0] = extract_obstacle_point(im0s)
        continue
    else:
        obstacle_coors[0] = obstacle_coors[1]
        obstacle_coors[1] = extract_obstacle_point(im0s)            
            
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float() # uint8 to fp16/32

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=opt.augment)[0]

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=opt.augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t3 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if cls!=0:
                    continue

                person_coor = [v.cpu().numpy().tolist() for v in xyxy]
                relative = relative_person_obstacle(im0, person_coor, obstacle_coors=obstacle_coors[0])

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Resulting image is in im0
        # Stream results
        if view_img:
            cv2.imshow("image", im0)
            cv2.waitKey(1)  # 1 millisecond

def detect():
    #Initialize ros
    rospy.init_node('projection_publisher')
    pointcloud_publisher = rospy.Publisher('map_points', PointCloud2, queue_size=1)
    
    setup()    
    frame_subscriber = rospy.Subscriber('tracked_frame_topic', TrackedFrame, frame_callback)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model', default=True)
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
