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

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='source', default='test1.mp4')
opt = parser.parse_args()


def check_depth(img, x_min, y_min, x_max, y_max, x_feat, y_feat):
	y_max = y_max + 10 if (y_max + 30) < img.shape[0] else y_max
	if x_max > x_feat > x_min:
		if y_feat > y_max:
			d = -1
		else:
			d = 1
	else:
		d = 0

	cv2.putText(img, str(d), (int(x_feat), int(y_feat)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 1)
	return d


def relative_person_obstacle(img, person_coor, obstacle_coors):
	count = 0
	result = ""
	person_region = Polygon(person_coor)
	for obstacle in obstacle_coors:
		x_feat, y_feat = obstacle[0], obstacle[1]
		if person_coor!=[]:
			d = check_depth(img, min(np.array(person_coor)[:, 0]), min(np.array(person_coor)[:, 1]),
							max(np.array(person_coor)[:, 0]), max(np.array(person_coor)[:, 1]), x_feat, y_feat)
		
		obstacle_coor = Point(obstacle[0], obstacle[1])
		if person_region.contains(obstacle_coor)==True:
			count+=1
	if count > 5:
		result = "In front of"
	elif count<3:
		result = "Behind"
	return result

def extract_obstacle_point(image, person_coor):
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
		if person_coor!=[]:
			person_region = Polygon(person_coor)
			# if person_region.contains(Point(x, y))==True:
			# 	continue
		obstacle_points.append([x, y])
	return obstacle_points

cfg = get_cfg()
cfg.MODEL.DEVICE='cuda' #for cpu
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

source_type = None
if opt.source.endswith(('jpg', 'png', 'jpeg')):
	source_type = 'image'
elif opt.source.endswith(('mp4', 'avi', 'MOV')):
	source_type = 'video'

print("Type: ", source_type)

if source_type=='image':
	im = cv2.imread(opt.source)
	outputs = predictor(im)
	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0, instance_mode=ColorMode.SEGMENTATION)
	person_masks = outputs["instances"][outputs['instances'].pred_classes == 0].get("pred_masks").to("cpu")
	person_box = outputs["instances"][outputs['instances'].pred_classes == 0].get("pred_boxes").to("cpu").tensor.numpy()

	for person_mask in person_masks:
		v.draw_soft_mask(person_mask, color='red')
	mask = outputs["instances"][outputs['instances'].pred_classes == 0].pred_masks.to("cpu").numpy().astype(int) * 255

	mask_coor = []
	if len(mask) > 0:
		mask = mask[0].astype(np.uint8)
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		for object in contours:
			for point in object:
				mask_coor.append((int(point[0][0]), int(point[0][1])))
			break

	im = v.get_output().get_image()[:, :, ::-1].copy()
	obstacle_coors = extract_obstacle_point(im, mask_coor)

	relative = relative_person_obstacle(im, mask_coor, obstacle_coors=obstacle_coors)

	if len(person_box)>0:
		person_box = person_box[0].tolist()
		cv2.putText(im, relative, (int(person_box[0] + (person_box[2]-person_box[0])/2), int(person_box[1] + (person_box[3]-person_box[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2, 2)
	cv2.imwrite("runs/" + opt.source, im)
elif source_type=='video':
	# Extract video properties
	video = cv2.VideoCapture(opt.source)
	width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames_per_second = video.get(cv2.CAP_PROP_FPS)
	video_writer = cv2.VideoWriter("runs/" + opt.source, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

	frame_count = 0
	obstacle_coors = [[], [], [], []]

	while True:
		ret, frame = video.read()
		if not ret:
			break

		print(frame_count)
		# if frame_count<150:
		# 	frame_count+=1
		# 	continue
		

		outputs = predictor(frame)
		v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0, instance_mode=ColorMode.SEGMENTATION)

		person_masks = outputs["instances"][outputs['instances'].pred_classes == 0].get("pred_masks").to("cpu")
		person_box = outputs["instances"][outputs['instances'].pred_classes == 0].get("pred_boxes").to("cpu").tensor.numpy()
		for person_mask in person_masks:
			v.draw_soft_mask(person_mask, color='cyan')

		frame_output = v.get_output().get_image()[:, :, ::-1].copy()
		mask = person_masks.numpy().astype(int)*255
		mask_coor = []
		#if len(mask)>0:
		#	mask = mask[0].astype(np.uint8)
		#	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		#	for object in contours:
		#		for point in object:
		#			mask_coor.append((int(point[0][0]), int(point[0][1])))
		#		break

		# if frame_count==0:
		# 	# obstacle_coors[0] = extract_obstacle_point(frame, mask_coor)
		# 	frame_count += 1
		# 	continue
		# elif frame_count==1:
		# 	# obstacle_coors[1] = extract_obstacle_point(frame, mask_coor)
		# 	frame_count += 1
		# 	continue
		# else:
		#obstacle_coors[0] = obstacle_coors[1]
		#obstacle_coors[1] = obstacle_coors[2]
		#obstacle_coors[2] = obstacle_coors[3]
		#obstacle_coors[3] = extract_obstacle_point(frame, mask_coor)

		#if obstacle_coors[0]==[]:
		#	continue
		# obstacle_coors_concat = np.concatenate((np.array(obstacle_coors[0]), np.array(obstacle_coors[1])))
		#relative = relative_person_obstacle(frame_output, mask_coor, obstacle_coors=obstacle_coors[0])
		#if len(person_box)>0:
		#	person_box = person_box[0].tolist()
		#	cv2.putText(frame_output, relative, (int(person_box[0] + (person_box[2]-person_box[0])/2), int(person_box[1] + (person_box[3]-person_box[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2, 2)
		#video_writer.write(frame_output)
		#frame_count += 1
		#frame_output = cv2.resize(frame_output, (640, 480))
		cv2.imshow('output', frame_output)
		cv2.waitKey(1)

	video.release()
	video_writer.release()
	cv2.destroyAllWindows()


