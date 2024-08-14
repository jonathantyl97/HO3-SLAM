# HO3-SLAM: Human-Object Occlusion Ordering SLAM

HO3-SLAM is an advanced traversability mapping framework that integrates several state-of-the-art algorithms to perform Simultaneous Localization and Mapping (SLAM) while deducing occlusion ordering between humans and objects in the environment.

## Features

- Integration of multiple SLAM frameworks:
  - ORB-SLAM3
  - DSO (Direct Sparse Odometry)
- Human pose estimation using OpenPose
- Object detection and tracking with Detectron2
- Custom occlusion ordering algorithm for human-object interactions
- Point cloud generation from ORB-SLAM3 or DSO for 3D environment reconstruction

## Prerequisites

To use HO3-SLAM, ensure you have the following dependencies installed:

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [DSO](https://github.com/JakobEngel/dso)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Point Cloud Library (PCL)](https://pointclouds.org/)

(Note: The code has been tested and is expected to work on both Ubuntu 18.04 and 20.04)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/jonathantyl97/HO3-SLAM.git
   cd HO3-SLAM
   ```

2. Install the required dependencies (instructions for each dependency can be found in their respective repositories).

- This repo contains modified ORB-SLAM3 code. Kindly follow the same instructions as those provided at https://github.com/raulmur/ORB_SLAM to install the software. After installation, you should be able to run the ROS Mono example.
  
- This repo contains a modified DSO code. Kindly follow the same instructions as those provided at https://github.com/JakobEngel/dso to install the software.
  
- This repo contains modified OpenPose code. Kindly follow the same instructions as those provided at https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#cmake-configuration to install the software. Ensure to enable Python bindings so that openpose can be used in Python programs.

After installation, you can verify that openpose works correctly using an inbuilt Python example. Follow instructions from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/03_python_api.md to run the examples.

4. Build the project:

The repo contains a ROS workspace with the required ROS nodes for the project. After the installation of ORB SLAM, DSO, and OpenPose, build the workspace using the command given below. After a successful build, source the workspace to be able to run the nodes. 
   ```
   cd ros_ws
   catkin_make
   ```

## Usage

(Here, you should provide basic usage instructions. For example:)

1. ROSbag creation:
To be able to use the recorded videos, they need to be converted into a ROS bag file. A Python utility script has been added to do this. You can run the script using the below command
   ```
   python3 full_framework/video2bag.py <input_video_path> <output_bag_path>
   ```
2. Running ORBSLAM Example:

Run these commands in separate terminals

Terminal 1: (ORB-SLAM3 Mono node)
   ```
rosrun ORB-SLAM3 Mono ORB-SLAM3/Vocabulary/ORBvoc.txt ./iphone.yaml
   ```
Terminal 2: (Traversability Estimation script)
   ```
cd ros_ws/src/occlusion_mapping/scripts/yolov7
python3 detect_detectron2.py
   ```
Terminal 3: (rviz visualizer)
   ```
rviz -d orbslam.rviz
   ```
Terminal 4: (Run the rosbag)
   ```
rosbag play <bag_path> --rate 0.5
   ```
3. Running DSO Example:

Run these commands in separate terminals

Terminal 1: (DSO ROS node)
   ```
rosrun dso_ros dso_live image:=/camera/image_raw calib=calib.txt
   ```
Terminal 2: (Frame merger script)
   ```
cd ros_ws/src/dso_ros/scripts
python3 frame_merger.py
   ```
Terminal 3: (Traversability estimation script)
   ```
cd full_framework/ros_ws/src/occlusion_mapping/scripts/yolov7
python3 detect_detectron2.py
   ```
Terminal 4:
   ```
rviz -d dso.rviz
   ```
Terminal 5: (Run the rosbag)
   ```
rosbag play <bag_path> --rate 0.5
   ```

## Contributing

Contributions to HO3-SLAM are welcome! Please feel free to submit pull requests, create issues, or spread the word.

## License


## Citation

If you use HO3-SLAM in your research, please cite our work:

Plain text:
J. T. Yu Liang and T. Kanji, "HO3-SLAM: Human-Object Occlusion Ordering as Add-on for Enhancing Traversability Prediction in Dynamic SLAM," 2023 IEEE 13th International Conference on Pattern Recognition Systems (ICPRS), Guayaquil, Ecuador, 2023, pp. 1-7, doi: 10.1109/ICPRS58416.2023.10179060. keywords: {Location awareness;Simultaneous localization and mapping;Pedestrians;Navigation;Prototypes;Detectors;Robustness},


BibTex:
@INPROCEEDINGS{10179060,
  author={Yu Liang, Jonathan Tay and Kanji, Tanaka},
  booktitle={2023 IEEE 13th International Conference on Pattern Recognition Systems (ICPRS)}, 
  title={HO3-SLAM: Human-Object Occlusion Ordering as Add-on for Enhancing Traversability Prediction in Dynamic SLAM}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  keywords={Location awareness;Simultaneous localization and mapping;Pedestrians;Navigation;Prototypes;Detectors;Robustness},
  doi={10.1109/ICPRS58416.2023.10179060}}

## Contact

For any queries, please contact Jonathan Tay at jonathantyl97@gmail.com
