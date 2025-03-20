Bounding Box Generation & 3D Point Cloud Visualization
This repository contains two key Python scripts for working with object detection in 2D and 3D:

generate_bounding_boxes.py - Generates bounding boxes and saves them as coordinate files. Shows basic functionality of what YOLO model is doing.
visualize_detection.py - Generates and processes bounding boxes and projects them onto a 3D point cloud for visualization. 


Input: A PCAP file containing LiDAR data.
Processing: Using Ouster's client, we extract reflectivity (better results) or near-ir data to generate 2D bounding boxes.
Output:
2D bounding boxes.
3D point cloud projections of these bounding boxes using XYZLUT for visualization. 

Process Overview:
Reflectivity images are essentially intensity maps generated from LiDAR scans, showing how much light is reflected back from objects in the environment. Using the Ouster SDK, we can extract reflectivity data from a PCAP file and convert it into a 2D image-like representation. We then apply YOLO object detection on these reflectivity images to generate 2D bounding box coordinates. With these coordinates we are able to projet these bounding box regions onto 3D point clouds. This essentially allows us to do 3D object detection using 2D images (significantly faster and less computationally expensive).


![image](https://github.com/user-attachments/assets/2f38170f-fefb-4a61-a09b-a298063facbe)

This is an example output of a red color mask applied to the region of the bounding box in 3D space with postprocessing of outlier points. 
