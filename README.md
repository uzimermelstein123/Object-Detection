## Object Detection Research
Overview
This project implements an object detection system that processes PCAP (packet capture) files and applies YOLOv11 detection algorithms on both reflectivity (recommended) and near-infrared (NIR) data streams.


Input:
3d point clouds, 2d reflectivity/near-ir images. All obtained from Ouster Sensors. https://ouster.com/

Output:
2d bounding box coordinates obtained from YOLO v11 detection model. 3d projection of bounding box coordinates using XYZ lookup table. 




![image](https://github.com/user-attachments/assets/6e199302-f4ef-4dd2-a0c5-eb4ad3204be5)

This is the point cloud data with a color mask obtained from the bounding box of the 2d reflectivity images. No postprocessing applied. 
