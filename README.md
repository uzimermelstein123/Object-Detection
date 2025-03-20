# Bounding Box Generation & 3D Point Cloud Visualization

This repository contains two key Python scripts for working with object detection in **2D and 3D**:

- **`generate_bounding_boxes.py`** - Generates bounding boxes and saves them as coordinate files. Shows basic functionality of what a YOLO model does.
- **`visualize_detection.py`** - Generates and processes bounding boxes and projects them onto a **3D point cloud** for visualization.

## 📥 Input
- A **PCAP file** containing LiDAR data.

## ⚙️ Processing
- Using **Ouster's client**, we extract **reflectivity** (**better results**) or **near-IR** data to generate **2D bounding boxes**.

## 📤 Output
- **2D bounding boxes**.
- **3D point cloud projections** of these bounding boxes using **XYZLUT**.

## 🔄 Process Overview

1. **Extract Reflectivity Images**
   - Reflectivity images are **intensity maps** generated from LiDAR scans, showing how much light is reflected back from objects in the environment.
   - Using the **Ouster SDK**, we extract reflectivity data from a **PCAP file** and convert it into a **2D image-like representation**.

2. **Apply YOLO Object Detection**
   - YOLO is applied to the **reflectivity images** to generate **2D bounding box coordinates** around detected objects.

3. **Project 2D Bounding Boxes onto 3D Point Cloud**
   - Using the **XYZLUT** (Lookup Table for XYZ coordinates), we project the **bounding box regions** from 2D onto the **3D point cloud**.
   - This effectively allows **3D object detection** using **2D images**, which is significantly **faster** and **less computationally expensive** compared to full **3D detection models**.


![image](https://github.com/user-attachments/assets/2f38170f-fefb-4a61-a09b-a298063facbe)

This is an example output of a red color mask applied to the region of the bounding box in 3D space with postprocessing of outlier points. 
