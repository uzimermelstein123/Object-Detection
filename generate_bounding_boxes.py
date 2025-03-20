from ouster.sdk import open_source, client
import numpy as np
import cv2 
from ultralytics import YOLO

pcap_path = r"C:\Users\umermelstein2024\Test Ouster\DataVideos\Ouster Data\first-recording.pcap"
metadata_path = r"C:\Users\umermelstein2024\Test Ouster\DataVideos\Ouster Data\first-recording.json"


source = open_source(pcap_path, meta=[metadata_path])

model = YOLO("yolo11m.pt")


for i, scan in enumerate(source):
    ref_data = scan.field(client.ChanField.REFLECTIVITY)
    ref_val = client.destagger(source.metadata, ref_data)



    results = model.predict(cv2.cvtColor(ref_val, cv2.COLOR_GRAY2RGB), imgsz=(128,1024), stream=True,
    classes=[0]) #Only people are detected, results are continuous stream


for j, res in enumerate(results):
    xyxy = res.boxes.xyxy.cpu().numpy()
    np.savetxt(f"./BoundingBoxCoords/boxes{j}_xyxy.txt", xyxy, fmt="%.6f", delimiter=",")

# Generate bounding box coordinates and save them to disk for further processing on point cloud  