from ouster.sdk import open_source, client
import numpy as np
import cv2 
from ultralytics import YOLO
from ouster.sdk import viz

#reference:
#https://static.ouster.dev/sdk-docs/python/viz/viz-api-tutorial.html 

pcap_path = r"C:\Users\umermelstein2024\Test Ouster\DataVideos\Ouster Data\first-recording.pcap"
metadata_path = r"C:\Users\umermelstein2024\Test Ouster\DataVideos\Ouster Data\first-recording.json"


source = open_source(pcap_path, meta=[metadata_path], index=True)


model_path = r"C:\Users\umermelstein2024\Test Ouster\runs\detect\yolo_v11_reflectivity10epochs2\weights\best.pt"

model = YOLO("yolo11m.pt") 

# scans = []
point_viz = viz.PointViz("Ouster LiDAR Point Cloud")

metadata = source.metadata  # This contains sensor calibration and frame structure


for i, scan in enumerate(source):
    scan = source[i]
    ref_data = scan.field(client.ChanField.REFLECTIVITY)
    range_data = scan.field(client.ChanField.RANGE)
    nearir_data = scan.field(client.ChanField.NEAR_IR)
    nearir_destaggered = client.destagger(source.metadata, nearir_data)
    ref_val = client.destagger(source.metadata, ref_data)
    # scans.append(ref_val)  

    
    xyzlut = client.XYZLut(metadata)  # Create lookup table to convert range data to XYZ
    xyz_coords = xyzlut(range_data)
    cloud = viz.Cloud(metadata)

    cloud.set_range(range_data)

    point_viz.add(cloud)

    img_aspect = (metadata.beam_altitude_angles[0] -
                metadata.beam_altitude_angles[-1]) / 360.0
    img_screen_height = 0.4  # [0..2]
    img_screen_len = img_screen_height / img_aspect

    signal = np.divide(ref_val, np.amax(ref_val), dtype=np.float32) #normalized reflective iage
    nearir_destaggered = np.divide(nearir_destaggered, np.amax(nearir_destaggered), dtype=np.float32) #normalized reflective iage

    #must convert to RGB
    res = model.predict(cv2.cvtColor(ref_val, cv2.COLOR_GRAY2RGB), imgsz=(128,1024), stream=True,
        classes=[0]                )

    #color_mask is all points matched to RGBA, (128*1024) to 4
    image_height, image_width = ref_val.shape 
    color_mask = np.zeros((xyz_coords.shape[0] * xyz_coords.shape[1], 4), dtype=np.uint8)
    roi_mask_2d = np.zeros((image_height, image_width), dtype=np.uint8)
    # new_roi_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    range_data_destaggered = client.destagger(metadata, range_data)

    for r in res:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            xmin, ymin, xmax, ymax = box  # Bounding box in reflectivity image space
            
            # cv2.rectangle(reflectivity_img, (int(xmin), int(ymin)),(int(xmax), int(ymax)), (0, 255, 0), 2)
        
            roi_values = range_data_destaggered[int(ymin):int(ymax), int(xmin):int(xmax)] 
            mean_distance = np.mean(roi_values)

            mean_70 = np.percentile(roi_values, 70)
            roi_mask_2d[int(ymin):int(ymax), int(xmin):int(xmax)] |= (roi_values < mean_70).astype(np.uint8)
        
            cv2.rectangle(signal, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
            cv2.rectangle(nearir_destaggered, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)

    # Apply the color mask to the LiDAR point cloud
    ref_img = viz.Image()
    ref_img.set_position(-img_screen_len / 2, img_screen_len / 2, 1 - img_screen_height, 1)

    nearir_img = viz.Image()
    nearir_img.set_position(-img_screen_len / 2, img_screen_len / 2, -1, -1 + img_screen_height)

    ref_img.set_image(signal) #signal defined above
    nearir_img.set_image(nearir_destaggered)

    point_viz.add(ref_img)
    point_viz.add(nearir_img)

    point_viz.update()


    roi_mask_staggered = client.destagger(metadata, roi_mask_2d, inverse=True) #stagger it using inverse True
    roi_mask_flat = roi_mask_staggered.flatten() #flatten to match shape of color_mask

    background_mask = roi_mask_flat == 0  # Explicitly ensure background mask
    foreground_mask = roi_mask_flat == 1  # Ensure foreground mask

    color_mask[background_mask] = [0, 0, 255, 255]  # Blue for background
    color_mask[foreground_mask] = [255, 0, 0, 255]  # Red for foreground

    cloud.set_mask(color_mask)
    point_viz.update()

    viz.add_default_controls(point_viz)

    signal_label = viz.Label(str(client.ChanField.REFLECTIVITY),
                        0.5,
                            0,
                            align_top=True)

    signal_label.set_scale(1)
    point_viz.add(signal_label) 

    nearir_label = viz.Label(str(client.ChanField.NEAR_IR),
                            0.5,
                            0.85,
                            align_top=False)
    nearir_label.set_scale(1)
    point_viz.add(nearir_label)

    point_viz.update()

    point_viz.run()