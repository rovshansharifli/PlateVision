import re
import os
import cv2
import time
import numpy as np
from det.detector import YOLOv8Detector
from rec.ocr import PaddleOcrRead

from ultralytics.engine.results import Results
from ultralytics.utils.plotting import save_one_box


IMAGE_PATH = 'test_image_1.jpg'
SAVE_PATH = 'results'


minimum_width_thres = 0.1
border_touch_min = 0.05
border_touch_max = 0.95
resize_min_veh_size = 640


def transfer_to_recognition(boxes: Results, frame: np.array,) -> None:
    count = 0
    crop_dict = {}
    shared_queue = []
    for box in boxes.boxes:
        if box.cls.item() in [2, 3, 4, 5]:
            if box.xywhn.tolist()[0][2] > minimum_width_thres:
                crop = save_one_box(
                    box.xyxy,
                    frame.copy(),
                    gain=1.0,
                    pad=0,
                    save=False
                )
                width_square = ((box.xywh[0][2]*box.xywh[0][3])*(box.xywh[0][2]/box.xywh[0][3])).item()
                x1, y1, x2, y2 = (box.xyxyn[0][0]).item(), (box.xyxyn[0][1]).item(), (box.xyxyn[0][2]).item(), (box.xyxyn[0][3]).item()
                # When the bounding box touch to the frame limits, border_touch flag is added
                border_touch = True if ((x1 < border_touch_min) or (y1<border_touch_min) or (x2>border_touch_max) or (y2>border_touch_max)) else False 

                crop_dict[count] = {0: count, 1: width_square, 2: border_touch, 3: crop, 4: time.time()}
                count+=1
                    
    for key in crop_dict.keys():
        # Adding to recognition when the frame object is went of the frame
        shared_queue.append({ 0: key,
                              1: crop_dict[key][3],
                              2: crop_dict[key][0],
                            })
    return shared_queue


def is_point_in_quad(p: list, points: list) -> bool:
    a, b, c, d = points
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    return (
        cross_product(p, a, b) >= 0 and
        cross_product(p, b, c) >= 0 and
        cross_product(p, c, d) >= 0 and
        cross_product(p, d, a) >= 0
    ) or (
        cross_product(p, a, b) <= 0 and
        cross_product(p, b, c) <= 0 and
        cross_product(p, c, d) <= 0 and
        cross_product(p, d, a) <= 0
    )
  
def read_plate(vehicle_image: np.array):
    new_size = ()
    if (vehicle_image.shape[0] < resize_min_veh_size) or (vehicle_image.shape[1] < resize_min_veh_size):
        if vehicle_image.shape[1] > vehicle_image.shape[0]:
            ratio = resize_min_veh_size / vehicle_image.shape[0]
            new_size = [vehicle_image.shape[1]*ratio, resize_min_veh_size]
        elif vehicle_image.shape[0] > vehicle_image.shape[1]:
            ratio = resize_min_veh_size / vehicle_image.shape[1]
            new_size = [resize_min_veh_size, vehicle_image.shape[0]*ratio]

    if len(new_size) != 0:
        vehicle_image = cv2.resize(vehicle_image, list(map(int, new_size)))

    # Run OCR
    preds = ocr.run(vehicle_image)
    print(preds)
    
    text_res = ''
    if preds[0] is not None:

        points = []
        for pred in preds[0]:

            p1, p2, p3, p4 = pred[0]

            text_height = ((p4[0]-p1[0])**2+(p4[1]-p1[1])**2)**0.5
            center_point = (sum([x[0] for x in pred[0]])/len(pred[0]), sum([y[1] for y in pred[0]])/len(pred[0]))
            text = pred[1][0].replace('O', '0')

            points.append([[p1, p2, p3, p4], center_point, text, text_height])

        merged_points = points
        merge_needed = False
        for i, point in enumerate(points):
            center_below = [point[1][0], point[1][1]+point[3]]
            in_quad = []
            for y, p in enumerate(points):
                if p != point:
                    if is_point_in_quad(center_below, p[0]):
                        # pattern = r'[0-9O]{2}[A-Za-z0]{2}[0-9O]{3}'
                        pattern_above = r'[0-9O]{2}(?!\d)'
                        pattern_below = r'[A-Za-z0]{2}[0-9O]{3}(?!\d)'
                        # Find all matches in the string
                        match_above = re.findall(pattern_above, point[2])
                        match_below = re.findall(pattern_below, p[2])
                        if (len(match_above)!=0) and (len(match_below)!=0): 
                            edited_text = ''
                            edited_text=f'{match_above[0]}{match_below[0]}'
                            in_quad.append([i, y, edited_text])
            if len(in_quad)!=0:
                for q in in_quad:
                    merged_points[q[0]][2] = q[2]
                    text_res = q[2]
                    merge_needed = True

        pattern_matching = []
        # [p1, p2, p3, p4], center_point, text, text_height]
        if not merge_needed:
            for point in points:
                pattern = r'[0-9O]{2}[A-Za-z0]{2}[0-9O]{3}(?!\d)'
                # Find all matches in the string
                matches = re.findall(pattern, point[2])
                if len(matches)!=0:
                    text_res = matches[0]
                    pattern_matching.append({0: matches[0], 1: point[1]})
        else:
            for point in merged_points:
                # Regular expression to match the DDLLDDD format
                pattern = r'[0-9O]{2}[A-Za-z0]{2}[0-9O]{3}(?!\d)'

                # Find all matches in the string
                matches = re.findall(pattern, point[2])
                if len(matches)!=0:
                    text_res = matches[0]
                    pattern_matching.append({0: matches[0], 1: point[1]})
        if len(pattern_matching) == 0:
            # If none works well
            longest_text = ''
            for point in merged_points:
                if (len(longest_text) < len(point[2])) and (len(point[2])<10):
                    longest_text = point[2]
            
            return longest_text
        elif len(pattern_matching) == 1:
            return text_res
        else:
            lowest = pattern_matching[0]
            for pat_m in pattern_matching:
                if lowest[1][1] < pat_m[1][1]:
                    lowest = pat_m
            return lowest[0]

if __name__ == '__main__':

    # Load the detection and recognition models
    model_faster = YOLOv8Detector('models/veh_det/best_pruned75_int8_openvino')
    ocr = PaddleOcrRead()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Load the image
    image = cv2.imread(IMAGE_PATH)

    # Run detection
    det_results = model_faster.detect(image)

    for veh in transfer_to_recognition(det_results, image):
        # Run recognition
        plate_text = read_plate(veh[1])
        print(plate_text)
        
        # Save the results
        img_rgb = cv2.cvtColor(veh[1], cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{SAVE_PATH}/{veh[2]}_predicted_{plate_text}.png', img_rgb)
    