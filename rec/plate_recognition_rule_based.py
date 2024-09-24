import os
import re
import cv2
import time
import string
import numpy as np
from .ocr import PaddleOcrRead
from .results import Results, TrackedVehicles
from multiprocessing import Queue

class PlateRecognition():
    def __init__(self, shared_queue: Queue, save_crop: bool = False, 
                 save_path: string = '', resize_min_veh_size: int = 640) -> None:
        
        self.ocr = PaddleOcrRead()
        self.counter = 0
        self.save_crop = save_crop
        self.shared_queue = shared_queue
        self.results = Results()
        self.tracked_vehicles = TrackedVehicles()
        self.resize_min_veh_size = resize_min_veh_size

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.save_path = save_path

    def is_point_in_quad(self, p: list, points: list) -> bool:
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

    def predict(self, vehicle_image: np.array) -> string:
        new_size = ()
        if (vehicle_image.shape[0] < self.resize_min_veh_size) or (vehicle_image.shape[1] < self.resize_min_veh_size):
            if vehicle_image.shape[1] > vehicle_image.shape[0]:
                ratio = self.resize_min_veh_size / vehicle_image.shape[0]
                new_size = [vehicle_image.shape[1]*ratio, self.resize_min_veh_size]
            elif vehicle_image.shape[0] > vehicle_image.shape[1]:
                ratio = self.resize_min_veh_size / vehicle_image.shape[1]
                new_size = [self.resize_min_veh_size, vehicle_image.shape[0]*ratio]

        if len(new_size) != 0:
            vehicle_image = cv2.resize(vehicle_image, list(map(int, new_size)))

        # Run OCR
        preds = self.ocr.run(vehicle_image)
        
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
                        if self.is_point_in_quad(center_below, p[0]):
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

    def run_engine(self, ) -> None:
        while True:
            queue_object = self.shared_queue.get()
            # { 0: track_id,
            #   1: crop,
            #   2: n_frame}
            if queue_object is not None:
                track_id = queue_object[0]
                crop = queue_object[1]
                n_frame = queue_object[2]

                if not self.tracked_vehicles.check_if_exists(track_id):
                    self.tracked_vehicles.add_tracked_vehicle(track_id, crop, n_frame)

                    non_rec_crops = self.tracked_vehicles.get_non_rec_crops()
                    if len(non_rec_crops) > 0:
                        plate_text = self.predict(self.tracked_vehicles.get_crop_by_id(track_id))
                        self.tracked_vehicles.set_plate_by_id(track_id, plate_text)
                        self.results.update(id=track_id, text=plate_text)
                        print(plate_text)
                            
                        cv2.imwrite(f'{self.save_path}/{track_id}_{n_frame}_predicted_{plate_text}.png', self.tracked_vehicles.get_crop_by_id(track_id))

                    self.tracked_vehicles.remove_expired_tracked_vehicles(n_frame)
                
                time.sleep(0.05)
