import os
import cv2
import time
import string
import numpy as np
from .ocr import PaddleOcrRead
from .results import Results, TrackedVehicles
from multiprocessing import Queue

class PlateRecognition():
    def __init__(self, shared_queue: Queue, save_crop: bool = False, save_path: string = '', ) -> None:
        self.ocr = PaddleOcrRead()
        self.counter = 0
        self.save_crop = save_crop
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.save_path = save_path
        self.shared_queue = shared_queue
        self.results = Results()
        self.tracked_vehicles = TrackedVehicles()

    def predict(self, vehicle_image: np.array) -> string:
        res = self.ocr.run(vehicle_image)
        # TODO: how to take care of the following case; Also the AZ read case
        # res = [[[[[143.0, 152.0], [191.0, 150.0], [192.0, 168.0], [144.0, 170.0]], ('99', 0.9850911498069763)], [[[152.0, 167.0], [206.0, 167.0], [206.0, 184.0], [152.0, 184.0]], ('FC017', 0.9911990165710449)]]]

        text_res = ''
        all_texts = []
        all_texts_loc = []
        if res[0] is not None:
            for text in res[0]:
                all_texts.append(text[1])
                all_texts_loc.append(text[0])

            text_res = ''.join([text[0] for text in all_texts])

        # TODO: take care of the rectangular Azerbaijani plate numbers and other written chars other than plate

        if self.save_crop:
            vehicle_image = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{self.save_path}/crop_{self.counter}_{text_res}.jpg', vehicle_image)

        self.counter+=1

        return text_res
    
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
                # box = queue_object[3]
                if not self.tracked_vehicles.check_if_exists(track_id):
                    self.tracked_vehicles.add_tracked_vehicle(track_id, crop, n_frame)

                    non_rec_crops = self.tracked_vehicles.get_non_rec_crops()
                    if len(non_rec_crops) > 0:
                        plate_text = self.predict(self.tracked_vehicles.get_crop_by_id(track_id))
                        self.tracked_vehicles.set_plate_by_id(track_id, plate_text)
                        self.results.update(id=track_id, text=plate_text)
                        print(plate_text)

                    self.tracked_vehicles.remove_expired_tracked_vehicles(n_frame)
                time.sleep(0.05)
