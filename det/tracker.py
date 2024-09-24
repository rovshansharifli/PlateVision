import torch
import string
import argparse
import numpy as np
from ultralytics.engine.results import Results
from ultralytics.trackers.byte_tracker import BYTETracker
from .detector import YOLOv8Detector

class Tracker():
    def __init__(self, model_path: string, frame_rate: float, cls_name: dict, device: string, img_size: int):
        self.model_faster = YOLOv8Detector(model_path, 
                                           cls_names = cls_name, 
                                           threshold = 0.5,
                                           img_size=img_size, 
                                           device = device)
        parser = argparse.ArgumentParser()
        parser.add_argument("--track_high_thresh", default=0.5 ,type=str)
        parser.add_argument("--track_low_thresh", default=0.1 ,type=str)
        parser.add_argument("--new_track_thresh", default=0.6 ,type=str)
        parser.add_argument("--track_buffer", default=30 ,type=str)
        parser.add_argument("--match_thresh", default=0.95 ,type=str)
        parser.add_argument("--fuse_score", default=True ,type=str)
        args = parser.parse_args() 

        self.tracker = BYTETracker(args=args, frame_rate=frame_rate)
        self.cls_name = cls_name

    def track(self, frame: np.array) -> Results:
        # Run detection
        det_results = self.model_faster.detect(frame)
        # Add the detection into tracker
        tracked_objects = self.tracker.update(det_results.boxes.numpy()) if (det_results is not None) else []

        if len(tracked_objects)!=0:
            idx = tracked_objects[:, -1] 
            det_results = det_results[idx]
            
            # Concatinating for avoiding tracker to deform the bounding boxes
            concat = torch.cat((det_results.boxes.data[:, :4], torch.as_tensor(tracked_objects[:, 4:-1])), 1)

            update_args = {"boxes": concat}
            det_results.update(**update_args)
        else:
            # If there is no tracks then return an empty results
            det_results = Results(orig_img=frame, 
                                path='', names=self.cls_name,
                                boxes=torch.empty((0, 7), dtype=torch.float32), )
        
        return det_results
