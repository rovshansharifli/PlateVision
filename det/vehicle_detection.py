import cv2
import time
import string
import numpy as np
from .camera import Camera

from ultralytics.engine.results import Results
from ultralytics.utils.plotting import save_one_box

from .tracker import Tracker

class VehicleDetection():
    def __init__(self, cam_stream: Camera, model_path: string = '',
                 resize: int = 2, visualize: bool = False, 
                 verbose: bool = False, device: string = 'CPU', img_size: int = 640):
        
        self.resize = resize
        self.device = device
        self.verbose = verbose
        self.visualize = visualize
        self.cam_stream = cam_stream
        self.frame_limit = cam_stream.get_FPS_of_camera()

        # Initializing the Tracker
        self.tracker = Tracker(model_path=model_path, 
                               frame_rate=cam_stream.get_FPS_of_camera(),
                               device = device,
                               img_size = img_size,
                               cls_name={0: 'Person', 1: 'LP', 2: 'Car', 3: 'Motorcycle', 4: 'Truck', 5: 'Bus'})
        self.minimum_width_thres = 0.1
        self.border_touch_min = 0.05
        self.border_touch_max = 0.95

        self.crop_dict = {}

    def transfer_to_recognition(self, boxes: Results, frame: np.array, frame_no: int) -> None:
        for box in boxes:
            track_id = -1
            if box.cls.item() in [2, 3, 4, 5]:
                if box.xywhn.tolist()[0][2] > self.minimum_width_thres:
                    if box.is_track:
                        crop = save_one_box(
                            box.xyxy,
                            frame.copy(),
                            gain=1.0,
                            pad=0,
                            save=False
                        )
                        track_id = int(box.id.item())
                        width_square = ((box.xywh[0][2]*box.xywh[0][3])*(box.xywh[0][2]/box.xywh[0][3])).item()
                        x1, y1, x2, y2 = (box.xyxyn[0][0]).item(), (box.xyxyn[0][1]).item(), (box.xyxyn[0][2]).item(), (box.xyxyn[0][3]).item()
                        # When the bounding box touch to the frame limits, border_touch flag is added
                        border_touch = True if ((x1 < self.border_touch_min) or (y1<self.border_touch_min) or (x2>self.border_touch_max) or (y2>self.border_touch_max)) else False 

                        if track_id in self.crop_dict.keys():
                            if (width_square > self.crop_dict[track_id][1]):
                                if border_touch == False:
                                    self.crop_dict[track_id] = {0: frame_no, 1: width_square, 2: border_touch, 3: crop, 4: self.crop_dict[track_id][4]}
                        else:
                            self.crop_dict[track_id] = {0: frame_no, 1: width_square, 2: border_touch, 3: crop, 4: time.time()}
            if track_id != -1:
                self.crop_dict[track_id] = {0: frame_no, 1: self.crop_dict[track_id][1], 2: self.crop_dict[track_id][2], 3: self.crop_dict[track_id][3], 4: self.crop_dict[track_id][4]}
        
        removed_keys = []
        for key in self.crop_dict.keys():
            # Adding to recognition when the frame object is went of the frame or after 5 seconds stayed on frame
            if (self.crop_dict[key][0] < (frame_no-self.frame_limit)) or ((time.time() - self.crop_dict[key][4]) > 5.0):
                # Add to a queue for recognition
                removed_keys.append(key)
        for k in removed_keys:
            del self.crop_dict[k]

    def run_engine(self, ):
        frame_no = 0
        frame_counter = 0
        full_time = 0
        start_time = time.time()
        last_read_frame_id = -1

        # Loop through the video frames
        while self.cam_stream.get_status():
            t = time.time()

            # Get frame from stream
            frame_dict = self.cam_stream.get_frame()
            frame_no, frame = frame_dict[0], frame_dict[1]

            if last_read_frame_id != frame_no:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                preds = self.tracker.track(frame)

                # Visualize the results on the frame
                # if self.visualize:
                #     annotated_frame = preds.plot()

                # Transfer results to recognition
                self.transfer_to_recognition(preds.boxes, frame.copy(), frame_no)
                last_read_frame_id = frame_no

                if self.visualize:
                    visualize_frame = cv2.resize(frame, 
                                            (int(frame.shape[1]/self.resize), 
                                            int(frame.shape[0]/self.resize)))
                    # visualize_frame = cv2.resize(annotated_frame, 
                    #                         (int(annotated_frame.shape[1]/self.resize), 
                    #                         int(annotated_frame.shape[0]/self.resize)))
            else:
                if self.visualize:
                    frame_dict = self.cam_stream.get_frame()
                    frame = frame_dict[1]
                    visualize_frame = cv2.resize(frame, 
                                                (int(frame.shape[1]/self.resize), 
                                                int(frame.shape[0]/self.resize)))

            if self.visualize:
                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", visualize_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            stop_time = time.time()
            frame_counter += 1
            full_time += stop_time-start_time
            start_time = stop_time

            if full_time > 1.0:
                # print(f'FPS: {frame_counter}')
                frame_counter = 0 
                full_time = 0
            
            # Reduce the frame rate
            time_diff = time.time() - t
            if (time_diff < 1.0/(self.frame_limit)): time.sleep( 1.0/(self.frame_limit) - time_diff )

        # Release the video capture object and close the display window
        cv2.destroyAllWindows()
