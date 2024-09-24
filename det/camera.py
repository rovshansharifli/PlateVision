import cv2 
import time
import string
import numpy as np
from threading import Thread

class Camera:
    def __init__(self, camera_url: string = '0', vid_stride: int = 1):

        """
        Camera class to handle camera and video stream. The stream will be 

        Attributes:
            camera_url (string): URL to a stream or video path.
            vid_stride (int): Video stride to get every n-th frame.

        Methods:
            start_thread: To start the camera thread.
            update: Run and update the current frame.
            get_frame: Moves the tensor to GPU memory, returning a new instance if necessary.
            stop: Stop the stream.
            get_status: Check the status of the stream.
            get_FPS_of_camera: Get FPS of the stream.
        """
        
        self.camera_url = camera_url if camera_url != '0' else 0
        self.vid_stride = vid_stride
        self.stopped = True

        # Start video capture
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            print(f'Error: Cannot access to the stream or video: {camera_url}')
            exit(0)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS of input stream: {}".format(self.fps))

        # Reading a single frame from stream for initialization 
        success, frame = self.cap.read()
        self.n_frame = 0
        if success:
            self.current_frame = {0: self.n_frame, 1: frame}
        else:
            print('Error: Cannot read frames')
            exit(0)

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start_thread(self, ) -> None: 
        self.stopped=False
        self.t.start()

    def update(self) -> None:
        while self.cap.isOpened():
            t = time.time()
            
            # Read a frame from the video
            self.cap.grab()
            self.n_frame+=1

            if self.stopped:
                break

            # Read every n-th frame
            if self.n_frame % self.vid_stride == 0:
                success, frame = self.cap.retrieve()
                self.current_frame = {0: self.n_frame, 1: frame}
                if not success:
                    print('No more frames to read')
                    self.stopped=True
                    break 

            # Fix the frame rate
            time_diff = time.time() - t
            if (time_diff < 1.0/(self.fps)): time.sleep( 1.0/(self.fps) - time_diff )

        self.cap.release()

    def get_frame(self) -> dict:

        return self.current_frame

    def stop(self) -> None:
        self.stopped=True

    def get_status(self) -> bool:
        return not self.stopped
    
    def get_FPS_of_camera(self,) -> float:
        return self.fps 
    