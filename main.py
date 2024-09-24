import cv2
from det.camera import Camera

VIDEO_PATH = 'videos/test_slow.mp4'

camera = Camera(VIDEO_PATH, vid_stride=1)
camera.start_thread()

while camera.get_status():
    frame_dict = camera.get_frame()
    frame_no, frame = frame_dict[0], frame_dict[1]

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
