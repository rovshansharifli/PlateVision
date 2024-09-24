from det.camera import Camera
from det.vehicle_detection import VehicleDetection

VIDEO_PATH = 'videos/test_slow.mp4'
# VIDEO_PATH = 'videos/test_fast.mp4'
# VIDEO_PATH = 'rtsp://localhost:8556/stream'
# VIDEO_PATH = '0'

VID_STRIDE = 1 # 3
VISUALIZATION = True
VISUALIZATION_RESIZE = 2
VEH_DET_MODEL_PATH = 'models/veh_det/best_pruned75_int8_openvino'
WRONG_RECOG_SAVE_PATH = 'videos/aeroport_yolu_detection_try25'
DEVICE = 'CPU'


if __name__ == '__main__':
    cam_stream = Camera(camera_url=VIDEO_PATH, vid_stride=VID_STRIDE)
    cam_stream.start_thread()

    detection = VehicleDetection(model_path=VEH_DET_MODEL_PATH, resize=VISUALIZATION_RESIZE, 
                                 visualize=VISUALIZATION, verbose=False, cam_stream=cam_stream,
                                 device=DEVICE)
    detection.run_engine()
