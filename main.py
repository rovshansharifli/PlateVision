from det.camera import Camera
from multiprocessing import Process, Queue
from det.vehicle_detection import VehicleDetection
# from rec.plate_recognition import PlateRecognition
from rec.plate_recognition_rule_based import PlateRecognition

VIDEO_PATH = 'videos/test_1.mp4'
# VIDEO_PATH = 'rtsp://localhost:8556/stream'
# VIDEO_PATH = '0'
SAVE_PATH = 'videos/results'

VID_STRIDE = 1 # 3
VISUALIZATION = True
VISUALIZATION_RESIZE = 2
VEH_DET_MODEL_PATH = 'models/veh_det/best_pruned75_int8_openvino'
DEVICE = 'CPU'

def run_detection_engine(queue: Queue):

    if DEVICE == 'CPU':
        model_path = VEH_DET_MODEL_PATH
    else:
        model_path = ''

    cam_stream = Camera(camera_url=VIDEO_PATH, vid_stride=VID_STRIDE)
    cam_stream.start_thread()

    detection = VehicleDetection(model_path=model_path, resize=VISUALIZATION_RESIZE, 
                                 visualize=VISUALIZATION, verbose=False, cam_stream=cam_stream,
                                 device=DEVICE, shared_queue = queue, img_size=640)
    detection.run_engine()

def run_recognition_engine(queue: Queue):
    recognition = PlateRecognition(shared_queue=queue, 
                                   save_crop=False, 
                                   save_path=SAVE_PATH)
    recognition.run_engine()

if __name__ == '__main__':
    queue = Queue()
    detection_process = Process(target=run_detection_engine, args=(queue,))
    recognition_process = Process(target=run_recognition_engine, args=(queue,))

    detection_process.start()
    recognition_process.start()

    detection_process.join()
    recognition_process.join()
