
import os
import torch
import string
import numpy as np
from typing import Tuple
from openvino.runtime import Core
from .ops_faster import *
from ultralytics.engine.results import Results

class YOLOv8Detector:
    def __init__(self, 
                 model_path: string,
                 cls_names: dict = {0: 'Person', 1: 'LP', 2: 'Car', 3: 'Motorcycle', 4: 'Truck', 5: 'Bus'}, 
                 threshold: int = 0.5, 
                 img_size: int = 640, 
                 device: string = 'CPU'
                 ) -> None:
        self.cls_names = cls_names
        self.conf_thres = threshold
        self.img_size = img_size
        self.device = device

        ie = Core()
        model_ir = ie.read_model(model=os.path.join(model_path, 'best.xml'), weights=os.path.join(model_path, 'best.bin'))
        self.model = ie.compile_model(
            model=model_ir, 
            device_name=device, 
            )
        self.input_layer_ir = self.model.input(0)

    def postprocess(self,
        pred_boxes:np.ndarray, 
        input_hw:Tuple[int, int], 
        orig_img:np.ndarray, 
        min_conf_threshold:float = 0.25, 
        nms_iou_threshold:float = 0.5, 
        agnosting_nms:bool = False, 
        max_detections:int = 300,
    ):
        """
        YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
        Parameters:
            pred_boxes (np.ndarray): model output prediction boxes
            input_hw (np.ndarray): preprocessed image
            orig_image (np.ndarray): image before preprocessing
            min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
            nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
            max_detections (int, *optional*, 300):  maximum detections after NMS
            pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
            retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
        Returns:
        pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and
                                            segment - segmentation polygons for each element in batch
        """
        nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}

        preds = non_max_suppression(
            (pred_boxes),
            min_conf_threshold,
            nms_iou_threshold,
            nc=len(self.cls_names),
            **nms_kwargs
        )
        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue
            pred[:, :4] = scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
        return results
    
    def detect(self, image: np.array) -> Results:
        preprocessed_image = processing_image(image)
        with torch.no_grad():
            result = self.model(preprocessed_image)
        boxes = result[self.model.output(0)]

        input_hw = preprocessed_image.shape[2:]
        detections = self.postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)[0]
        bboxes_ = detections['det'].tolist() if (type(detections['det']) != list) else detections['det']

        boxes = torch.tensor(bboxes_) if len(bboxes_)!=0 else torch.empty((0, 7), dtype=torch.float32)

        detections = Results(orig_img=image, 
                             path='', names=self.cls_names,
                             boxes=boxes, )
        
        return detections
    