from paddleocr import PaddleOCR

class PaddleOcrRead:

    def __init__(self):
        self.ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False, 
                             rec_char_dict_path='allowed_chars.txt',
                             det_model_dir = 'models/paddle_det/det/en/en_PP-OCRv3_det_infer',
                             rec_model_dir = 'models/paddle_det/rec/en/en_PP-OCRv4_rec_infer',
                             det_limit_side_len=640, precision='int8',
                             )
        
    def run(self, crop_img):
        return self.ocr.ocr(crop_img, cls=False)
