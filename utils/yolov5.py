from models.common import DetectMultiBackend
from utils.general import cv2, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.datasets import file_to_torch, numpy_img
import logging


class Yolov5:
    def __init__(self, weights, device, data):
        self.weights = weights
        self.device = device

    def load_model(self, weights, device, data):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, data=data)
        
    def preprocces_img(self, img, imgsz):
        self.npy_im = numpy_img(img, imgsz)
        self.tensor_im = file_to_torch(self.npy_im, self.device)

    def detect(self):
        # Inference
        pred = self.model(self.tensor_im) # shape: torch.Size([1, 3, 640, 480])

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
        for det in pred:
            det[:, :4] = scale_coords(self.tensor_im.shape[2:], det[:, :4], self.npy_im.shape).round()
        
        self.det = det

    def show_img(self, view_img=True):
        # Write results
        for *xyxy, conf, cls in reversed(self.det):
            annotator = Annotator(self.npy_im, line_width=3, example=str(self.model.names))
            logging.info("\t+ Label: %s, Conf: %.5f", self.model.names[int(cls)], conf.item())
            if view_img:  # Add bbox to image
                label = "%s %.2f" % (self.model.names[int(cls)], conf)
                annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Stream results
        im0 = annotator.result()
        if view_img:
            cv2.imshow("frame", im0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
