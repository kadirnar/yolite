from models.common import DetectMultiBackend
from utils.general import cv2, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.datasets import file_to_torch, numpy_img
import logging


def load_model(weights, device, data):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    return model


def preprocces_img(img, imgsz, device):
    npy_im = numpy_img(img, imgsz)
    tensor_im = file_to_torch(npy_im, device)
    return tensor_im, npy_im


def detect(model, tensor_im, npy_im):
    # Inference
    pred = model(tensor_im)  # shape: torch.Size([1, 3, 640, 480])

    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

    for det in pred:
        annotator = Annotator(npy_im, line_width=3, example=str(model.names))
        det[:, :4] = scale_coords(tensor_im.shape[2:], det[:, :4], npy_im.shape).round()

    return det, annotator


def show_img(model, det, annotator, view_img=True):
    # Write results
    for *xyxy, conf, cls in reversed(det):
        logging.info("\t+ Label: %s, Conf: %.5f" % (model.names[int(cls)], conf.item()))
        if view_img:  # Add bbox to image
            c = int(cls)  # integer class
            label = "%s %.2f" % (model.names[c], conf)
            annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    im0 = annotator.result()
    if view_img:
        cv2.imshow("frame", im0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
