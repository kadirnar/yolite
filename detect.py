import torch
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import cv2,non_max_suppression,scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import numpy as np


weights = "yolov5s.pt"
source = "data/images/bus.jpg"
data = "data/coco128.yaml"
device = "cpu"
imgsz = (640, 640)
view_img = True


# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data)


def resize_image(image, long_size, interpolation=cv2.INTER_LINEAR):
    height, width, channel = image.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    return image


def file_to_torch(img, size):
    img = cv2.imread(img)
    img = resize_image(img, size)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = img.copy()
    im = torch.from_numpy(img)
    im = im.float().div(255.0).to(device)
    return im


im = file_to_torch(source, 640)

# Inference
pred = model(im)  # shape: torch.Size([1, 3, 640, 480])

# NMS
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

# Process predictions
for i, det in enumerate(pred):  # per image
    im0 = im0s.copy()
    annotator = Annotator(im0, line_width=3, example=str(model.names))

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            if view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = "%s %.2f" % (model.names[c], conf)
                annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    im0 = annotator.result()
    if view_img:
        cv2.imshow("frame", im0)
