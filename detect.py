import torch
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, cv2, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


weights='yolov5s.pt'
source= 'data/images/bus.jpg'
data='data/coco128.yaml'
device='cpu'
view_img=True


# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data)
names, pt =  model.names, model.pt
imgsz = check_img_size(640,640)  # check image size


# Run inference
dataset = LoadImages(source, img_size=imgsz, auto=pt)
for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im)

    # NMS
    pred = non_max_suppression(pred, conf_thres= 0.25, iou_thres= 0.45, max_det=1000)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        annotator = Annotator(im0, line_width=3, example=str(names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = '%s %.2f' % (names[c], conf)
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        im0 = annotator.result()
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(0)  # 1 millisecond
