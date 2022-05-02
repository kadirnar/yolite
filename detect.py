from helpers import *

weights = "yolov5s.pt"
img = "data/images/bus.jpg"
data = "data/coco128.yaml"
device = "cpu"
imgsz = 640
view_img = True


model = load_model(weights, device, data)

tensor_im, npy_im = preprocces_img(img, imgsz, device)

det, annotator = detect(model, tensor_im, npy_im)

show_img(model, det, annotator, view_img=True)
