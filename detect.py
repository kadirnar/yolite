from helpers import *

weights = "yolov5s.pt"
img = "data/images/bus.jpg"
data = "data/coco128.yaml"
device = "cpu"
imgsz = 640
view_img = True


model = Yolov5(weights, device, data)
model.preprocces_img(img, imgsz)
model.detect(data)
model.show_img(model, model.det, view_img)