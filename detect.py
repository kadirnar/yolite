from utils.yolov5 import Yolov5

weights = "yolov5s.pt"
img = "data/images/bus.jpg"
data = "data/coco128.yaml"
device = "cpu"
imgsz = 640
view_img = True


model = Yolov5(weights, device, data)
model.load_model(weights, device, data)
model.preprocces_img(img, imgsz)
model.detect()
model.show_img(view_img)
