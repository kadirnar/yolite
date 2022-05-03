<div align="center">
<h1>
  Yolov5-Lite: Minimal YoloV5 Implementation
</h1>
<img src="doc/readme_yolov5.png" alt="Yolov5-Lite" width="800">
</div>

## <div align="center">Overview</div>

It has been simplified by editing detect.py in the yolov5 repository.

### Installation

```
git clone https://github.com/kadirnar/yolov5-lite
cd yolov5-lite
pip install -r requirements.txt
```

## Yolov5-Lite Prediction: 
It is the edited version of the codes in the detect file.
```
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
            logging.info("\t+ Label: %s, Conf: %.5f" % (self.model.names[int(cls)], conf.item()))
            if view_img:  # Add bbox to image
                label = "%s %.2f" % (self.model.names[int(cls)], conf)
                annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Stream results
        im0 = annotator.result()
        if view_img:
            cv2.imshow("frame", im0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
```

## Yolov5-Lite Run Code:
You can take the detect.py file as an example to load and visualize your yolov5 models.

```
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
```

### Reference:

 - [YOLOv5](https://github.com/ultralytics/yolov5)
 - [YOLOv5-Pip](https://github.com/fcakyon/yolov5-pip)

