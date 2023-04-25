import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import random
from .utils import box_iou

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300,shape=(320,320)):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output = [torch.zeros((0, 4), device=prediction.device)] * prediction.shape[0]
    output_conf = [torch.zeros((0), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS ; index 5存的类别index,
        # agnostic 会让同一类别计算nms
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        # output[xi] = x[i]
        boxes = boxes[i]
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2

        # TODO 添加拓边
        boxes_w = boxes[:,2]-boxes[:,0]
        boxes_h = boxes[:,3]-boxes[:,1]
        _xmin = torch.from_numpy(np.array([ random.randint(-5, 5)/100. for _ in range(len(boxes_w))])).to(boxes.device)
        _xmax = torch.from_numpy(np.array([random.randint(-5, 5) / 100. for _ in range(len(boxes_w))])).to(boxes.device)
        _ymin = torch.from_numpy(np.array([random.randint(-5, 5) / 100. for _ in range(len(boxes_w))])).to(boxes.device)
        _ymax = torch.from_numpy(np.array([random.randint(-5, 5) / 100. for _ in range(len(boxes_w))])).to(boxes.device)
        boxes[:,0] += boxes_w*_xmin
        boxes[:, 2] += boxes_w * _xmax
        boxes[:, 1] += boxes_h * _ymin
        boxes[:, 3] += boxes_h * _ymax
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2

        output[xi] = boxes
        output_conf[xi] = scores[i]
    return output,torch.cat(output_conf)


class Dataset(data.Dataset):
    def __init__(self,slide,crop_coords,scale,input_shape=(1024,1024),RANK=-1):
        self.slide = slide
        self.crop_coords = crop_coords
        self.nF = len(self.crop_coords)
        self.scale = scale
        self.input_shape = input_shape
    def __len__(self):
        return self.nF
    def __getitem__(self, index):
        xmin,ymin,xmax,ymax = self.crop_coords[index]
        h,w = ymax-ymin,xmax-xmin
        try:
            cur_region = np.array(self.slide.read((xmin, ymin), (w, h), self.scale))  # RGB
        except:
            cur_region = np.ones((*self.input_shape, 3), dtype=np.uint8)
        input_img = np.ones((*self.input_shape,3),dtype=np.uint8)
        h,w,_ = cur_region.shape
        h = min(self.input_shape[0],h)
        w = min(self.input_shape[1], w)
        input_img[:h,:w] = cur_region[:h,:w]
        input_img = input_img.transpose((2, 0, 1))
        input_img = np.ascontiguousarray(input_img)
        ret = {'cur_region': input_img,'tile_coords': np.array([xmin,ymin,xmax,ymax])}
        return ret

class Detect_Cell():
    def __init__(self,weights,input_size=320,device=None,half=True):
        self.device = device if device is not None else self.select_device()
        # from models.experimental import attempt_load
        # self.model = attempt_load(weights, map_location=self.device)
        self.model = torch.jit.load(weights, map_location=self.device)
        self.half = half and self.device.type!='cpu'
        if self.half:
            self.model.half()
        self.model.eval()
        self.input_size = (input_size,input_size)
        self.model(torch.zeros(1, 3, *(input_size,input_size)).to(self.device).type_as(next(self.model.parameters())))  # run once
    def select_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def infer(self,infer_input:torch.Tensor(),normal=True,conf_thres=0.1,iou_thres=0.6):

        assert torch.is_tensor(infer_input), "infer_input is not a torch.Tensor"

        infer_input = infer_input.to(self.device)
        infer_input = infer_input.half() if self.half else infer_input.float()
        infer_input = infer_input/255.0 if normal else infer_input

        pred = self.model(infer_input)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000,shape=self.input_size)
        # pred [(n,6)] the len of [] is bs
        return pred

if __name__ == '__main__':
    if True: # test model
        from models.experimental import attempt_load
        weights = "/home/data/kww/yolov5/runs/train/exp/3/weights/best_1024.torchscript"
        detector1= Detect_Cell(weights,input_size=1024)

        pt_weights = "/home/data/kww/yolov5/runs/train/exp/3/weights/best.pt"
        model = attempt_load(pt_weights).to(torch.device('cuda'))
        a = torch.zeros(1, 3, 1024,1024).to(torch.device('cuda'))
        model(a) # 必须先过一次模型~修改一次grid先

        import cv2
        img_path = "/home/data/kww/datasets/cell_detection/deploy/91.jpg"
        img = cv2.imread(img_path)
        img = letterbox(img, (1024,1024))[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = img[np.newaxis,...]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(torch.device('cuda'))
        # img = torch.load("/home/data/kww/datasets/cell_detection/deploy/tmp.pt").to(torch.device('cuda'))
        img = img/255.0
        pred1 = detector1.infer(img,normal=False,conf_thres=0.01)
        pred2 = model(img)[0]
        pred2 = non_max_suppression(pred2,0.01,0.6,classes=None, agnostic=False, max_det=1000,shape=(1024,1024))

        import pdb;pdb.set_trace()
    if False:
        # convert pt 2 torchscript
        weights1 = "/home/data/kww/yolov5/runs/train/exp/3/weights/best.pt"
        from models.experimental import attempt_load
        model = attempt_load(weights1).to(torch.device('cuda'))
        a = torch.zeros(1, 3, 1024,1024).to(torch.device('cuda'))
        tmp = model(a) # 必须先过一次模型~修改一次grid先
        ts = torch.jit.trace(model, a, strict=False)
        ts.save(weights1.replace(".pt","_1024.torchscript"))

        # test torchscript
        # weights1 = "/home/data/kww/yolov5/runs/train/exp/3/weights/best.pt"
        # weights2 = weights1.replace(".pt",".torchscript")
        # model = attempt_load(weights1).to(torch.device('cuda'))
        # model2 = torch.jit.load(weights2,map_location=torch.device('cuda'))
        # a = torch.zeros(3, 3, 1024,1024).to(torch.device('cuda'))
        # import pdb
        # pdb.set_trace()