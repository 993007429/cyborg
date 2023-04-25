import numpy as np
import torch
import cv2


def crop_padd(img, height=64, width=64, color='black'):
    if color == 'black':
        output = np.zeros((height, width, 4), dtype=np.uint8)
    else:
        output = np.zeros((height, width, 4), dtype=np.uint8)
        output[..., :3] = np.ones((height, width, 3), dtype=np.uint8)
        output *= 255

    h, w, _ = img.shape
    center_y, center_x = h // 2, w // 2
    new_h, new_w = min(h, height), min(w, width)

    output_center_y, output_center_x = output.shape[:2]
    output_center_y //= 2
    output_center_x //= 2
    output[output_center_y - new_h // 2:output_center_y - new_h // 2 + new_h,
    output_center_x - new_w // 2:output_center_x - new_w // 2 + new_w, :-1] = \
        img[center_y - new_h // 2:center_y - new_h // 2 + new_h, center_x - new_w // 2:center_x - new_w // 2 + new_w]
    output[output_center_y - new_h // 2:output_center_y - new_h // 2 + new_h,
    output_center_x - new_w // 2:output_center_x - new_w // 2 + new_w, -1] = 1

    return output

def crop(img,height=224,width=224):
    h,w,_ = img.shape
    crop_h = max(h-height,0)//2
    crop_w = max(w-width,0)//2
    return img[crop_h:crop_h+height,crop_w:crop_w+width]
def image_transfer(img,max_shape=(224,224),padding_color="white"):
    patch_size = 16
    h,w = img.shape[1:]
    w_ = w//patch_size*patch_size+(w%patch_size!=0)*patch_size
    h_ = h // patch_size * patch_size + (h % patch_size != 0) * patch_size
    if padding_color == "black":
        tmp_img = np.zeros((3,h_,w_))
    else:
        tmp_img = np.ones((3,h_,w_))

    tmp_img[:,:h,:w] = img
    tmp_img = torch.Tensor(tmp_img)
    img = tmp_img.view(3,h_//patch_size,patch_size,w_//patch_size,patch_size).permute(0,1,3,2,4).contiguous()\
        .view(3,h_//patch_size*w_//patch_size,patch_size,patch_size)
    max_length = (max_shape[0]//patch_size)*(max_shape[0]//patch_size)
    mask = torch.zeros(max_length+1)
    samples = torch.zeros(3,max_length,patch_size,patch_size)

    samples[:,:h_//patch_size*w_//patch_size] = img

    mask[:h_//patch_size*w_//patch_size+1] = 1
    return samples,mask




class Classify_Cell():
    def __init__(self, fm_weight,fc_weight, input_size=64, device=None, half=False):
        self.device = device if device is not None else self.select_device()
        self.fm_model = torch.jit.load(fm_weight, map_location=self.device)
        self.fc_model = torch.jit.load(fc_weight, map_location=self.device)
        input_data_ = torch.zeros(1, 3, 224, 224).to(self.device).type_as(next(self.fm_model.parameters()))

        self.fc_model(self.fm_model(input_data_))

        self.half = half and self.device.type != 'cpu'
        if self.half:
            self.fm_model.half()
            self.fc_model.half()
        self.fm_model.eval()
        self.fc_model.eval()

    def select_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, infer_input: torch.Tensor(),mask=None, normal=True, conf_thres=0.001):
        assert torch.is_tensor(infer_input), "infer_input is not a torch.Tensor"

        infer_input = infer_input.to(self.device)
        infer_input = infer_input.half() if self.half else infer_input.float()
        infer_input = (infer_input / 255.0 - 0.5)/0.5 if normal else infer_input
        with torch.no_grad():
            if mask is not None:
                fm = self.fm_model(infer_input, mask)
            else:
                fm = self.fm_model(infer_input)

            probs = self.fc_model(fm)

        return fm,probs


class Classify_Cell1():
    def __init__(self, cell_cls_weight, input_size=64, device=None, half=False):
        print(cell_cls_weight)
        self.device = device if device is not None else self.select_device()
        self.cell_model = torch.jit.load(cell_cls_weight, map_location=self.device)
        input_data_ = torch.zeros(1, 3, 224, 224).to(self.device).type_as(next(self.cell_model.parameters()))
        self.cell_model(input_data_)

        self.half = half and self.device.type != 'cpu'
        if self.half:
            self.cell_model.half()
        self.cell_model.eval()

    def select_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, infer_input: torch.Tensor(),mask=None, normal=True, conf_thres=0.001):
        assert torch.is_tensor(infer_input), "infer_input is not a torch.Tensor"

        infer_input = infer_input.to(self.device)
        infer_input = infer_input.half() if self.half else infer_input.float()
        # import pdb; pdb.set_trace()
        print(infer_input.size())
        with torch.no_grad():
            fm, probs = self.cell_model(infer_input)

        return fm[:,:256], probs

if __name__ == '__main__':

    from dist_train import opts
    from network.factory import get_network
    from utils import load_model
    import os
    with torch.no_grad():

        opt = opts()
        network = get_network(opt.network_way)(pretrained=True)
        device = torch.device('cuda')
        classifier = get_network("classifier")(opt).eval().to(device)
        classifier = load_model(classifier, os.path.join(opt.save_dir, 'classifier_best.pth'), None,opt.resume, opt.lr, opt.lr_step)
        network = get_network(opt.network_way)(opt).eval().to(device)
        network = load_model(network, os.path.join(opt.save_dir, 'backbone_best.pth'), None, opt.resume, opt.lr,
                             opt.lr_step)

        # convert pt 2 torchscript
        # a = torch.zeros(1, 3, 64, 64).to(device)
        # tmp = network(a)
        # ts = torch.jit.trace(network, a, strict=False)
        # ts.save(os.path.join(opt.save_dir, 'backbone_best.pth').replace(".pth",".torchscript"))
        #
        # print(a.shape)
        # print(tmp.shape)
        #
        # gg= classifier(tmp)
        # ts = torch.jit.trace(classifier, tmp, strict=False)
        # ts.save(os.path.join(opt.save_dir, 'classifier_best.pth').replace(".pth",".torchscript"))

        # test torchscript
        # fm_weight = os.path.join(opt.save_dir, 'backbone_best.pth').replace(".pth",".torchscript")
        # fc_weight = os.path.join(opt.save_dir, 'classifier_best.pth').replace(".pth",".torchscript")
        # img_path = "/data5/kww/datasets/tct_classification/pos_pt/actinomycetes&1&11&1_11662_41650.pt"
        # imgs = torch.load(img_path,map_location=torch.device('cuda'))
        # cell_classify = Classify_Cell(fm_weight,fc_weight,64)
        # fm,probs = cell_classify.infer(imgs)
        #
        # imgs = (imgs / 255.0 - 0.5) / 0.5
        # fm1 = network(imgs)
        # probs = classifier(fm1)
        # print(fm[0, :10])
        # print(fm1[0, :10])
        # import pdb;pdb.set_trace()
        #
        # a=10



