import torch
import os
class Classify_Wsi():
    def __init__(self, weight, input_size=768, device=None, half=False):
        self.device = device if device is not None else self.select_device()
        self.model = torch.jit.load(weight, map_location=self.device).eval()

        self.model(torch.zeros(1,128,input_size).to(self.device).type_as(next(self.model.parameters())))
        self.half = half and self.device.type != 'cpu'
        if self.half:
            self.model.half()


    def select_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, infer_input: torch.Tensor()):
        assert torch.is_tensor(infer_input), "infer_input is not a torch.Tensor"

        infer_input = infer_input.to(self.device)
        infer_input = infer_input.half() if self.half else infer_input.float()
        with torch.no_grad():
            return self.model(infer_input)

if __name__ == '__main__':
    from network.wsinet import denseWsiNet
    from dis_train_with_cell import opts
    from network.factory import get_network
    from utils import load_model
    opt = opts()
    opt.save_dir = "./results/2_3wsi_classify_with_cell/models"

    device = torch.device('cuda')

    if opt.network_way == "Resnet_features":
        network = get_network(opt.network_way)(pretrained=True)
        classifier = get_network("classifier")(num_classes=2).to(device).eval()
        classifier = load_model(classifier, os.path.join(opt.load_dir, 'classifier_best.pth'), None,opt.resume, opt.lr, opt.lr_step)
    else:
        network = get_network(opt.network_way)(opt)
    network = network.to(device).eval()

    network = load_model(network, os.path.join(opt.load_dir, 'backbone_best.pth'), None, opt.resume, opt.lr,
                               opt.lr_step)

    model_wsi = denseWsiNet(class_num=opt.num_classes, in_channels=256, use_self='global')
    model_wsi = load_model(model_wsi, os.path.join(opt.save_dir, 'wsi_best.pth'), None, opt.resume, opt.lr,
                               opt.lr_step)
    model_wsi = model_wsi.to(device).eval()

    det_outputput = torch.load("/home/data/kww/datasets/wsi_classify/train/5_3715.pt").to(device)
    det_outputput = (det_outputput / 255.0 - 0.5) / 0.5

    with torch.no_grad():
        fm = network(det_outputput)
        ts = torch.jit.trace(network, det_outputput, strict=False)
        ts.save(os.path.join(opt.save_dir, 'fm.torchscript'))

        probs = classifier(fm)
        ts = torch.jit.trace(classifier, fm, strict=False)
        ts.save(os.path.join(opt.save_dir, 'fc.torchscript'))

        fm = fm[:128]
        fm = torch.unsqueeze(fm, 0)
        wsi_pred1, _, attention_weigh1 = model_wsi(fm, None, None, None, False)
        ts = torch.jit.trace(model_wsi, fm, strict=False)
        ts.save(os.path.join(opt.save_dir, 'wsi_class.torchscript'))

        wsi_classfier = Classify_Wsi(os.path.join(opt.save_dir, 'wsi_class.torchscript'))
        wsi_pred, _, attention_weigh = wsi_classfier.infer(fm)
        import pdb;pdb.set_trace()
        a=10

