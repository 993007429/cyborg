import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
sys.path.insert(0, '..')

import cv2 as cv
import torch
from skimage import io
from models.detr import build_model
from src.cell_det_cls import cal_pdl1_np
from Slide.dispatch import openSlide
import argparse
import numpy as np


if __name__ == '__main__':
    # 测试代码
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--slide_path', type=str, default="/data1/Caijt/PDL1_Parallel/A080 PD-L1 V+.kfb",
                        help='Slide Path')
    parser.add_argument('--roi_coords', type=str, nargs=2, default=None)

    # * Model
    parser.add_argument('--num_classes', type=int, default=10, help="Number of cell categories")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")

    args = parser.parse_args()


    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')
    # 加载模型
    net_weightspath = os.path.join(model_dir, 'pdl1_20220721.pth')
    net_weights_dict = torch.load(net_weightspath, map_location=lambda storage, loc: storage)
    net = build_model(args)
    net.load_state_dict(net_weights_dict)

    device = 0
    net.cuda(device)
    net.eval()

    # slide = openSlide(r'D:\迈杰PDL1已分析\30301-F3749-IHC.kfb')

    # patch_img = slide.read((5000 , 5000), (1024, 1024), 1)

    # for ax in range(10):
    #
    #     patch_img = slide.read((66696+100*(ax-5), 55106), (1024,1024), 1)
    #
    ax = 12
    patch_img = io.imread(r'D:\test_pdl1\0_34816.png')
    mean, std = np.load('mean_std.npy')
    # import pdb; pdb.set_trace()
    points, labels, pd_probs = cal_pdl1_np(patch_img, mean, std, net, device)
    print(points)

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),(0, 255, 0), (0, 0, 255)]
    for (x, y), cls in zip(points.astype(int), labels.astype(int)):
        cv.circle(patch_img, (x, y), radius=4, color=colors[cls], thickness=-1, lineType=cv.LINE_AA)
    io.imsave(f'result{ax}.png', patch_img)
