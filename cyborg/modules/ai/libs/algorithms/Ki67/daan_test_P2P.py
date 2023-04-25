import os, sys
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ki67_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,proj_root)
sys.path.insert(0,ki67_root)
os.environ['KMP_DUPLICATE_LIB_OK']="True"
import numpy as np
from Slide.dispatch import openSlide
import cv2
from utils import threading_classification, count_test_summary
import pandas as pd
from models.p2p_head import P2PNet
from models.backbone import ResNet34_BackBone
import torch
import time
from torch import nn
from Ki67.src.multi_cls_cell_seg_manual import generate_result_mask_P2P
from models.albu_unet import AlbuNet as detnet
from models.Network import resnet50
from glob import glob
from PIL import Image


class WSI_Tester:
    def __init__(self):
        self.sample_level = 0
        self.vis_level = 2
        self.resize_ratio = 2 ** (self.vis_level - self.sample_level)
        self.patch_size = 1536
        self.overlap = 0
        self.batch_size = 64
        self.n_classes = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        self.selected_patches = 30
        self.truncated_ratio = 0.6
        self.radius = 4
        self.thickness = -1
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.out_path = './results2/'
        os.makedirs(self.out_path, exist_ok=True)

        # hyperparameters of cell counting
        self.label_dict = {'阴性肿瘤': 0, '阳性肿瘤': 1, '阴性淋巴': 2, '阳性淋巴': 3, '纤维细胞': 4, '其它细胞': 5}
        self.color_dict = {0: (0, 255, 255), 1: (0, 0, 255), 2: (26, 26, 139), 3: (0, 255, 0), "0": (0, 0, 255), "1": (255, 255, 0), "2": (0, 255, 0), "3": (0, 0, 255),
                      "4": (255, 0, 0), "5": (255, 0, 0), "6": (128, 0, 255)}

        # load cell detection model
        # self.detnet = detnet(input_channels=3, num_classes=7, pretrained=False)
        # model_dir = os.path.join(os.path.dirname(__file__), 'Model')
        # net_weightspath = "F1_best_weight_956_0.546832606553227.pth"
        # net_weights_dict = torch.load(os.path.join(model_dir, net_weightspath), map_location=lambda storage, loc: storage)
        # self.detnet.load_state_dict(net_weights_dict)
        # self.detnet = nn.DataParallel(self.detnet).to(self.device)
        # self.detnet.eval()
        BackBone = ResNet34_BackBone(pretrained=False)
        self.detnet = P2PNet(backbone=BackBone, num_classes=7)
        model_dir = os.path.join(os.path.dirname(__file__), 'Model')
        net_weights_dict = torch.load(os.path.join(model_dir, 'F1_best_weight_956_0.5468326065532271.pth'), map_location=lambda storage, loc: storage)
        self.detnet.load_state_dict(net_weights_dict)
        self.detnet.cuda(self.device)
        self.detnet.eval()

        # load classification checkpoint
        self.cnet = resnet50(pretrained=False, classes=5)
        self.cnet = nn.DataParallel(self.cnet).to(self.device)
        net_weightspath = 'resnet50_save1.pth'
        checkpoint = torch.load(os.path.join(model_dir, net_weightspath))
        self.cnet.load_state_dict(checkpoint['Classifier'])
        self.cnet.eval()


    def _morohology_erode(self, pred):
        """
            delete the prediction in the edges and small regions.
            :param pred:
            :return: prediction after open operation.
        """
        full_pred = np.ones(self.W * self.H) * 3
        full_pred[self.indexs] = pred
        full_pred = full_pred.reshape(self.H, self.W)
        binary_pred = np.where(full_pred <= 2, 1, 0).astype(np.uint8)

        opening = cv2.morphologyEx(binary_pred, cv2.MORPH_OPEN, self.open_kernel, iterations=1)
        return np.where(opening == 1, pred, 3)


    def _collect_info(self, pred_result_summary, prop_n=[0,0,0]):
        df = pd.DataFrame(pred_result_summary)
        filtered_df = pd.DataFrame()
        if 3 in df['类别'].tolist():
            kth = len(df['肿瘤细胞']) - self.selected_patches
            thre = np.partition(np.array(df['肿瘤细胞']), kth)[kth]
            filtered_df = df.drop(df[df['肿瘤细胞'] < thre].index)
        else:
            for i in range(3):
                df_c = df.drop(df[df['类别'] != i].index)
                kth = len(df_c['肿瘤细胞']) - prop_n[i]
                if kth > 0:
                    thre = np.partition(np.array(df_c['肿瘤细胞']), kth-1)[kth-1]
                    df_f = df_c.drop(df_c[df_c['肿瘤细胞'] < thre].index)
                    filtered_df = pd.concat([filtered_df, df_f])
        # filtered_df.to_csv(os.path.join(self.out_path, 'summary_pred.csv'))
        ki67_total = 0
        pos_total = 0
        neg_total = 0
        for idx, row in filtered_df.iterrows():
            pos, neg, ki67 = row['阳性肿瘤'], row['阴性肿瘤'], row['ki67指数']
            ki67_total += ki67
            pos_total += pos
            neg_total += neg

        print('patch size Ki67 index: %.2f' % (ki67_total / filtered_df.shape[0]))
        print('cell size Ki67 index: %.2f' % (pos_total / (pos_total + neg_total)))
        return filtered_df


    def _calculate_ki67(self, cur_slide, selected_patches):
        pred_result_summary = {'slide name': [], '阴性肿瘤': [], '阳性肿瘤': [], '阴性淋巴': [],
                               '阳性淋巴': [], '纤维细胞': [], '其它细胞': [],
                               '细胞总数': [],'肿瘤细胞': [], 'ki67指数': [], '类别': [], '检测坐标': []}
        for cls, inds in selected_patches.items():
            for ind in inds:
                x, y = ind // self.H * self.patch_size, ind % self.H * self.patch_size
                ## calculation ##
                image = cur_slide.read((x, y),(self.patch_size, self.patch_size))
                center_coords_pred, labels_pred = generate_result_mask_P2P(image, self.detnet, patch_size=512, overlap=0)
                labels_pred = labels_pred - 1
                '''center_coords_distinguish, labels_distinguish is all you need'''

                # # Save ##
                # img = image.copy()
                # for label, center in zip(labels_pred, center_coords_pred):
                #     if label in self.color_dict:
                #         img = cv2.circle(img, (center[0], center[1]), 4, self.color_dict[label], -1)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(self.out_path + '/%s_%s_pred.png' % (x, y), img)
                pred_count_dict = count_test_summary(center_coords_pred, labels_pred, self.label_dict)
                pred_result_summary['slide name'].append(str(x) + '_' + str(y))
                for k, v in pred_count_dict.items():
                    pred_result_summary[k].append(pred_count_dict[k])
                pred_result_summary['类别'].append(cls)
                pred_result_summary['检测坐标'].append(np.concatenate((center_coords_pred,labels_pred[:,np.newaxis]), axis=1))

        return pred_result_summary

    def run(self, slide):
        """  classification evaluation phase  """
        start_time = time.time()
        excutor = threading_classification(slide, self.cnet, batch_size=64, device=self.device)
        excutor.execute()
        self.heatmap = excutor.heatmap
        self.W, self.H = excutor.W, excutor.H
        self.indexs = excutor.processed_index
        print("Hot spots selection processing time: %.2f" % (time.time() - start_time))

        """  selecting highly-confident patches  """
        pred = torch.argmax(self.heatmap, dim=1).cpu().numpy()
        # pred = self._morohology_erode(pred)
        prop = [np.sum(pred==i) for i in range(3)]
        print("The number of selected samples belongs to weak, median, strong: %s, %s, %s" % (
        prop[0], prop[1], prop[2]))

        # save image to check the classification results.
        Image.fromarray(excutor.read_region.transpose(1,0,2)).save(
            self.out_path + slide.filename.split('\\')[1].split('.')[0] + '_img.png')
        full_pred = np.ones(self.W * self.H) * 3
        full_pred[self.indexs] = pred
        Image.fromarray((np.eye(self.n_classes)[full_pred.reshape(self.W, self.H).astype(np.int)][:, :, :3][:, :,
                         ::-1] * 255).astype(np.uint8)).save(
            self.out_path + slide.filename.split('\\')[1].split('.')[0] + '_pred.png')

        selected_patches = {0:[], 1:[], 2:[], 3:[]}
        if self.truncated_ratio * sum(prop).item() <= self.selected_patches:
            print(
                'The selected patches are not confident, so we select %s patches most like tumor.' % self.selected_patches)
            tumor_confidence = torch.sum(self.heatmap[:, :3], dim=1)
            inds = torch.argsort(tumor_confidence, descending=True)
            for ind in inds[:3 * self.selected_patches:3]:
                selected_patches[3].append(self.indexs[ind.item()])

            """  calculating the ki67 index of selected patches  """
            pred_result_summary = self._calculate_ki67(slide,selected_patches)
            filtered_df = self._collect_info(pred_result_summary)

        else:
            """ The candidate samples in the 1-th round"""
            prop_n1 = [np.round(i / sum(prop) * (self.selected_patches+10)) for i in prop]
            """ The candidate samples in the 2-th round"""
            prop_n2 = [np.round(i / sum(prop) * self.selected_patches).astype(np.int) for i in prop]
            """ Select prop * self.truncated_ratio samples from the classification results. If this value is less than
            the prop_n1, select prop_n1. Otherwise, select v * self.truncated_ratio.
            """
            selected_prop = []
            for i, v in enumerate(prop):
                candi = v * self.truncated_ratio
                if candi < prop_n1[i]:
                    selected_prop.append(prop_n1[i].astype(np.int))
                else:
                    selected_prop.append(np.round(candi).astype(np.int))


            for cls in range(3):
                if prop_n1[cls] == 0 or selected_prop[cls] == 0:
                    continue
                heatmap_cls = self.heatmap[:, cls]
                inds = torch.argsort(heatmap_cls, descending=True)[
                       0:selected_prop[cls]:int(selected_prop[cls] / prop_n1[cls])]
                for ind in inds:
                    selected_patches[cls].append(self.indexs[ind.item()])

            """  calculating the ki67 index of selected patches  """
            pred_result_summary = self._calculate_ki67(slide, selected_patches)
            filtered_df = self._collect_info(pred_result_summary, prop_n2)
        return filtered_df

if __name__ == "__main__":
    pths = glob('./ki677/*.sdpc')
    pths.sort(key=lambda x: int(x.split('\\')[1][:9]))
    ki67_dict = {}
    for pth in pths:
        tic = time.time()
        print("slide name: %s"%pth)
        cur_slide = openSlide(pth)
        filtered_df = WSI_Tester().run(cur_slide)
        ki67_dict[pth.split('\\')[1][:-5]] = filtered_df['阳性肿瘤'].sum() / filtered_df['肿瘤细胞'].sum() * 100.0
        print(time.time()-tic)
    pd.DataFrame(ki67_dict.items()).to_csv('Daan_Ki67_prediction_P2P.csv')


