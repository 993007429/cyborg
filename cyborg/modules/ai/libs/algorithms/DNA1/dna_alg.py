import os

import math
import torch
import numpy as np

from cyborg.infra.oss import oss
from .src.detect_tct_disk import disc_detect
from .src.detect_nuclei import nuclei_det
from .src.utils import plot_dna
from .configs import DnaBaseConfig


class DnaAlgBase:
    def __init__(self):
        self.config = DnaBaseConfig()
        self.model = None
        self.result = {}

        self.patch_size = 1024
        self.standard_mpp = 0.242042 * 2

    def load_nuclei_detector(self, model_name='yolov7_1024', weights_name='YOLOv7_102451_epoch_124.torchscript'):
        det_model_file = oss.get_object_to_io(oss.path_join('AI', 'DNA1', 'Model', model_name, weights_name))
        model = torch.jit.load(det_model_file)
        model = model if not torch.cuda.is_available() else model.cuda()
        model.eval()
        if self.config.is_half:
            model.half()
            model(torch.zeros(1, 3, 1024, 1024).half().cuda())
        else:
            model(torch.zeros(1, 3, 1024, 1024).float().cuda())
        return model

    def dna_test(self, slide):
        self.model = self.load_nuclei_detector()
        x_coords, y_coords = disc_detect(slide, self.config.is_disc_detect)
        iods, areas, bboxes = nuclei_det(slide, self.model, self.config.is_half, self.standard_mpp, patch_size=1024,
                                         overlap=0, num_workers=5, batch_size=8, x_coords=x_coords, y_coords=y_coords)

        topk_num = math.ceil(0.15 * iods.shape[0])
        control_iod = np.mean(iods[:topk_num])
        nuclei_dna_index = (iods / control_iod)

        normal_idx = np.where(nuclei_dna_index.round(2) <= self.config.abnormal_low_thres)[0]
        abnormal_high_idx = np.where(nuclei_dna_index.round(2) >= self.config.abnormal_high_thres)[0]
        abnormal_low_idx = np.where(np.logical_and(nuclei_dna_index.round(2) > self.config.abnormal_low_thres,
                                                   nuclei_dna_index.round(2) < self.config.abnormal_high_thres))[0]

        # print(bboxes[abnormal_high_idx])
        for i in bboxes[abnormal_high_idx]:
            img_index = [i[1] // 2048 + 2, i[0] // 2048 - 2]
            # print(img_index)
        num_normal = normal_idx.size
        num_abnormal_high = abnormal_high_idx.size
        num_abnormal_low = abnormal_low_idx.size

        normal_ratio = num_normal / bboxes.shape[0]
        # print(normal_ratio)
        abnormal_high_ratio = num_abnormal_high / bboxes.shape[0]
        abnormal_low_ratio = num_abnormal_low / bboxes.shape[0]

        mean_di_normal = 0 if num_normal == 0 else np.mean(nuclei_dna_index[normal_idx])
        mean_di_abnormal_high = 0 if num_abnormal_high == 0 else np.mean(nuclei_dna_index[abnormal_high_idx])
        mean_di_abnormal_low = 0 if num_abnormal_low == 0 else np.mean(nuclei_dna_index[abnormal_low_idx])

        std_di_normal = 0 if num_normal == 0 else np.std(nuclei_dna_index[normal_idx])
        std_di_abnormal_high = 0 if num_abnormal_high == 0 else np.std(nuclei_dna_index[abnormal_high_idx])
        std_di_abnormal_low = 0 if num_abnormal_low == 0 else np.std(nuclei_dna_index[abnormal_low_idx])
        # import pdb; pdb.set_trace()

        diagnosis_dict = {'insufficient_nuclei': '有效检测细胞不足',
                          'no_abnormal_nucleus': '未见DNA倍体异常细胞',
                          'a_few_abnormal_nuclei': '可见少量DNA倍体异常细胞（1-2个）',
                          'plenty_of_abnormal_nuclei': '可见DNA倍体异常细胞（>=3个）',
                          'normal_proliferation': '可见少量细胞增生（5%-10%）',
                          'abnormal_proliferation': '可见细胞异常增生（>=10%）',
                          'abnormal_nuclei_peak': '可见异倍体细胞峰'
                          }

        if bboxes.shape[0] < 1000:
            dna_diagnosis = 'insufficient_nuclei'
        else:
            if num_abnormal_high == 0:
                if 0.05 < num_abnormal_low / iods.size < 0.1:
                    dna_diagnosis = 'normal_proliferation'
                elif num_abnormal_low / iods.size >= 0.1:
                    dna_diagnosis = 'abnormal_proliferation'
                else:
                    dna_diagnosis = 'no_abnormal_nucleus'

            elif 0 < num_abnormal_high < 3:
                dna_diagnosis = 'a_few_abnormal_nuclei'
            else:
                dna_diagnosis = 'plenty_of_abnormal_nuclei'

        plot_dna(nuclei_dna_index, areas, save_dir=os.path.dirname(slide.filename),
                 abnormal_high_thres=self.config.abnormal_high_thres, abnormal_low_thres=self.config.abnormal_low_thres)

        self.result['dna_statics'] = {
            'num_normal': num_normal,
            'num_abnormal_high': num_abnormal_high,
            'num_abnormal_low': num_abnormal_low,
            'normal_ratio': round(normal_ratio * 100, 3),
            'abnormal_high_ratio': round(abnormal_high_ratio * 100, 3),
            'abnormal_low_ratio': round(abnormal_low_ratio * 100, 3),
            'mean_di_normal': round(mean_di_normal, 3),
            'mean_di_abnormal_high': round(mean_di_abnormal_high, 3),
            'mean_di_abnormal_low': round(mean_di_abnormal_low, 3),
            'std_di_normal': round(std_di_normal, 3),
            'std_di_abnormal_high': round(std_di_abnormal_high, 3),
            'std_di_abnormal_low': round(std_di_abnormal_low, 3)
        }
        self.result['nuclei_bboxes'] = bboxes
        self.result['dna_index_values'] = nuclei_dna_index
        self.result['iod_values'] = iods
        self.result['area'] = areas
        self.result['control_iod'] = control_iod
        self.result['num_nuclei'] = bboxes.shape[0]
        self.result['num_normal'] = num_normal
        self.result['num_abnormal_low'] = num_abnormal_low
        self.result['num_abnormal_high'] = num_abnormal_high
        self.result['dna_diagnosis'] = dna_diagnosis

        return self.result


class DNA_1020(DnaAlgBase):
    def __init__(self):
        super(DNA_1020, self).__init__()
        self.model = self.load_nuclei_detector()
