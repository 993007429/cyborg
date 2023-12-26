import logging
import sys,os

import configparser
import subprocess
import json
import numpy as np
import time

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
current_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_root)

config_path = os.path.join(current_root,'config.ini')

logger = logging.getLogger(__name__)

label_dict = {
    '微弱的不完整膜阳性肿瘤细胞': 0,
    '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
    '阴性肿瘤细胞': 2,
    '强度的完整细胞膜阳性肿瘤细胞': 3,
    '中-强度的不完整细胞膜阳性肿瘤细胞': 4,
    "其他": 5,
}


def pointinpolygon(center_coords=[],labels=[], x_coords=[],y_coords=[]):
    validate_center_coords = []
    validate_labels = []
    polygon_list = []
    print(len(x_coords))
    print(len(y_coords))
    for idx, coord in enumerate(x_coords):
        point = (coord,y_coords[idx])
        polygon_list.append(point)
    polygon = Polygon(polygon_list)
    for iidx, coord in enumerate(center_coords):
        point = Point(coord[0],coord[1])
        if polygon.contains(point):
            validate_center_coords.append(coord)
            validate_labels.append(labels[iidx])
    return validate_center_coords,validate_labels

def roi_filter(center_coords, labels, x_coords, y_coords):

    if len(y_coords) > 0 and len(x_coords) > 0:
        center_coords,labels =  pointinpolygon(center_coords,labels,x_coords,y_coords)
    return center_coords, labels

#重要！不要改！奇怪的评分逻辑！
def cal_score_reverse(a_wsi_result):
    cell_count1 = float(a_wsi_result[label_dict['微弱的不完整膜阳性肿瘤细胞']]) #0 
    cell_count2 = float(a_wsi_result[label_dict['弱-中等的完整细胞膜阳性肿瘤细胞']])# 1
    cell_count3 = float(a_wsi_result[label_dict['中-强度的不完整细胞膜阳性肿瘤细胞']]) # 3
    cell_count4 = float(a_wsi_result[label_dict['强度的完整细胞膜阳性肿瘤细胞']]) # 4
    if(cell_count1)>=(cell_count4):
        cell_count2 = cell_count3+cell_count2
    if(cell_count1)<(cell_count4):
        cell_count1 = cell_count3+cell_count1
    cell_count3 = 0
    cell_count5 = cell_count2 + cell_count4  # 完整细胞膜阳性肿瘤细胞
    cell_count6 = cell_count1 + cell_count2 + cell_count3 + cell_count4  # 阳性肿瘤细胞
    cell_count7 = float(a_wsi_result[label_dict['阴性肿瘤细胞']]) # 2
    cell_count8 = cell_count6 + cell_count7 + 1e-16 # 所有肿瘤细胞

    if (cell_count4/cell_count8) > 0.1:
        return 3,False
    elif (cell_count4/cell_count8)>= 0.005 or (cell_count5/cell_count8) > 0.1:
        return 2,False
    elif ((cell_count2/cell_count8)>= 0.02 and (cell_count2/cell_count8)<=0.1) or (((cell_count1+cell_count3)/cell_count8)> 0.1 and (cell_count5/cell_count8)<=0.1):
        return 1,False
    elif ((cell_count1+cell_count3)/cell_count8 <= 0.1) or (cell_count6/cell_count8)< 0.01:
        return 0,False
    return -1,False

def analysis_wsi(num_classes,cell_labels):
    flg = False
    cell_count_dict = {}
    cell_count_dict['all'] = len(cell_labels)
    for i in range(num_classes):
        cell_count_dict[i]=0
    for idx, data in enumerate(cell_labels):
        cell_count_dict[data] += 1
    score,flg = cal_score_reverse(cell_count_dict)
    return score,cell_count_dict,flg

def run_her2_v1(slice_path,roi_list):
    logger.info('config loading')
    config = configparser.ConfigParser()
    config.read(config_path)
    logger.info(current_root)
    logger.info(config_path)
    '''
    return variables
    '''
    pts_with_roi_id = {}
    cls_with_roi_id = {}
    error_code = 0

    '''
     config 

    '''
    region_activate = config.get('region','region_activate').strip().lower() == 'true'
    region_python = os.path.join(current_root,config.get('region','region_python'))
    ###
    logger.info('开始her2_v1')
    logger.info('config loading')
    
    logger.info(region_activate)

    if region_activate:
        '''
        remove the mask if exsits

        '''
        name = os.path.basename(slice_path).split('.')[0]
        mask_save_path = current_root+'/tmp_data/' + name + '_mask.png'
        vis_save_path = current_root+'/tmp_data/' + name + '_vis.png'
        if os.path.exists(mask_save_path):
            os.remove(mask_save_path)
        if os.path.exists(vis_save_path):
            os.remove(vis_save_path)
        ###
        logger.info('region start')
        main_pid = os.getpid()
        command = [sys.executable,region_python,'--allow-run-as-root',str(main_pid),slice_path]
        tim1 = time.time()
        region_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)

        stdout, stderr = region_process.communicate()
        if region_process.returncode == 0 and os.path.exists(mask_save_path):
            logger.info("区域模型执行完毕")
            tim2 = time.time()
            sentence = 'region process time: {:.2f} mins'.format((tim2-tim1)/60)
            logger.info(sentence)

        else:
            print("区域模型执行出错")
            logger.info(stderr)
            error_code = 1
            return pts_with_roi_id,pts_with_roi_id,error_code


    ''' 
    load cell config
    '''
    cell_python = os.path.join(current_root,config.get('cell','cell_python'))
    ###


    for idx,roi in enumerate(roi_list):
        roiid, x_coords, y_coords = roi['id'], roi['x'], roi['y']

        '''
        remove the reuslt if exsits

        '''   
        result_root = os.path.dirname(slice_path)
        result_pth = os.path.join(result_root,'her2_ai_result.npz')
        if os.path.exists(result_pth):
            os.remove(result_pth)
        ###

        main_pid = os.getpid()
        roi_x = json.dumps(roi['x'])
        roi_y = json.dumps(roi['y'])

        command = [
            sys.executable, cell_python,
            '--ppid',str(main_pid),
            '--slice_path',slice_path,
            '--roi_x',roi_x,
            '--roi_y',roi_y
        ]
        tim1 = time.time()
        cell_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
        stdout, stderr = cell_process.communicate()
        if cell_process.returncode == 0 and os.path.exists(result_pth):
            logger.info("细胞模型执行完毕")
            tim2 = time.time()
            sentence = 'cell process time: {:.2f} mins'.format((tim2-tim1)/60)
            logger.info(sentence)
        else:
            logger.info(stderr)
            print("细胞模型执行出错")
            error_code = 2
            return pts_with_roi_id,pts_with_roi_id,error_code
        
        # np.savez(save_path,points=points,labels=labels,patch_xy=patch_xy,probs=probs)
    
        data = np.load(result_pth)
        pts = data['points']
        labels = data['labels']

        pts_with_roi_id[roiid] = pts
        cls_with_roi_id[roiid] = labels
        data.close()

    '''
    remove masks

    '''
    name = os.path.basename(slice_path).split('.')[0]
    mask_save_path = os.path.join(current_root,'tmp_data',name + '_mask.png')
    vis_save_path = os.path.join(current_root,'tmp_data',name + '_vis.png')
    thresh_save_path = os.path.join(current_root,'tmp_data', name + '_test.png')
    if os.path.exists(mask_save_path):
        os.remove(mask_save_path)
    if os.path.exists(vis_save_path):
        os.remove(vis_save_path)
    if os.path.exists(thresh_save_path):
        os.remove(thresh_save_path)

    return pts_with_roi_id,cls_with_roi_id,error_code
 

def run_alg(slice_path,roi_list):
    pts_with_roi_id,cls_with_roi_id,error_code = run_her2_v1(slice_path,roi_list)
    if error_code == 0:
        #print(len(pts_with_roi_id[100]))
        cls_wsi = []
        for k in cls_with_roi_id.keys():
                cls_wsi.extend(cls_with_roi_id[k])
        #score,cell_count_dict,flg
        r1, r2,r3 = analysis_wsi(6, cls_wsi)
    else:
        r1 = -1
        r2 = {}
    return pts_with_roi_id,cls_with_roi_id,r2,r1,error_code



   