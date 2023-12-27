import os
import configparser
import numpy as np
import torch
import json

import torch.multiprocessing as mp
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.ai.libs.algorithms.Her2_v1.cell_process import get_opt, config_path, main_worker


if __name__ == '__main__':
    # command = ['python', cell_python,str(main_pid),slice_path,roi_x,roi_y]
    cell_dict = {}
    opt = get_opt()
    cell_dict['pid'] = opt.ppid
    cell_dict['slice_path'] = opt.slice_path
    cell_dict['roi_x'] = json.loads(opt.roi_x)
    cell_dict['roi_y'] = json.loads(opt.roi_y)

    print('main_pid: ', cell_dict['pid'])
    config = configparser.ConfigParser()
    config.read(config_path)
    '''
    cell config 

    '''
    cell_dict['cell_crop_len'] = int(config.get('cell', 'crop_len'))
    cell_model = config.get('cell', 'cell_model')
    cell_dict['cell_model'] = os.path.join('AI', 'Her2_v1', 'Model', cell_model)
    cell_dict['cell_deduplicate_activate'] = config.get('cell', 'cell_deduplicate_activate').strip().lower() == 'true'
    cell_dict['cell_deduplicate_range'] = int(config.get('cell', 'cell_deduplicate_range'))

    '''
    common

    '''
    mpp_std = float(config.get('common', 'mpp_std'))
    cur_mpp = open_slide(cell_dict['slice_path']).mpp
    if cur_mpp is None:
        cur_mpp = mpp_std
    cur_mpp = 0.25
    # mpp_ratio = 0.25/round(cur_mpp,3)
    mpp_ratio = mpp_std / round(cur_mpp, 3)
    cell_dict['mpp_ratio'] = mpp_ratio
    cell_dict['cur_mpp'] = cur_mpp
    cell_dict['foreground_activate'] = config.get('common', 'foreground_activate').strip().lower() == 'true'
    cell_dict['thresh_scale'] = int(config.get('common', 'thresh_scale'))
    cell_dict['region_activate'] = config.get('region', 'region_activate').strip().lower() == 'true'

    port_id = 10000 + np.random.randint(0, 1000)
    cell_dict['dist_url'] = 'tcp://127.0.0.1:' + str(port_id)
    num_gpus = torch.cuda.device_count()
    cell_dict['num_gpus'] = num_gpus
    cell_dict['backend'] = 'gloo'

    if num_gpus == 1:
        preds = main_worker(gpu=0, world_size=num_gpus, cell_dict=cell_dict)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, cell_dict))
