import sys,os
import torch.multiprocessing as mp
import configparser
import numpy as np
import torch

from cyborg.modules.ai.libs.algorithms.Her2_v1.region_process import config_path, main_worker

if __name__ == '__main__':
    region_ = {}
    region_['pid'] = sys.argv[-2]
    region_['slice_path'] = sys.argv[-1]
    print('main_pid: ',sys.argv[-2])
    config = configparser.ConfigParser()
    config.read(config_path)
    '''
    region config 

    '''
    region_['region_threshold'] = float(config.get('region','region_threshold'))
    region_['region_iou_threshold'] = float(config.get('region','iou_threshold'))
    region_['region_crop_len'] = int(config.get('region','crop_len'))
    region_['region_stride_ratio'] = float(config.get('region','stride_ratio'))
    region_model = config.get('region','region_model')
    #region_['region_model'] = os.path.join(current_root,'Model',region_model)
    region_['region_model'] = os.path.join('AI', 'Her2_v1', 'Model', region_model)
    region_['region_vis'] = config.get('region','region_vis').strip().lower() == 'true'
    '''
    common

    '''
    mpp_std = float(config.get('common','mpp_std'))
    region_['thresh_scale'] = int(config.get('common','thresh_scale'))
    region_['mask_scale'] =list(map(int, config.get('common','mask_scale').split(",")))
    ###


    port_id = 10000+np.random.randint(0,1000)
    region_['dist_url'] = 'tcp://127.0.0.1:'+str(port_id)
    num_gpus = torch.cuda.device_count()
    region_['num_gpus'] = num_gpus
    region_['backend'] = 'gloo'
    
    if num_gpus == 1:
        preds = main_worker(gpu=0,world_size=num_gpus,region_dict=region_)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus,region_))
