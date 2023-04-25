import time
import numpy as np
import os
import cv2
import os
import pdb
from glob import glob
from PIL import Image
import time
import torch
from imageio import imsave, imread
from torch.autograd import Variable
from test_scripts.utils import walk_dir, save_json, my_save_img, save_img, my_save_json
import torch.nn.functional as F

def split_image(image, patch_size, overlap):
    h,w = image.shape[0:2]
    stride = patch_size - overlap
    patch_list = []
    coord_list = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            crop_img = image[y:y+patch_size, x:x+patch_size, :]
            crop_h, crop_w = crop_img.shape[0:2]
            pad_h, pad_w = patch_size-crop_h, patch_size-crop_w
            if pad_h>0 or pad_w>0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0,0)), 'constant')
            coord_list.append(np.array([x, y]))
            patch_list.append(crop_img)
    patch_image = np.array(patch_list)

    return patch_image, coord_list

def del_duplicate(outputs_points, outputs_scores, interval):
    n = len(outputs_points)
    fused = np.full(n, False)
    filtered_points = []
    filtered_labels = []
    for i, point in enumerate(outputs_points):
        if fused[i] == False:
            distance = np.linalg.norm(point - outputs_points, 2, axis=1)
            distance_bool = np.where((distance < interval))[0]
            max_score = 0
            for index in distance_bool:
                fused[index] = True
                if np.max(outputs_scores[index]) > max_score:
                    max_score = np.max(outputs_scores[index])
                    max_index = index
            filtered_points.append(outputs_points[max_index])
            filtered_labels.append(np.argmax(outputs_scores[max_index]))
    return np.array(filtered_points), np.array(filtered_labels)

def get_pred_results(outputs):
    outputs_scores = F.softmax(outputs['pred_logits'][0], dim=-1).cpu().numpy()
    outputs_points = outputs['pred_points'][0].cpu().numpy()
    argmax_label = torch.argmax(outputs['pred_logits'][0], dim=-1).cpu().numpy()
    valid_pred_bool = (argmax_label > 0)
    pred_sum = valid_pred_bool.sum()

    if pred_sum > 0:
        outputs_points = outputs_points[valid_pred_bool]
        outputs_scores = outputs_scores[valid_pred_bool]
        pred_points, pred_labels = del_duplicate(outputs_points, outputs_scores, 16)
        pred_points = np.array(pred_points)
        pred_labels = np.array(pred_labels)
    else:
        pred_points = None
        pred_labels = None

    return pred_points, pred_labels

def generate_result_mask(image, net, patch_size=512, overlap=128):

    patch_imgs, coord_list = split_image(image , patch_size, overlap)
    patch_imgs = patch_imgs.transpose((0, 3, 1, 2))
    patch_imgs = patch_imgs * (2. / 255) - 1.
    final_pred_labels = []
    final_pred_points = []
    for this_batch, coord in zip(patch_imgs, coord_list):
        with torch.no_grad():
            data_variable = Variable(torch.from_numpy(this_batch).float())
            if net.parameters().__next__().is_cuda:
                data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                data_variable = data_variable.unsqueeze(dim=0)
                outputs = net(data_variable)
                pred_points, pred_labels = get_pred_results(outputs)
                if pred_points is not None:
                    pred_points = pred_points + coord
                    pred_points = list(pred_points)
                    pred_labels = list(pred_labels)
                    final_pred_points.extend(pred_points)
                    final_pred_labels.extend(pred_labels)

    final_pred_points = np.array(final_pred_points)
    final_pred_labels = np.array(final_pred_labels)
    # final_pred_points, final_pred_labels = del_duplicate(final_pred_points, final_pred_labels, 16)

    return final_pred_points, final_pred_labels

def test_p2p_ki67_0418(net, image_dir, result_save_dir, eval_save_folder):
    image_list = glob(image_dir + '/*/slices/*/*.png')
    print('image_list', image_list)
    for img_path in image_list:
        # img_path = "/data2/Caijt/KI67_under_going/ki67_deployment_V2/Data/KI67_data_V2_done/2 2021_10_02_19_31_00_22071_done/32.png"
        image = imread(img_path)
        im = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = np.array(im)
        start_time = time.time()
        center_coords_pred, labels_pred = generate_result_mask(image, net, overlap = 0)
        labels_pred = labels_pred - 1
        end_time = time.time()
        calculation_during = end_time - start_time
        print('calculation time', calculation_during, 's')
        print('-----------------------------------------')
        base_name = os.path.basename(os.path.dirname(img_path)) + '_' + os.path.basename(img_path).split('.')[0]
        # save_pred_img(img_path, id_name, center_coords_pred, labels_pred, color_dict, result_save_dir)
        annotation_path = os.path.dirname(img_path) + '/index.json'
        my_save_img(img_path, base_name, annotation_path, center_coords_pred, labels_pred, result_save_dir)
        my_save_json(img_path, base_name, annotation_path, center_coords_pred, labels_pred, eval_save_folder)
        # pdb.set_trace()

def test_p2p_ki67(net, image_dir, result_save_dir, eval_save_folder):
    image_list = [pth for pth in walk_dir(image_dir, ['.png', '.jpg', '.bmp']) if 'val' in pth]
    print('image_list', image_list)
    for img_path in image_list:
        # img_path = "/data2/Caijt/KI67_under_going/ki67_deployment_V2/Data/KI67_data_V2_done/2 2021_10_02_19_31_00_22071_done/32.png"
        image = imread(img_path)
        im = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = np.array(im)
        start_time = time.time()
        center_coords_pred, labels_pred = generate_result_mask(image, net, overlap = 0)
        labels_pred = labels_pred - 1
        end_time = time.time()
        calculation_during = end_time - start_time
        print('calculation time', calculation_during, 's')
        print('-----------------------------------------')
        dir_name = os.path.basename(os.path.dirname(img_path))
        id_name = str(dir_name[0])
        # save_pred_img(img_path, id_name, center_coords_pred, labels_pred, color_dict, result_save_dir)
        annotation_path = os.path.splitext(img_path)[0] + '.json'
        save_img(img_path, annotation_path, center_coords_pred, labels_pred, result_save_dir)
        save_json(img_path, id_name, annotation_path, center_coords_pred, labels_pred, eval_save_folder)
        # pdb.set_trace()
    return

# if __name__ == '__main__':
#     color_dict = {"1": (0, 255, 0), "2": (255, 0, 0), "3": (255, 255, 0),
#                   "4": (139, 105, 20), "5": (0, 0, 255), "6": (128, 0, 255)}
#     outputs = get_fake_outputs()
#     threshold = 0.3
#     result_save_dir = '/data2/Caijt/KI67_under_going/Densitymap_P2P_compare/save_images/'
#     img_path = "/data2/Caijt/KI67_under_going/ki67_deployment_V2/Data/KI67_data_V2_done/4 2021_11_30_09_59_23_597355_done/1.png"
#     image = imread(img_path)
#     pred_labels, pred_points = generate_result_mask(image, outputs, threshold=threshold)
#     save_pred_img(img_path, pred_points, pred_labels, color_dict, result_save_dir)