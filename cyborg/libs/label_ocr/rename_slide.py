#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 20:12
# @Author  : Can Cui
# @File    : rename_slide.py
# @Software: PyCharm
# @Comment:


import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_tct.LabelOCR.tools.infer import utility
from test_tct.LabelOCR.tools.infer.predict_system import TextSystem
from imageio import imread
from Slide.dispatch import openSlide
from ParseCell.parse_utils.utils import walk_dir, supported_wsi_format
import re

args = utility.parse_args()
text_sys = TextSystem(args)

def predict(img_path):
    image = imread(img_path)[:,:,:3]
    dt_boxes, rec_res, crop_img_list = text_sys(image)
    return dt_boxes, rec_res, crop_img_list



def label_ocr(slide_path):
    '''
    :param label_image_path:
    :param save_path:  save cropped text area image
    :return:  list of text detected from label
    '''
    slide = openSlide(slide_path)

    label_image_path = os.path.join('../temp1', os.path.splitext(os.path.basename(slide_path))[0]+'.png')
    # print(os.path.abspath(label_image_path))
    # import pdb; pdb.set_trace()
    os.makedirs(os.path.dirname(label_image_path), exist_ok=True)

    slide.saveLabel(label_image_path)

    try:
        def hasNumbers(text, num_digits=3):
            num = 0
            for a in text:
                if a.isdigit():
                    num+=1
            return num>=num_digits

        det_box_list, text_list, crop_img_list = predict(label_image_path)
        valid_text = ''
        valid_img_list = []
        height, width = 0, 0

        for text, crop_img in zip(text_list, crop_img_list):
            if hasNumbers(text[0]):#1. case id  contains digits
                h, w = crop_img.shape[:2]
                if w>h: #2. case id is horizontal
                    valid_text += text[0]
        valid_text = re.sub("([^\x00-\x7F])+", "", valid_text)

        print(valid_text)

        if len(valid_text)>6:
            del slide
            os.rename(slide_path, os.path.join(os.path.dirname(slide_path), valid_text+'.sdpc'))
            os.rename(label_image_path, os.path.join(os.path.dirname(label_image_path), valid_text+'.png'))

    except:
        pass


def run(slide_dir):
    img_list= walk_dir(slide_dir, supported_wsi_format)
    for slide_path in img_list:
        label_ocr(slide_path)

if __name__ == '__main__':
    run("T:\适配医院数据\千麦")
    # run("T:\适配医院数据\千麦")