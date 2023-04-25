# import zxing
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyzbar.pyzbar import decode
from tools.infer import utility
from tools.infer.predict_system import TextSystem
from imageio import imread, imsave
import numpy as np
import re
import cv2
#args = utility.parse_args()
#text_sys = TextSystem(args)

def predict(img_path):
    image = imread(img_path)[:,:,:3]
    args = utility.parse_args()
    text_sys = TextSystem(args)
    dt_boxes, rec_res, crop_img_list = text_sys(image)
    return dt_boxes, rec_res, crop_img_list

def label_ocr(label_image_path, save_path):
    '''
    :param label_image_path:
    :param save_path:  save cropped text area image
    :return:  list of text detected from label
    '''
    # try:
    def hasNumbers(text, num_digits=3):
        num = 0
        for a in text:
            if a.isdigit():
                num+=1
        return num>=num_digits

    det_box_list, text_list, crop_img_list = predict(label_image_path)
    valid_text_list = []
    valid_img_list = []
    height, width = 0, 0
    # import pdb; pdb.set_trace()
    for text, crop_img in zip(text_list, crop_img_list):
        if hasNumbers(text[0]):#1. case id  contains digits
            h, w = crop_img.shape[:2]
            if w>h: #2. case id is horizontal
                if text[0][0].isalpha():
                    valid_text_list.insert(0, text[0])
                    valid_img_list.insert(0,crop_img)
                else:
                    valid_text_list.append(text[0])
                    valid_img_list.append(crop_img)
                height += h
                width = max(width, w)
    valid_text = ''.join(valid_text_list)
    # Remove Chinese Characters
    valid_text = re.sub("([^\x00-\x7F])+", "", valid_text)

    if len(valid_text) == 0: # ocr failed
        label_img=cv2.resize(imread(label_image_path),(1296,217))
        # label_img= imread(label_image_path)
        if save_path:
            imsave(save_path, label_img)
    else:
        new_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        start_h = 0
        for np_img in valid_img_list:
            h, w, _ = np_img.shape
            new_image[start_h:start_h + h, 0:w, :] = np_img
            start_h += h
        new_image = cv2.resize(new_image, (1296, 217))
        if save_path:
            imsave(save_path, new_image)
    # except:
    #     valid_text = ''
    return valid_text

def barcode_reader(image):
    # decodes all barcodes from an image
    decoded_objects = decode(image)
    if len(decoded_objects) == 0:
        return None
    else:
        data = decoded_objects[0].data
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return data

def qrcode_reader(image):
    detector = cv2.QRCodeDetector()
    retval, points, straight_qrcode = detector.detectAndDecode(image)
    # import pdb; pdb.set_trace()
    return retval

# def zxing_reader(image):
#     reader = zxing.BarCodeReader()
#     barcode = reader.decode(image)
#     print(barcode)
def label_recognition(slide_label_path, save_path, mode=4):
    '''mode{1:不识别， 2:barcode, 3:qrcode, 4:ocr'''

    label_image = np.array(imread(slide_label_path))[:,:,0:3]
    #1. 二维码
    barcode_data = qrcode_reader(label_image)
    if not barcode_data:
        #2. 一维码
        barcode_data = barcode_reader(label_image)
    #3. OCR
    ocr_label = label_ocr(slide_label_path, save_path)
    if mode<4:
        return barcode_data if barcode_data is not None else ''
    else:
        return ocr_label


if __name__ == '__main__':
    image = np.array(imread(r'H:\data3\大连医科大学LCT初测\data\2022_01_25_14_02_27_735263\slices\20220125140227816100\label.png'))[:,:,0:3]
    print(image.shape)
    # data = barcode_reader(image)
    print(label_ocr(r'H:\data3\大连医科大学LCT初测\data\2022_01_25_14_03_41_83595\slices\20220125140341965293\label.png', 'test.png'))
