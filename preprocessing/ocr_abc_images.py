# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/2 10:13
# @Author:Ma Jie
# @FileName: preprocessing.py
# -----------------------------------------------
import requests
import os
import json
from PIL import Image
import os


def ocr_space_file(filename, overlay=True, api_key='PKMXB8054888A', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               'OCRengine': 2
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload
                          )
    return r.content.decode()


def get_img_list(file_path):
    img_list = [img for img in os.listdir(file_path) if not os.path.isdir(img) and img.endswith('png')]
    img_list.sort()
    return img_list


def is_alpha(character):
    if ord(character[0].lower()) in range(97, 123):
        return 1
    else:
        return 0


def is_existing(character, img_info):
    for char_dict in img_info:
        if char_dict['WordText'] is character:
            print(character, char_dict['WordText'])
            print('{} is in img info list'.format(character))
            return 0
    else:
        print('{} is not in img info list'.format(character))
        return 1


def write_info(info, img, img_path, json_dict):
    img_info = []

    for line_text in info:
        char_dict = {}
        detailed_char_info = {}

        if len(line_text['LineText']) == 1 and is_alpha(line_text['LineText']):
            if is_existing(line_text['LineText'], img_info):
                detailed_char_info['Center'] = (line_text['Words'][0]['Left'] + line_text['Words'][0]['Width'] / 2,
                                                line_text['Words'][0]['Top'] + line_text['Words'][0]['Height'] / 2)
                detailed_char_info['Left'] = line_text['Words'][0]['Left']
                detailed_char_info['Top'] = line_text['Words'][0]['Top']
                detailed_char_info['Height'] = line_text['Words'][0]['Height']
                detailed_char_info['Width'] = line_text['Words'][0]['Width']

                char_dict['WordText'] = line_text['Words'][0]['WordText']
                char_dict['Coordinate'] = detailed_char_info

                img_info.append(char_dict)

        elif len(line_text['LineText']) == 1 and (not is_alpha(line_text['LineText'])):
            pass

        elif len(line_text['LineText']) == 2 and is_alpha(line_text['Words'][0]['WordText'][1]):

            if is_existing(line_text['Words'][0]['WordText'][1], img_info):
                detailed_char_info['Center'] = (line_text['Words'][0]['Left'] + line_text['Words'][0]['Width'] / 2,
                                                line_text['Words'][0]['Top'] + line_text['Words'][0]['Height'] / 2)
                detailed_char_info['Left'] = line_text['Words'][0]['Left']
                detailed_char_info['Top'] = line_text['Words'][0]['Top']
                detailed_char_info['Height'] = line_text['Words'][0]['Height']
                detailed_char_info['Width'] = line_text['Words'][0]['Width']

                char_dict['WordText'] = line_text['Words'][0]['WordText'][1]
                char_dict['Coordinate'] = detailed_char_info

                img_info.append(char_dict)
        else:
            pass

    json_dict[img] = img_info


if __name__ == '__main__':
    slice_path_list = ['train', 'val', 'test']

    for slice_path in slice_path_list:
        img_path = '/data/kf/majie/codehub/2019-mytqa/data/' + slice_path + '/abc_question_images/'
        img_list = get_img_list(img_path)
        json_dict = {}
        img_info_json_path = os.path.join(img_path, 'abc_question_images.json')

        for img in img_list:
            info = ocr_space_file(os.path.join(img_path, img))
            print(info)
            info = json.loads(info)
            print(img)
            info = info['ParsedResults'][0]['TextOverlay']['Lines']
            write_info(info, img, img_path, json_dict)

        with open(img_info_json_path, 'w') as f:
            json.dump(json_dict, f)
            f.write('\n')