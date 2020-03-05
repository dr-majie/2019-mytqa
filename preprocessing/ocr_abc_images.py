#-----------------------------------------------
#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time:2020/3/2 10:13
#@Author:Ma Jie
#@FileName: preprocessing.py
#-----------------------------------------------
import requests


def ocr_space_file(filename, overlay=True, api_key='64f27b7a7588957', language='eng'):
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


def ocr_space_url(url, overlay=False, api_key='64f27b7a7588957', language='eng'):
    """ OCR.space API request with remote file.
        Python3.5 - not tested on 2.7
    :param url: Image url.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'url': url,
               'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    r = requests.post('https://api.ocr.space/parse/image',
                      data=payload
                      )
    return r.content.decode()


# Use examples:
test_file = ocr_space_file(filename='/data/kf/majie/codehub/2019-mytqa/data/tqa_train_val_test/train/abc_question_images/earth_day_night_10170.png', language='eng')
print(test_file)
# test_url = ocr_space_url(url='http://i.imgur.com/31d5L5y.jpg')
# print(test_url)