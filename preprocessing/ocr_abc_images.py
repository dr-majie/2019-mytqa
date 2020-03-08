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
"""
test_file_raw = ocr_space_file(filename='/data/kf/majie/dataset/tqa_train_val_test/train/abc_question_images/aquifers_16530.png', language='eng')
test_file=json.loads(test_file_raw)


with open("/data/kf/majie/dataset/tqa_train_val_test/train/abc_question_images/record.json", "r") as f:
    test_file = json.load(f)
    
print(type(test_file))

print(json_text)
print(type(json_text))


print(test_file)
print(type(test_file))
for item in test_file.items():
    print(item)
print(type(test_file['ParsedResults'][0]))
print(type(test_file['ParsedResults'][0]['TextOverlay']))
print(type(test_file['ParsedResults'][0]['TextOverlay']['Lines']))
"""


"""
texts=test_file['ParsedResults'][0]['TextOverlay']['Lines']
text_list=[]
for text in texts:
    #print((text['Words'][0]))
    dict_text=text['Words'][0]
    dict_text['Center']=(dict_text['Left']+dict_text['Width']/2 , dict_text['Top']-dict_text['Height']/2)
    text_list.append(dict_text)
#print(text_list)

image_info={}
image_info['Name']="aquifers_16530.png"
image_info['Words']=text_list
print(image_info)

with open('/data/kf/majie/dataset/tqa_train_val_test/train/abc_question_images/atest.json', 'w') as json_file:
        json.dump(image_info,json_file)
        json_file.write('\n')
        print("ok1")
        json.dump(image_info, json_file)
        json_file.write('\n')
        print("ok2")
        json.dump(image_info, json_file)
        print("ok3")

"""

# test_url = ocr_space_url(url='http://i.imgur.com/31d5L5y.jpg')
# print(test_url)
keylist = [chr(i) for i in range(65, 91)]
keylist += [chr(i) for i in range(97, 123)]
#print(keylist)
def Not_Alpha(s):
    if s in keylist:
        return 0
    return 1
#print(Not_alpha('djawl'))






import os
import json
url_lists=['train','val','test']
for url_list in url_lists:
    url = '/data/kf/majie/dataset/tqa_train_val_test/'+url_list+'/abc_question_images'
    #print(url)
    file_list = os.listdir(url)
    write_url = '/data/kf/majie/dataset/tqa_train_val_test/' + url_list + '/abc_question_images/abc_images_info.json'
    with open(write_url, 'w') as json_file:
        json_file.write('\n')
    for file in file_list[0:3]:
        dirs = url +'/'+ file
        #print(dirs)
        ocr_file = ocr_space_file(filename=dirs,language='eng')
        json_file = json.loads(ocr_file)
        texts = json_file['ParsedResults'][0]['TextOverlay']['Lines']
        text_list = []
        for text in texts:
            # print((text['Words'][0]))
            dict_text = text['Words'][0]
            if Not_Alpha(dict_text['WordText']):
                print("************************")
                print(dict_text)
                print(dirs)
            dict_text['Center'] = (dict_text['Left'] + dict_text['Width'] / 2, dict_text['Top'] - dict_text['Height'] / 2)
            text_list.append(dict_text)

        image_info = {}
        image_info['Name'] = file
        image_info['Words'] = text_list
        #print(image_info)

        write_url='/data/kf/majie/dataset/tqa_train_val_test/'+url_list+'/abc_question_images/abc_images_info.json'
        with open(write_url,'a') as json_file:
                json.dump(image_info, json_file)
                json_file.write('\n')
    print(url_list+" write ok!")

