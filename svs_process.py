# svs格式转换为png格式
import numpy as np
import openslide
from shutil import copyfile
from PIL import Image

import os


def read_img(path):
    files_list=[]
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.split('.')[-1] == 'svs':
                name=os.path.join(root,name)
                files_list.append(name)
    return(files_list)


def svs2png(file_list):
    for i in file_list:
        try:
            svs=openslide.open_slide(i)
            img = np.array(svs.read_region((0, 0), 2, svs.level_dimensions[2]))
            img = Image.fromarray(img)
            new_name=os.path.split(i)[1]
            new_name=os.path.splitext(new_name)[0]+'.png'
            img.save('png/'+new_name)
        except:
            pass
        continue


svs2png(read_img('/lustre/home/acct-lctest/stu411/LUNG/LUAD/Image/'))

train_list = open('train.txt').readlines()
test_list = open('test.txt').readlines()
for root, dirs, files in os.walk('png/'):
    for name in files:
        img_id = name[0:12]+'\n'
        name_path = os.path.join(root, name)
        if img_id in train_list:
            newpath='train/'+name
            copyfile(name_path,newpath)
        if img_id in test_list:
            newpath = 'test/' + name
            copyfile(name_path,newpath)

test_len=len([name for name in os.listdir('test/')])
train_len=len([name for name in os.listdir('train/')])
print('test','train:',test_len,train_len)
