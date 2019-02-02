import re
import os
from sklearn.externals import joblib

def get_id(name):
    if 'LFPW_image' in name:
        t = name.split('_')[0:4]
        t = t[0] + t[1] + t[2] + t[3]
    elif 'IBUG_image' in name:
        t = name.split('_')[0:3]
        t = t[0] + t[1] + t[2]
    else:
        t = name.split('_')[0:2]
        t = t[0] + t[1]
    return t

def ids(name):
    name = name.rstrip('.jpg')
    b = re.split("(_AFW.+|_HELEN.+|_IBUG.+|_LFPW.+)", name)
    b[1] = b[1][1:]
    return get_id(b[0]), get_id(b[1])


split = joblib.load('cache/split.pkl')

ext_dir = '/data/icme/expand/images'

for img in os.listdir(ext_dir):
    id1, id2 = ids(img)
    if split[id1] == 'train' and split[id2] == 'train':
        src = os.path.join(ext_dir, img)
        dst = os.path.join('/data/icme/train/0/', img)
        cmd = 'ln -s %s %s' % (src, dst)
        os.system(cmd)
