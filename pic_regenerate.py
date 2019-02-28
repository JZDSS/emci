from sklearn.externals import joblib
from pdb import get_id
import os
from savepic_lmk import get_name2
from get_index import save_pic_and_lmk

split = joblib.load('cache/split.pkl')
for k in split.keys():
    split[k] = []

names = os.listdir('/home/zhang/correctdata/data/picture')
for name in names:
    id = get_id(name)
    split[id].append(name)

for a in split.items():
    length = len(a[1])            #a[1]为anotest输入
    c = get_name2(a[1])
    save_pic_and_lmk(c)
    b=2


