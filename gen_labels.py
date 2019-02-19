from sklearn.externals import joblib

split = joblib.load('cache/split.pkl')
b = {}
l = 0
for id, phase in split.items():
    if phase == 'train':
        b[id] = l
        l += 1
joblib.dump(b, 'cache/labels.pkl')
