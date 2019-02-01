from sklearn.externals import joblib
import os

names_dir = '/home/orion/Desktop/my_code/names/'
images_dir = '/data/expand/images/'
landmarks_dir = '/data/expand/landmarks/'
picture_draws_dir = '/data/expand/picture_draws/'

names_list = []
delete_names = []


#**********************************#
#删除images，landmark
# for i in range(1, 8):
#     names = joblib.load(names_dir + str(i) + '.pkl')
#     print(str(i), ':', len(names))
#     names_list += names
#
# print('Total:', len(names_list))
#
# print('........................')
#
# images = os.listdir(images_dir)
# print('images_num:', len(images))
#
# for j in images:
#     if j not in names_list:
#         os.remove(images_dir + j)
#         os.remove(landmarks_dir + j + '.txt')
#         delete_names.append(j)
#         # print('delete:', j)
#
# joblib.dump(delete_names, '/home/orion/Desktop/my_code/delete_names.pkl', compress=3)
# images = os.listdir(images_dir)
# print('deleted_num:', len(delete_names))
# print('reserved_num:', len(images))


#*******************************#
# #删除picture_draws
# for i in range(1, 8):
#     names = joblib.load(names_dir + str(i) + '.pkl')
#     images = os.listdir(picture_draws_dir + str(i) + '/')
#     deleted_list = []
#     print(i, 'images:', len(images), 'names:', len(names))
#     for j in images:
#         if j not in names:
#             os.remove(picture_draws_dir + str(i) + '/' + j)
#             deleted_list.append(j)
#     images = os.listdir(picture_draws_dir + str(i) + '/')
#     print(i, 'deleted_num:', len(deleted_list), 'reserved_num:', len(images))
#     print('\n')



#********************************#
# #查找错误文件名
# list = []
# for j in names_list:
#     if j not in images:
#         list.append(j)
#
# print(list)
#
# for i in range(1, 8):
#     names = joblib.load(names_dir + str(i) + '.pkl')
#     for j in list:
#         if j in names:
#             print(i)


#******************************#
##pkl文件生成脚本
# #输入文件夹路径，最后为序号
# in_dir = '/data/expand/picture_draws/1'
# #输出路径，最后会在这个路径下生成如 1.kl的文件
# out_dir = '/home/orion/Desktop'
#
# filenames = os.listdir(in_dir)
# print('Nums:', len(filenames))
# joblib.dump(filenames, out_dir + '/' + in_dir[len(in_dir) - 1] + '.pkl', compress=3)