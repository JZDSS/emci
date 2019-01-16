# 测试代码
from face_dataset.dataset import face_dataset, root

dataset = face_dataset(root, "train")

# print(dataset[1][0].shape, dataset[1][1])
dataset.show(1)

# for i in range(len(dataset)):
#     print(dataset[i])