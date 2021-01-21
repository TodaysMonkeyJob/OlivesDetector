import os
from random import randint
import shutil
import numpy as np

PATH = os.getcwd()

# Define data path
data_path = PATH + '/datasets'
data_dir_list = os.listdir(data_path)
print(data_dir_list)

root_dir = 'datasets/'  # data root path
classes_dir = data_dir_list  # total labels

val_ratio = 0.15
test_ratio = 0.05

for cls in classes_dir:
    os.makedirs(root_dir + 'train/' + cls)
    os.makedirs(root_dir + 'val/' + cls)
    os.makedirs(root_dir + 'test/' + cls)

    # Creating partitions of the data after shuffeling
    src = root_dir + cls  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                               int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]
    print(cls)
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir +'val/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir +'test/' + cls)
