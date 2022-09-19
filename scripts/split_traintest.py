import os
import numpy as np
import shutil
from getClassesList import get_classes_list


rootdir = '/home/nadir/liveness/mobilenet-v2-custom-dataset/imgs' #path of the original folder
datadir = '/home/nadir/liveness/mobilenet-v2-custom-dataset/data' #path of the original folder

# classes = ['live', 'spoof']
classes = get_classes_list(rootdir)
l = len(classes)
print(classes)
# classes = ['1397', '54260', '32430', '33712', '11086', '18264', '58882', '75431', '9610', '68084', '3860', '77742', '63018', '66072', '48519', '39108', '29050', '69796', '7691', '29460', '62237', '75001', '85728', '79267', '50518', '7238', '8078', '65054', '83323', '70166', '81417', '55038', '58006', '58916', '20761', '45870']

for cl in classes:
    os.makedirs(datadir +'/train/' + cl)

    os.makedirs(datadir +'/test/' + cl)

    source = rootdir + '/' + cl

    allFileNames = os.listdir(source)

    np.random.shuffle(allFileNames)

    test_ratio = 0.1

    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [source+'/' + name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/'  + name for name in test_FileNames.tolist()]

    for name in train_FileNames:
      shutil.copy(name, datadir +'/train/' + cl)

    for name in test_FileNames:
      shutil.copy(name, datadir +'/test/' + cl)
