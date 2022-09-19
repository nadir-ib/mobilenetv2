from PIL import Image
from PIL import UnidentifiedImageError
import os
root = "/home/demo/nadir/mobilenetv2/data/test"

a = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]

#print(a)


for img_p in a:
    try:
        img = Image.open(img_p)
    except UnidentifiedImageError:
            print(img_p)