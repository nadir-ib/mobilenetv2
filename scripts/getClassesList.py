import os

# folder = '/home/nadir/liveness/mobilenet-v2-custom-dataset/imgs'
def get_classes_list(folder):
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return sub_folders
