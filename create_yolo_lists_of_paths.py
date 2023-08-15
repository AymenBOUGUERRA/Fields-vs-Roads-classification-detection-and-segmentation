import os
def get_image_paths(directory):
    image_paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                image_path = os.path.join(subdir, file)
                image_paths.append(image_path)
    return image_paths

train_dir = '/home/aymen/PycharmProjects/owlvit_segment_anything/segmentation_dataset/train/images/'
valid_dir = '/home/aymen/PycharmProjects/owlvit_segment_anything/segmentation_dataset/valid/images/'

train_image_paths = get_image_paths(train_dir)
valid_image_paths = get_image_paths(valid_dir)

with open('segmentation_dataset/train.txt', 'w') as f:
    for image_path in train_image_paths:
        f.write(image_path + '\n')

with open('segmentation_dataset/val.txt', 'w') as f:
    for image_path in valid_image_paths:
        f.write(image_path + '\n')
