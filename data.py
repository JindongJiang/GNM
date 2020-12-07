import re
import os.path
import torch
from torch.utils.data import Dataset
import PIL
import PIL.Image as Image
from torchvision.transforms import ToTensor
import cv2

class MultiMNIST(Dataset):
    def __init__(self, args, mode='train', phase_image=True):
        super(MultiMNIST, self).__init__()
        self.args = args
        root = args.data_dir

        self.phase_image = phase_image

        if phase_image:
            root = [os.path.join(root, s) for s in os.listdir(root) if f'{mode}-image' in s]
            if len(root) > 1:
                root.sort(key=lambda s: int(re.split('-|\.', s)[-1]))
            self.all_image_name_list = []
            for data_dir in root:
                image_name_list = [os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.endswith('.png')]
                image_name_list.sort(
                    key=lambda s: int(re.split('\.|/', s)[-2])
                )
                self.all_image_name_list.extend(image_name_list)

            if mode == 'train':
                self.all_image_name_list = self.all_image_name_list[:60000]
            elif mode in ['test', 'val']:
                self.all_image_name_list = self.all_image_name_list[:6000]
            else:
                raise NotImplemented
        else:
            if mode == 'train':
                img_fn = [os.path.join(root, s) for s in os.listdir(root) if
                          s.startswith('train-image') and s.endswith('.pt')]
                if len(img_fn) > 1:
                    img_fn.sort(key=lambda s: int(re.split('-|/|\.', s)[-2]))
            elif mode == 'test':
                img_fn = [os.path.join(root, s) for s in os.listdir(root) if
                          s.startswith('test-image') and s.endswith('.pt')]
                if len(img_fn) > 1:
                    img_fn.sort(key=lambda s: int(re.split('-|/|\.', s)[-2]))
            elif mode == 'val':
                img_fn = [os.path.join(root, s) for s in os.listdir(root) if
                          s.startswith('val-image') and s.endswith('.pt')]
                if len(img_fn) > 1:
                    img_fn.sort(key=lambda s: int(re.split('-|/|\.', s)[-2]))
            else:
                raise NotImplemented

            self.data = torch.cat([torch.load(fn) for fn in img_fn], dim=0)

    def __getitem__(self, index):

        if self.phase_image:
            data = cv2.imread(self.all_image_name_list[index])
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = ToTensor()(data)[:3]
        else:
            data = self.data[index]

        if self.args.data.inp_channel == 1:
            data = data[:1]
            return data
        else:
            return data

    def __len__(self):
        if self.phase_image:
            return len(self.all_image_name_list)
        else:
            return self.data.size(0)


class Blender(Dataset):
    def __init__(self, args, mode='train', phase_label=False):
        super(Blender, self).__init__()
        self.args = args

        self.phase_label = phase_label

        if mode == 'train':
            image_dir_list = [os.path.join(d, 'images') for d in args.data.blender_dir_list_train]
        elif mode == 'test' or mode == 'val':
            image_dir_list = [os.path.join(d, 'images') for d in args.data.blender_dir_list_test]
        else:
            raise NotImplemented

        self.image_list = []

        for dir in image_dir_list:
            image_list_i = [os.path.join(dir, fn) for fn in os.listdir(dir) if fn.endswith('png')]
            image_list_i.sort(key=lambda s: int(re.split('_|-|/|\.', s)[-2]))
            self.image_list.extend(image_list_i)

        if mode == 'val':
            self.image_list = self.image_list[:6000]
        elif mode == 'test':
            self.image_list = self.image_list[6000:12000]

    def __getitem__(self, index):

        img = Image.open(self.image_list[index])
        img = img.resize((self.args.data.img_h, self.args.data.img_w), PIL.Image.BILINEAR)
        out = ToTensor()(img)[:3]

        if self.phase_label:
            return out, 0
        else:
            return out

    def __len__(self):
        return len(self.image_list)