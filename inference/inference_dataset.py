### dataloader for human parsing inference
### Bowen Wu. SYSU

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloaders import custom_transforms as tr
import os
import os.path as osp

from PIL import Image
import numpy as np

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

class InferenceDataset(Dataset):
    
    def __init__(self, opts):
        self.data_root = opts.data_root
        file = open(osp.join(self.data_root, opts.img_list))
        self.output_dir = opts.output_dir
        self.phase = opts.phase
        self.data_list = file.readlines()
        self.delete_exist()
        self.len = len(self.data_list)
        
        self.scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    
    def delete_exist(self):
        new_list = []
        for img_path in self.data_list:
            img_path = img_path.strip('\n')
            img_split = img_path.split('/')
            img_id = img_split[-1][:-4]
            # print('########debug#########')
            # print(img_path)
            # print(img_split)
            # print(img_id)
            # print('####################')
            video_id = img_split[-2]
            vis_output_dir = os.path.join(self.output_dir, self.phase + '_parsing_vis', video_id)
            label_output_dir = os.path.join(self.output_dir, self.phase + '_parsing', video_id)
            output_name = '{}_vis.png'.format(img_id)
            label_output_name = '{}_label.png'.format(img_id)
            if osp.exists(osp.join(vis_output_dir, output_name)) and osp.exists(osp.join(label_output_dir, label_output_name)):
                print('skip', img_path)
            else:
                new_list.append(img_path)
        self.data_list = new_list
        print('delete exist finish')
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = read_img(osp.join(self.data_root, img_path))
        testloader_list = []
        testloader_flip_list = []
        for pv in self.scale_list:
            composed_transforms_ts = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])
            testloader_list.append(img_transform(img, composed_transforms_ts))
            # print(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
        
        # calculate output path
        img_split = img_path.split('/')
        img_id = img_split[-1][:-4]
        video_id = img_split[-2]
        vis_output_dir = os.path.join(self.output_dir, self.phase + '_parsing_vis', video_id)
        label_output_dir = os.path.join(self.output_dir, self.phase + '_parsing', video_id)
        label_output_name = '{}_label.png'.format(img_id)
        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)
        output_name = '{}_vis.png'.format(img_id)
        output_path = osp.join(vis_output_dir, output_name)
        label_output_path = osp.join(label_output_dir, label_output_name)
        return {
            'testloader_list':testloader_list,
            'testloader_flip_list':testloader_flip_list,
            'img_path':img_path,
            'output_path':output_path,
            'label_output_path':label_output_path,
        }


# by Bowen Wu
class FashionGenDataset(Dataset):

    def __init__(self, data_root, result_dir, phase='train'):
        self.data_root = data_root
        self.output_dir = result_dir
        self.scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
        if phase == 'train':
            self.h5_filename = os.path.join(self.data_root, 'fashiongen_256_256_train.h5')
        elif phase == 'validation':
            self.h5_filename = os.path.join(self.data_root, 'fashiongen_256_256_validation.h5')
        import h5py
        self.h5_file = h5py.File(self.h5_filename, mode='r')
        self.phase = phase
    def __len__(self):
        return len(self.h5_file['input_image'])
    
    def __getitem__(self, index):
        # print('h5_file', self.h5_file.keys())
        img = Image.fromarray(self.h5_file['input_image'][index].astype('uint8'), 'RGB')
        key = str(index).zfill(6)
        testloader_list = []
        testloader_flip_list = []
        for pv in self.scale_list:
            composed_transforms_ts = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])
            testloader_list.append(img_transform(img, composed_transforms_ts))
            # print(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
        # calcualte output path
        vis_output_dir = os.path.join(self.output_dir, self.phase + '_parsing_vis')
        vis_output_name = '{}_vis.png'.format(key)
        label_output_dir = os.path.join(self.output_dir, self.phase + '_parsing')
        label_output_name = '{}_label.png'.format(key)
        origin_output_dir = os.path.join(self.output_dir, self.phase + '_origin')
        origin_output_name = '{}.png'.format(key)
        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)
        if not os.path.exists(origin_output_dir):
            os.makedirs(origin_output_dir)
        img.save(osp.join(origin_output_dir, origin_output_name))
        return {
            'testloader_list':testloader_list,
            'testloader_flip_list':testloader_flip_list,
            # 'image':img,
            'vis_output_path':osp.join(vis_output_dir, vis_output_name),
            'label_output_path':osp.join(label_output_dir, label_output_name),
            'origin_output_path':osp.join(origin_output_dir, origin_output_name)
        }

class TryonDataset(Dataset):

    def __init__(self, data_root, result_dir):
        self.data_root = data_root
        self.output_dir = result_dir
        self.scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
        self.image_list = self.get_image_list(data_root)

    def get_image_list(self, data_root):
        image_list = []
        for dress in os.listdir(data_root):
            for j in os.listdir(osp.join(data_root, dress)):
                for item in os.listdir(osp.join(data_root, dress, j)):
                    image_list.append(osp.join(dress, j, item))
        return image_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = read_img(osp.join(self.data_root, img_path))
        testloader_list = []
        testloader_flip_list = []
        for pv in self.scale_list:
            composed_transforms_ts = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])
            testloader_list.append(img_transform(img, composed_transforms_ts))
            # print(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
        # calcualte output path
        vis_output_dir = os.path.join(self.output_dir, 'parsing_vis')
        vis_output_name = img_path.replace('.png', '_vis.png').replace('.jpg', '_vis.png')
        vis_output_full_path = osp.join(vis_output_dir, vis_output_name)
        label_output_dir = os.path.join(self.output_dir, 'parsing')
        label_output_name = img_path.replace('.png', '_label.png').replace('.jpg', '_label.png')
        label_output_full_path = osp.join(label_output_dir, label_output_name)
        if not os.path.exists(osp.split(vis_output_full_path)[0]):
            # print(osp.split(vis_output_full_path)[0])
            os.makedirs(osp.split(vis_output_full_path)[0])
        if not os.path.exists(osp.split(label_output_full_path)[0]):
            os.makedirs(osp.split(label_output_full_path)[0])
        return {
            'testloader_list':testloader_list,
            'testloader_flip_list':testloader_flip_list,
            # 'image':img,
            'vis_output_path':osp.join(vis_output_dir, vis_output_name),
            'label_output_path':osp.join(label_output_dir, label_output_name),
        }


def get_infernce_dataloader(opts):
    inference_dataset = InferenceDataset(opts)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=16)
    return inference_dataloader

def get_fashiongen_dataloader(data_root, result_dir, phase='train'):
    inference_dataset = FashionGenDataset(data_root, result_dir, phase)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=1)
    return inference_dataloader

def get_tryon_dataloader(data_root, result_dir):
    inference_dataset = TryonDataset(data_root, result_dir)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=1)
    return inference_dataloader

# if __name__ == "__main__":
#     opts = {
#         'img_list':'train_val_part7.txt',
#         'data_root':'/opt/ohpc/xulin/pose_dataset_256p',
#         'output_dir':'/data/wubowen/parsing_result_test',
#     }

#     inference_dataloader = get_infernce_dataloader(opts)
#     for i_batch, data in enumerate(inference_dataloader):
#         testloader_list = data['testloader_list'][0]
#         testloader_flip_list = data['testloader_flip_list'][0]
#         output_path = data['output_path']
#         print(i_batch, data)
