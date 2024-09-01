import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

import torchvision.transforms as transforms

from torch.nn.utils.rnn import pad_sequence

def random_mirror(rgb, gt):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)

    return rgb, gt
def random_mirror_no_crop(rgb, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, modal_x

def random_scale(rgb, gt, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    # print(type(rgb), type(gt), type(modal_x))
    # print(type(depth))
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return rgb, gt, scale
def random_scale_no_crop(rgb, modal_x, depth, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    # print(type(rgb), type(gt), type(modal_x))
    # print(type(depth))
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, modal_x, depth, scale

def collate_fn(batch):
  return {
      'data': torch.stack([x['data'] for x in batch]),
      'label': torch.stack([x['label'] for x in batch])
}
def collate_fn_val(batch):
  return {
      'data': torch.stack([x['data'] for x in batch]),
      'label': torch.stack([x['label'] for x in batch])
}
# def collate_fn(batch):
#     images, labels = zip(*batch)
#     # images = [torch.tensor(img, dtype=torch.float32) for img in images]
#     images = pad_sequence(images, batch_first=True, padding_value=0)
#     labels = torch.tensor(labels, dtype=torch.long)
#     return images, labels


# class TrainPre(object):
#     def __init__(self, norm_mean, norm_std):
#         self.norm_mean = norm_mean
#         self.norm_std = norm_std
#         self.to_tensor = transforms.ToTensor()
#     def __call__(self, rgb, modal_x, depth):
#         return rgb, modal_x, depth
class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.to_tensor = transforms.ToTensor()
    def __call__(self, rgb, gt):
        rgb, gt = random_mirror(rgb, gt)
        # print("different", rgb.shape, gt.shape)
        if config.train_scale_array is not None:
            rgb, gt, scale = random_scale(rgb, gt, config.train_scale_array)
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_rgb = p_rgb.transpose(2, 0, 1)
        gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
        # print(gt.shape)
        # gt = p_rgb.transpose(2, 0, 1)
        # print("same", p_rgb.shape, gt.shape)

        return p_rgb, gt
    # def __init__(self, norm_mean, norm_std):
    #     self.norm_mean = norm_mean
    #     self.norm_std = norm_std
    #     self.to_tensor = transforms.ToTensor()
    # def __call__(self, rgb, modal_x, depth):
    #     rgb, modal_x = random_mirror_no_crop(rgb, modal_x)
    #     if config.train_scale_array is not None:
    #         rgb, modal_x, depth, scale = random_scale_no_crop(rgb, modal_x, depth, config.train_scale_array)
    #
    #     rgb = normalize(rgb, self.norm_mean, self.norm_std)
    #     modal_x = normalize(modal_x, self.norm_mean, self.norm_std)
    #     depth = self.to_tensor(depth).squeeze()
    #     depth = depth.numpy()
    #
    #     crop_size = (config.image_height, config.image_width)
    #     crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
    #
    #     p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
    #     p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
    #     depth, _ = random_crop_pad_to_shape(depth, crop_pos, crop_size, 0)
    #
    #     p_rgb = p_rgb.transpose(2, 0, 1)
    #     p_modal_x = p_modal_x.transpose(2, 0, 1)
    #     return p_rgb, p_modal_x, depth
class ValPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.to_tensor = transforms.ToTensor()

    def __call__(self, rgb, gt):
        # print(rgb.shape, gt.shape)

        # rgb = self.to_tensor(rgb).squeeze().permute(1, 2, 0)
        # rgb = 2.0 * rgb - 1.0
        # rgb = rgb.numpy()
        # modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        # for normalize
        # rgb = cv2.resize(rgb, (self.scale_size[0], self.scale_size[1]))

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)

        # modal_x = modal_x.transpose(2, 0, 1).astype(np.float32)

        # depth = self.to_tensor(depth).squeeze()
        # print(rgb.shape, gt.shape)

        return rgb, gt #modal_x, depth

    # def __init__(self):
    #     self.to_tensor = transforms.ToTensor()
    #
    # def __call__(self, rgb, gt, modal_x, depth):
    #     rgb = self.to_tensor(rgb).squeeze()
    #     rgb = 2*rgb-1
    #     x = self.to_tensor(modal_x).squeeze()
    #     modal_x = 2*x-1
    #     depth = self.to_tensor(depth).squeeze()
    #
    #     return rgb, gt, modal_x, depth

# for rgbx_eval
# class ValPre(object):
#     def __init__(self):
#         self.to_tensor = transforms.ToTensor()
#     def __call__(self, rgb, gt, modal_x, depth):
#         depth = self.to_tensor(depth).squeeze()
#         return rgb, gt, modal_x, depth
def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_root_eval': config.rgb_root_folder_eval,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_root_eval': config.gt_root_folder_eval,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder,
                    'x_root2': config.x_root_folder2,

                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)
    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)
    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn)

    return train_loader, train_sampler


def get_val_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_root_eval': config.rgb_root_folder_eval,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_root_eval': config.gt_root_folder_eval,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder_eval,
                    'x_root2': config.x_root_folder_eval2,

                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_pre = TrainPre(config.norm_mean, config.norm_std)
    # val_pre = ValPre()
    val_dataset = dataset(data_setting, 'val', val_pre, 652)

    val_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=1,
                                 num_workers=config.num_workers,
                                 drop_last=True,
                                 shuffle=is_shuffle,
                                 pin_memory=True,
                                 sampler=val_sampler
                                 )
    return val_loader, val_sampler