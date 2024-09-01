import os
from pickletools import uint8
import cv2
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

import h5py
def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_path_eval = setting['rgb_root_eval']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_path_eval = setting['gt_root_eval']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_path2 = setting['x_root2']

        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        # if self._file_length is not None:
        #     item_name = self._construct_new_file_names(self._file_length)[index]
        # else:
        #     item_name = self._file_names[index]

        item_name = self._file_names[index]
        # print("idx: ", index)
        # print("item name2: ", item_name)

        if self._split_name == 'train':
            rgb_path = os.path.join(self._rgb_path, remove_leading_slash(item_name.split()[0]))
            depth_gt_path = os.path.join(self._rgb_path, remove_leading_slash(item_name.split()[1]))
            gt_path = os.path.join(self._gt_path, f"label{index}.png")
            # gt_path = os.path.join(self._gt_path, f"label{index}.npy")
            # os.path.join 시 앞 parameter 에 '/'가 있어야함!
            # pred_vpd = np.load(
            #     f"/media/jslee/Data2/jslee_two/jisu/VPD/depth/file/pred_vpd/pred_vpd{index}.npy")  # 1 480 640
            # pred_zoe = np.load(
            #     f"/media/jslee/Data2/jslee_two/jisu/VPD/depth/file/pred_zoe/pred_zoe{index}.npy")  # 1 480 640

        else:
            rgb_path = os.path.join(self._rgb_path, remove_leading_slash(item_name.split()[0]))
            depth_gt_path = os.path.join(self._rgb_path, remove_leading_slash(item_name.split()[1]))
            gt_path = os.path.join(self._gt_path_eval, f"label{index}.png")
            # gt_path = os.path.join(self._gt_path_eval, f"label{index}.npy")
            # pred_vpd = np.load(f"/media/jslee/Data2/jslee_two/jisu/VPD/depth/file/pred_vpd_val/pred_vpd_val{index}.npy")  # 1 480 640
            # pred_zoe = np.load(f"/media/jslee/Data2/jslee_two/jisu/VPD/depth/file/pred_zoe_val/pred_zoe_val{index}.npy")  # 1 384 512

        # x_path = os.path.join(self._x_path, f"hha_{index}.png")
        # x_path2 = os.path.join(self._x_path2, f"hha_{index}.png")
        # if self._split_name == 'val':
        #     x_path = os.path.join(self._x_path, f"hha_val{index}.png")
        #     x_path2 = os.path.join(self._x_path2, f"hha_val{index}.png")


        # Check the following settings if necessary
        rgb_original = self._open_image(rgb_path, cv2.COLOR_BGR2RGB, dtype=np.float32)
        depth = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        # depth = self._open_image(gt_path, cv2.IMREAD_UNCHANGED, dtype=np.float32)


        # gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        gt = np.array(Image.open(gt_path), dtype=np.uint8)
        # gt = int(np.load(gt_path))

        # if self._x_single_channel:
        #     x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE,dtype=np.float32)
        #     x = cv2.merge([x, x, x])

        #     x2 = self._open_image(x_path2, cv2.IMREAD_GRAYSCALE, dtype=np.float32)
        #     x2 = cv2.merge([x2, x2, x2])

        # else:
        #     x = self._open_image(x_path, cv2.COLOR_BGR2RGB, dtype=np.float32)
        #     x2 = self._open_image(x_path2, cv2.COLOR_BGR2RGB, dtype=np.float32)


        if self.preprocess is not None:
            rgb, gt= self.preprocess(rgb_original, gt)
            depth = depth / 1000
            # rgb2, gt2, x2, depth = self.preprocess(rgb_original, gt, x2, depth)

        # if self._split_name == 'train':
        #     rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        #     gt = torch.from_numpy(np.ascontiguousarray(gt)).long()

            # x = torch.from_numpy(np.ascontiguousarray(x)).float()
            # x2 = torch.from_numpy(np.ascontiguousarray(x2)).float()

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float() #for collate_fn_val
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        # x = torch.from_numpy(np.ascontiguousarray(x)).float()
        # print(type(gt))
        # path_to_depth = '/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/nyuv2-meta-data/labels40.mat'
        # f = h5py.File(path_to_depth)
        # class_id = f['labels']
        # print(class_id.shape)
        output_dict = dict(data=rgb, label=gt, fn=str(item_name), n=len(self._file_names), ix= index, depth= depth)
                           #,pred_vpd= pred_vpd, pred_zoe= pred_zoe, modal_x=x, modal_x2=x2,)

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names
    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
