# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Shift window testing and flip testing is modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

from IEBins.iebins.networks.NewCRFDepth import NewCRFDepth
from IEBins.iebins.utils import post_process_depth, flip_lr, compute_errors
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib
import torch.nn as nn
from custom_resnet import Custom_resnet
from transformers import AutoImageProcessor, ResNetForImageClassification, SegformerForImageClassification, SegformerForSemanticSegmentation


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def load_iebins(args):
    # CRF model
    model_ie = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model_ie.train()
    num_params = sum([np.prod(p.size()) for p in model_ie.parameters()])
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model_ie.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))
    model_ie = torch.nn.DataParallel(model_ie)
    model_ie.cuda()
    print("== Model Initialized")
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model_ie.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
    cudnn.benchmark = True
    # ===== Evaluation ======
    return model_ie

def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        print("================using gpu================")
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.exp_name)
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)
    
    # if args.do_evaluate:
    #     result_metrics = {}
    #     for metric in metric_name:
    #         result_metrics[metric] = 0.0
    #     result_metrics_ie = {}
    #     for metric in metric_name:
    #         result_metrics_ie[metric] = 0.0

    print("\n1. Define Model")
    # ==========================iebins==========================
    model_ie = load_iebins(args)
    model_ie.eval()
    # ==========================================================
    # ===========================mim============================
    model = GLPDepth(args=args).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()
    # ==========================================================

    print("\n2. Define Dataloader")
    if args.dataset == 'imagepath': # not for do_evaluate in case of imagepath
        dataset_kwargs = {'dataset_name': 'ImagePath', 'data_path': args.data_path}
    else:
        dataset_kwargs = {'data_path': args.data_path, 'dataset_name': args.dataset,
                          'is_train': False}

    test_dataset = get_dataset(**dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)

    dataset_kwargs_nyu = {'data_path': '../data/nyu/', 'dataset_name': 'nyudepthv2',
                      'is_train': False}
    test_dataset2 = get_dataset(**dataset_kwargs_nyu)
    test_loader_nyu = DataLoader(test_dataset2, batch_size=1, shuffle=False,
                             pin_memory=True)                                      # nyu_dataloader
    # ====================================================================================
    # validate(model_ie=model_ie, model=model, test_loader=test_loader, device=device, args=args)
    # make_histogram(data_loader=test_loader, args=args)
    # validate_rgbx_model(data_loader=test_loader, args=args, method='softmax', save_mask=False)
    # validate_rgbx_cls_model(data_loader=test_loader, args=args, method='softmax',save_mask=False)
    # validate_resnet(data_loader=test_loader, args=args, save_mask=False)
    # validate_models(data_loader=test_loader, args=args) # just for check
    # resizer(data_loader=test_loader, args=args)
    # mask_per_sence(data_loader=test_loader, args=args)
    nyu_kitti_model(data_loader=test_loader, data_loader2=test_loader_nyu, args=args)
    # number_nyu_kitti_rmse(data_loader=test_loader, args=args)
    # nyu_kitti_model_resnet(data_loader=test_loader, args=args)
    # weight_fig(data_loader=test_loader, args=args)
    # ====================================================================================

def validate(model_ie, model, test_loader, device, args):
    print("\n3. Inference & Evaluate")
    if args.do_evaluate:
        result_metrics = {}
        for metric in metric_name:
            result_metrics[metric] = 0.0
        result_metrics_ie = {}
        for metric in metric_name:
            result_metrics_ie[metric] = 0.0

    for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        input_RGB = batch['image'].to(device)
        filename = batch['filename']
        input_RGB_ie = batch['image_ie'].to(device)  # torch.Size([1, 352, 1216, 3])
        # input_RGB_ie = torch.autograd.Variable(batch['image_ie'].to(device)) # torch.Size([1, 352, 1216, 3])

        with torch.no_grad():
            # ===============iebins===============
            pred_ie, _, _ = model_ie(input_RGB_ie)
            image_flipped = flip_lr(input_RGB_ie)
            pred_ie_flipped, _, _ = model_ie(image_flipped)
            pred_d_ie = post_process_depth(pred_ie[-1], pred_ie_flipped[-1])
            # ===============mim-depth===============
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size - 1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i * interval:i * interval + h])
                    sliding_masks[..., :, i * interval:i * interval + h] += 1
                input_RGB = torch.cat(sliding_images, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
            pred = model(input_RGB)
        pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = pred_d.shape[0] // 2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3])) / 2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i * interval:i * interval + h] += pred_d[i:i + 1]
            pred_d = pred_s / sliding_masks

        if args.do_evaluate:
            depth_gt = batch['depth'].to(device)
            depth_gt_ie = depth_gt
            pred_d, pred_d_ie = pred_d.squeeze(), pred_d_ie.squeeze()
            # print("pred shape", pred_d.shape, pred_d_ie.shape)    # torch.Size([352, 1216]) torch.Size([352, 1216])
            # np.save(f"./file/pred_mim_train/pred_mim{batch_idx}", pred_d.cpu().numpy())
            # np.save(f"./file/pred_ieb_train/pred_ieb{batch_idx}", pred_d_ie.cpu().numpy())

            depth_gt1 = depth_gt.squeeze()
            depth_gt2 = depth_gt_ie.squeeze()
            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt1)
            pred_crop_ie, gt_crop_ie = metrics.cropping_img(args, pred_d_ie, depth_gt2)
            # print("pred shape", pred_crop.shape, pred_crop_ie.shape)  # random?
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            computed_result_ie = metrics.eval_depth(pred_crop_ie, gt_crop_ie)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
                result_metrics_ie[metric] += computed_result_ie[metric]

        # if args.save_eval_pngs:
        #     save_path = os.path.join(result_path, filename[0])
        #     if save_path.split('.')[-1] == 'jpg':
        #         save_path = save_path.replace('jpg', 'png')
        #     pred_d = pred_d.squeeze()
        #     if args.dataset == 'nyudepthv2':
        #         pred_d = pred_d.cpu().numpy() * 1000.0
        #         cv2.imwrite(save_path, pred_d.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #     else:
        #         pred_d = pred_d.cpu().numpy() * 256.0
        #         cv2.imwrite(save_path, pred_d.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #
        # if args.save_visualize:
        #     save_path = os.path.join(result_path, filename[0])
        #     pred_d_numpy = pred_d.squeeze().cpu().numpy()
        #     pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        #     pred_d_numpy = pred_d_numpy.astype(np.uint8)
        #     pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        #     cv2.imwrite(save_path, pred_d_color)

        # logging.progress_bar(batch_idx, len(test_loader), 1, 1)

    if args.do_evaluate:
        for key in result_metrics.keys():
            result_metrics[key] = result_metrics[key] / (batch_idx + 1)
        display_result = logging.display_result(result_metrics)
        if args.kitti_crop:
            print("\nCrop Method: ", args.kitti_crop)
        print('mim: ', display_result)
        for key in result_metrics_ie.keys():
            result_metrics_ie[key] = result_metrics_ie[key] / (batch_idx + 1)
        display_result_ie = logging.display_result(result_metrics_ie)
        if args.kitti_crop:
            print("\nCrop Method: ", args.kitti_crop)
        print('iebins: ', display_result_ie)
    print("Done")

def make_histogram(data_loader, args):

    histogram = torch.zeros(1000).cuda()  #numpy can't be passed to GPU
    mim_1 = torch.zeros(1).cuda()
    ieb_1 = torch.zeros(1).cuda()
    x_range = torch.linspace(-2, 2, 1000)

    for batch_idx, batch in tqdm(enumerate(data_loader) ,total = len(data_loader)):
        depth_gt = batch['depth'].cuda().squeeze()
        pred_ieb = batch['pred_ieb'].cuda().squeeze()
        pred_mim = batch['pred_mim'].cuda().squeeze()
        # pred_zoe = batch['pred_zoe_seg'].cuda().squeeze()
        # pred_vpd = batch['pred_vpd_seg'].cuda().squeeze()

        a = torch.abs(depth_gt - pred_mim)
        b = torch.abs(depth_gt - pred_ieb)

        diff = a - b
        hist = torch.histc(diff.flatten(), bins= 1000, min = -2, max = 2)
        histogram += hist

        better_mim = torch.sum(diff <= x_range[315])     #after find_th
        better_ieb = torch.sum(diff >= x_range[676])

        mim_1 += better_mim
        ieb_1 += better_ieb

        # mim_ = torch.zeros_like(diff)

        mim_ = torch.full_like(diff, 2)
        mim_[diff <= x_range[315]] = 0
        mim_[diff > x_range[676]] = 1

        # palette = [0, 0, 0, 255, 0, 0, 0, 128, 0]
        palette = [255, 0, 0, 0, 125, 0, 0, 0, 0]
        mim_ = mim_.cpu().numpy()
        png = Image.fromarray(mim_).convert('P')
        png.putpalette(palette)
        png.save(f'/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label2/val/label{batch_idx}.png')


    histogram = histogram.cpu()
    th1 = find_th(x_range, histogram, percent=0.1)
    th2 = find_th(x_range, histogram, percent=0.9)
    print("th: ", th1, th2)
    plt.bar(x_range, histogram, width=0.03)
    plt.title('histogram of error difference')
    plt.xlabel('e_diff(meter)')
    plt.ylabel('frequency')

    # plt.savefig("./histogram_10000_5")

    print(f"sum of mim1: {mim_1}")
    print(f"sum of ieb1: {ieb_1}")

def find_th(x_range, histogram, percent):
    cdf = torch.cumsum(histogram, dim= 0)
    threshold = percent * torch.sum(histogram)
    index = np.where(cdf >= threshold)[0][0]
    print("index :", index)
    return x_range[index]

def validate_rgbx_model(data_loader, args, method, save_mask):
    from rgbx.models.builder import EncoderDecoder as segmodel
    from rgbx.utils.pyt_utils import load_model
    from rgbx.config import config


    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    network = segmodel(cfg=config, criterion=None, norm_layer = nn.BatchNorm2d)
    model = load_model(network,
                       "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/log_NYUDepthv2_mit_b4/checkpoint/epoch-2.pth")
    model = model.cuda()
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            image = batch['image_ie'].cuda()
            depth_gt = batch['depth'].cuda().squeeze()
            pred_ieb = batch['pred_ieb'].cuda().squeeze()
            pred_mim = batch['pred_mim'].cuda().squeeze()
            # path = batch['img_path']
            # image = normalize(image)
            out = model(image)  #1 3 480 640


            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            mask = out.argmax(dim=1).squeeze()

            # save mask
            if save_mask:
                palette = [255, 0, 0, 0, 128, 0, 0, 0, 0]
                # palette = [0, 0, 0, 255, 0, 0, 0, 128, 0]
                mim_ = mask.cpu().numpy().astype(np.uint8)
                png = Image.fromarray(mim_).convert('P')
                png.putpalette(palette)
                result_path = '/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/mask'
                logging.check_and_make_dirs(result_path)
                png.save(result_path + f'/label_val{batch_idx}.png')

                prob = proba_out.cpu().numpy()
                mim_prob = prob[0,:,:]
                mim_prob = colorize(mim_prob, 0 ,1)
                Image.fromarray(mim_prob).save(os.path.join(result_path, f"label_val{batch_idx}_prob.png"))

            if method == 'argmax':
                pred = torch.full_like(pred_mim, 2)
                m = (mask == 0)
                i = (mask == 1)
                pred[i] = pred_ieb[i]
                pred[m] = pred_mim[m]
            elif method == 'softmax':
                pred = proba_out[0, :, :] * pred_mim + proba_out[1, :, :] * pred_ieb
            elif method == 'mean':
                pred = 0.5 * pred_mim + 0.5 * pred_ieb

            # save_path = os.path.join("/media/jslee/Data2/jslee_two/jisu/VPD/depth/ffff", f"a{batch_idx}.png")
            # pred_d_numpy = pred.cpu().numpy()
            # pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            # pred_d_numpy = pred_d_numpy.astype(np.uint8)
            # pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_OCEAN)
            # cv2.imwrite(save_path, pred_d_color)

            pred_crop, gt_crop = metrics.cropping_img(args, pred, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
        if args.do_evaluate:
            for key in result_metrics.keys():
                result_metrics[key] = result_metrics[key] / (batch_idx + 1)
            display_result = logging.display_result(result_metrics)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print('result: ', display_result)
        print("Done")
def validate_rgbx_cls_model(data_loader, args, method, save_mask):
    from rgbx.models.builder import EncoderDecoder as segmodel
    from rgbx.utils.pyt_utils import load_model
    from rgbx.config import config
    mim_list, ieb_list = [], []


    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    network = segmodel(cfg=config, criterion=None, norm_layer = nn.BatchNorm2d)
    model = load_model(network,
                       # "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/log_NYUDepthv2_mit_b2_6e-5_eo20_warm2_675/checkpoint/epoch-3.pth")
                       "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/log_NYUDepthv2_mit_b4_8e-4_ep15_warm3_675_best/checkpoint/epoch-7.pth")
    model = model.cuda()
    weight_kitti = []
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            image = batch['image_ie'].cuda()
            depth_gt = batch['depth'].cuda().squeeze()
            pred_ieb = batch['pred_ieb'].cuda().squeeze()
            pred_mim = batch['pred_mim'].cuda().squeeze()
            # path = batch['img_path']
            # image = normalize(image)
            out = model(image)
            # print(out.shape) torch.Size([1, 2])
            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            # mim_list.append(proba_out[0].cpu())
            # ieb_list.append(proba_out[1].cpu())
            weight_kitti.append([proba_out[0].cpu(), proba_out[1].cpu()])
            mask = proba_out.argmax().squeeze()
            if method == 'argmax':
                if mask == 0:
                    pred = pred_mim
                else:
                    pred = pred_ieb
            else:
                pred = proba_out[0] * pred_mim + proba_out[1] * pred_ieb
            pred_crop, gt_crop = metrics.cropping_img(args, pred, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
        if args.do_evaluate:
            for key in result_metrics.keys():
                result_metrics[key] = result_metrics[key] / (batch_idx + 1)
            display_result = logging.display_result(result_metrics)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print('result: ', display_result)
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.hist(mim_list, color='red', bins=100, range=[0, 1], label='Mim')
        # plt.legend()
        # plt.subplot(2, 1, 2)
        # plt.hist(ieb_list, color='blue', bins=100, range=[0, 1], label='Ieb')
        # plt.xlabel('probability')
        # plt.legend()
        # plt.savefig('./mim_ieb.png')
        # plt.close()
        np.savetxt("./weight_kitti", weight_kitti)
        print("Done")

def validate_resnet(data_loader, args, save_mask):
    from rgbx.models.builder import EncoderDecoder as segmodel
    from rgbx.utils.pyt_utils import load_model
    from rgbx.config import config


    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    network = Custom_resnet(resnet_model)
    model = load_model(network,
                       "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/log_NYUDepthv2_mit_b4/checkpoint/epoch-1.pth")
    model = model.cuda()
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            image = batch['image_ie'].cuda()
            depth_gt = batch['depth'].cuda().squeeze()
            pred_ieb = batch['pred_ieb'].cuda().squeeze()
            pred_mim = batch['pred_mim'].cuda().squeeze()
            # path = batch['img_path']
            # image = normalize(image)
            out = model(image).logits
            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            mask = proba_out.argmax().squeeze()
            # print(out, proba_out, mask)

            if mask == 0:
                print("mim")
                pred = pred_mim
            else:
                pred = pred_ieb
            # save mask
            if save_mask:
                palette = [255, 0, 0, 0, 128, 0, 0, 0, 0]
                # palette = [0, 0, 0, 255, 0, 0, 0, 128, 0]
                mim_ = mask.cpu().numpy().astype(np.uint8)
                png = Image.fromarray(mim_).convert('P')
                png.putpalette(palette)
                result_path = '/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/mask'
                logging.check_and_make_dirs(result_path)
                png.save(result_path + f'/label_val{batch_idx}.png')

                prob = proba_out.cpu().numpy()
                mim_prob = prob[0,:,:]
                mim_prob = colorize(mim_prob, 0 ,1)
                Image.fromarray(mim_prob).save(os.path.join(result_path, f"label_val{batch_idx}_prob.png"))


            pred_crop, gt_crop = metrics.cropping_img(args, pred, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
        if args.do_evaluate:
            for key in result_metrics.keys():
                result_metrics[key] = result_metrics[key] / (batch_idx + 1)
            display_result = logging.display_result(result_metrics)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print('result: ', display_result)
        print("Done")

def normalize(img):
    # pytorch pretrained model need the input range: 0-1
    # print(img.shape, "------------------") # 1  480 640 3
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    img = img.type(torch.float64) / 255.0
    img = img - mean
    img = img / std
    img = img.permute(0, 3, 1, 2).type(torch.float32)
    return img
def colorize(value, vmin=None, vmax=None, cmap='rainbow_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    # Converts a depth map to a color image.
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.colormaps.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def validate_models(data_loader, args):
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred_ieb = batch['pred_ieb'].cuda().squeeze()
            pred_mim = batch['pred_mim'].cuda().squeeze()
            depth_gt = batch['depth'].cuda().squeeze()
            pred = pred_mim * 0.6 + pred_ieb * 0.4
            pred_crop, gt_crop = metrics.cropping_img(args, pred, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
        if args.do_evaluate:
            for key in result_metrics.keys():
                result_metrics[key] = result_metrics[key] / (batch_idx + 1)
            display_result = logging.display_result(result_metrics)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print('result: ', display_result)
        print("Done")
def resizer(data_loader, args):
    from resizer import Resizer
    import torchvision.transforms as T
    import torch.nn.functional as F

    from PIL import Image
    transform = T.ToPILImage()
    model = Resizer()
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        img = batch['image']
        depth = batch['depth']
        # print(img.shape, depth.shape) torch.Size([1, 3, 352, 1216]) torch.Size([1, 352, 1216])
        post_img = model(img)

        # 그림이 잘 안 나옴
        # img_resize = transform(post_img.squeeze())
        # img_resize.save(f"./resize/image_path{batch_idx}.png")
        # img_ori = transform(img.squeeze())
        # img_ori.save(f"./resize/image_path{batch_idx}_ori.png")

        # 그림 잘 나옴
        # img_inter = F.interpolate(img, size =(480, 640), mode = 'bilinear')
        # img_inter = transform(img_inter.squeeze())
        # img_inter.save(f"./resize/image_path{batch_idx}.png")

        # 그림 잘 나옴
        crop = T.RandomResizedCrop(size=(480, 640))
        image_crop = crop(img)
        image_crop = transform(image_crop.squeeze())
        image_crop.save(f"./resize/image_path{batch_idx}.png")
def mask_per_sence(data_loader, args):
    num_ieb = num_mim = index = 0
    img_path_total = ''
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        depth_gt = batch['depth'].cuda().squeeze()
        pred_ieb = batch['pred_ieb'].cuda().squeeze()  # !!!for validation!!!
        pred_mim = batch['pred_mim'].cuda().squeeze()

        pred_crop_mim, gt_crop1 = metrics.cropping_img(args, pred_mim, depth_gt)
        pred_crop_ieb, gt_crop2 = metrics.cropping_img(args, pred_ieb, depth_gt)
        diff_mim = pred_crop_mim - gt_crop1
        diff_ieb = pred_crop_ieb - gt_crop2
        rmse1 = torch.sqrt(torch.mean(torch.pow(diff_mim, 2)))
        rmse2 = torch.sqrt(torch.mean(torch.pow(diff_ieb, 2)))
        if rmse1 <= rmse2:
            num_mim += 1
            np.save(f"/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label_scene2/train/label{index}", 0)
            img_path = batch['img_path']
            img_path_total += (img_path[0] + '\n')
            index += 1

            # pred = pred_mim
        elif rmse1 > rmse2 and num_ieb < 675:
            num_ieb += 1
            np.save(f"/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label_scene2/train/label{index}", 1)
            img_path = batch['img_path']
            img_path_total += (img_path[0] + '\n')
            index += 1

            # pred = pred_ieb
        else:
            continue
    #     pred_crop, gt_crop = metrics.cropping_img(args, pred, depth_gt)
    #     computed_result = metrics.eval_depth(pred_crop, gt_crop)
    #     for metric in metric_name:
    #         result_metrics[metric] += computed_result[metric]
    # if args.do_evaluate:
    #     for key in result_metrics.keys():
    #         result_metrics[key] = result_metrics[key] / (batch_idx + 1)
    #     display_result = logging.display_result(result_metrics)
    #     if args.kitti_crop:
    #         print("\nCrop Method: ", args.kitti_crop)
    #     print('result: ', display_result)
    # print("Done")
    with open("my_file.txt", "w") as file:
        file.write(img_path_total)
    print("num_mim: ", num_mim) # 0
    print("num_ieb: ", num_ieb) # 1

def number_nyu_kitti_rmse(data_loader, args):
    num_ieb = num_mim = index = 0
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    rmse_list = []
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        depth_gt = batch['depth'].cuda().squeeze()
        pred_mim = batch['pred_mim'].cuda().squeeze()
        # path = batch['img_path']
        # print(path)
        pred_crop_mim, gt_crop1 = metrics.cropping_img(args, pred_mim, depth_gt)
        diff_mim = pred_crop_mim - gt_crop1
        rmse1 = torch.sqrt(torch.mean(torch.pow(diff_mim, 2)))
        rmse_list.append((batch_idx, rmse1))
        if batch_idx >= 10:
            break
    sort_list = sorted(rmse_list, key=lambda x: x[1])
    top_rmse = sort_list[:5]
    for batch_idx, rmse in top_rmse:
        print("Batch Index:", batch_idx)







        # rmse2 = torch.sqrt(torch.mean(torch.pow(diff_ieb, 2)))
    #     if rmse1 <= rmse2:
    #         num_mim += 1
    #     else:
    #         num_ieb += 1
    # print("num_mim: ", num_mim) # 0
    # print("num_ieb: ", num_ieb) # 1
def nyu_kitti_model(data_loader, data_loader2, args):
    from rgbx.models.builder import EncoderDecoder as segmodel
    from rgbx.utils.pyt_utils import load_model
    from rgbx.config import config
    kitti_list, nyu_list = [], []
    kitti_list2, nyu_list2 = [], []

    network = segmodel(cfg=config, criterion=None, norm_layer = nn.BatchNorm2d)
    model = load_model(network,
                       "/media/jslee/Data2/jslee_two/jisu/RGBX_scene/log_NYUDepthv2_mit_b2_8e-4_5ep_1warm_best_4800/checkpoint/epoch-1.pth")
    model = model.cuda()
    nk_weight = []

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            # image = batch['image_ie'].cuda()
            image = batch['image'].cuda()
            out = model(image)
            # print(out.shape) torch.Size([1, 2])
            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            mask = proba_out.argmax().squeeze()
            kitti_list.append(proba_out[0].cpu())
            nyu_list.append(proba_out[1].cpu())
            if mask == 0:
                # print("kitti")
                pass
            else:
                # print("nyu")
                pass
            # nk_weight.append([proba_out[0].cpu(), proba_out[1].cpu()])
        for batch_idx, batch in tqdm(enumerate(data_loader2), total=len(data_loader2)):
            # image = batch['image_ie'].cuda()
            image = batch['image'].cuda()
            out = model(image)
            # print(out.shape) torch.Size([1, 2])
            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            mask = proba_out.argmax().squeeze()
            kitti_list2.append(proba_out[0].cpu())
            nyu_list2.append(proba_out[1].cpu())
            if mask == 0:
                # print("kitti")
                pass
            else:
                # print("nyu")
                pass
            # nk_weight.append([proba_out[0].cpu(), proba_out[1].cpu()])

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(kitti_list, color='blue', bins=100, range=[0, 1], label='Zoe-K')
        plt.hist(nyu_list, color='red', bins=100, range=[0, 1], label='Zoe-N')
        plt.xlabel('probability on kitti')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.hist(kitti_list2, color='blue', bins=100, range=[0, 1], label='Zoe-K')
        plt.hist(nyu_list2, color='red', bins=100, range=[0, 1], label='Zoe-N')
        plt.xlabel('probability on nyu')
        plt.legend()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('./kitti_nyu_weight.png')
        plt.close()
        # np.savetxt('./kitti_nyu_weight', nk_weight)
def nyu_kitti_model_resnet(data_loader, args):
    from rgbx.models.builder import EncoderDecoder as segmodel
    from rgbx.utils.pyt_utils import load_model
    from rgbx.config import config
    resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    network = Custom_resnet(resnet_model)
    model = load_model(network,
                       "/media/jslee/Data2/jslee_two/jisu/RGBX_scene/log_NYUDepthv2_mit_b2/checkpoint/epoch-5.pth")
    model = model.cuda()
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            # image = batch['image_ie'].cuda()
            image = batch['image'].cuda()
            out = model(image).logits
            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            mask = proba_out.argmax().squeeze()
            print(proba_out)
            if mask == 0:
                print("kitti")
                # pass
            else:
                # print("nyu")
                pass

def weight_fig(data_loader, args):
    from rgbx.models.builder import EncoderDecoder as segmodel
    from rgbx.utils.pyt_utils import load_model
    from rgbx.config import config
    mim_list, ieb_list = [], []


    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    network = segmodel(cfg=config, criterion=None, norm_layer = nn.BatchNorm2d)
    model = load_model(network,
                       "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/log_NYUDepthv2_mit_b4/checkpoint/epoch-7.pth")
                       # "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/log_NYUDepthv2_mit_b2_6e-5_eo20_warm2_675/checkpoint/epoch-3.pth")
    model = model.cuda()
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            image = batch['image_ie'].cuda()
            depth_gt = batch['depth'].cuda().squeeze()
            pred_ieb = batch['pred_ieb'].cuda().squeeze()
            pred_mim = batch['pred_mim'].cuda().squeeze()
            # path = batch['img_path']
            # image = normalize(image)
            out = model(image)
            # print(out.shape) torch.Size([1, 2])
            softmax = nn.Softmax(dim=1)
            proba_out = softmax(out).squeeze()
            # print(proba_out)
            mim_list.append(proba_out[0].cpu())
            ieb_list.append(proba_out[1].cpu())
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(mim_list, color='red', bins=100, range=[0, 1], label='Mim')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(ieb_list, color='blue', bins=100, range=[0, 1], label='Ieb')
    plt.xlabel('probability')
    plt.legend()
    plt.savefig('./mim_ieb.png')
    plt.close()

if __name__ == "__main__":
    main()
