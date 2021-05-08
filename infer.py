'''
Inference code for VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import tqdm
import json
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mask_util
import cv2

from net.lwef4_softlabel_b3_tc import LWef as RefineNet
refine_ckpt_path = './refine_060.ckpt'
refine_ckpt = torch.load(refine_ckpt_path)
refine_model = RefineNet(2, arch='tf_efficientnet_b3', pretrained=False)
refine_model.load_state_dict(refine_ckpt['state_dict'], strict=True)
refine_model.cuda().eval()

BBOX_SCALE = 1.2
REFINE_SHAPE = (384, 384)
palette = Image.open('/versa/dataset/TIANCHI2021/PreRoundData/Annotations/606396/00001.png').getpalette()



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_ins', default=10, type=int,
                        help="Number of instances")
    parser.add_argument('--num_queries', default=360, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--img_path', default='data/ytvos/valid/JPEGImages/')
    parser.add_argument('--ann_path', default='data/ytvos/annotations/instances_val_sub.json')
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
         'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
         'train','horse','turtle','bear','motorbike','giraffe','leopard',
         'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
         'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
         'tennis_racket']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600]]
transform = T.Compose([
    T.Resize(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def split_func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def main(args):
    device = torch.device(args.device)
    print('Using device: ', device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    num_frames = args.num_frames
    num_ins = args.num_ins
    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path, map_location='cpu')['model']
        # state_dict["mask_head.lay5.weight"] = torch.zeros(24, 48, 3, 3)
        # state_dict["mask_head.lay5.bias"] = torch.zeros(24)
        # del state_dict["mask_head.conv_offset.weight"]
        # del state_dict["mask_head.conv_offset.bias"]
        # del state_dict["mask_head.dcn.weight"]
        model.load_state_dict(state_dict)
        folder = args.img_path
        videos = os.listdir(folder)
        vis_num = len(videos)
        for i in range(vis_num):
            print("Process video: ", i, videos[i])
            total_names = os.listdir(os.path.join(folder, videos[i]))
            for file_names in split_func(total_names, num_frames):
                length = len(file_names)
                img_set = []
                raw_img = []
                if length < num_frames:
                    clip_names = file_names * (math.ceil(num_frames / length))
                    clip_names = clip_names[:num_frames]
                else:
                    clip_names = file_names[:num_frames]
                for k in range(num_frames):
                    im = Image.open(os.path.join(folder, videos[i], clip_names[k]))
                    img_set.append(transform(im).unsqueeze(0).to(device))
                    raw_img.append(np.array(im))
                img = torch.cat(img_set, 0)
                # inference time is calculated for this operation
                outputs = model(img)
                # end of model inference
                logits, boxes, masks = outputs['pred_logits'].softmax(-1)[0, :, :-1], outputs['pred_boxes'][0], \
                                       outputs['pred_masks'][0]
                pred_masks = F.interpolate(masks.reshape(num_frames, num_ins, masks.shape[-2], masks.shape[-1]),
                                           (im.size[1], im.size[0]), mode="bilinear", align_corners=False).sigmoid().cpu().detach().numpy() > 0.5
                pred_logits = logits.reshape(num_frames, num_ins, logits.shape[-1]).cpu().detach().numpy()
                pred_masks = pred_masks[:length]
                pred_logits = pred_logits[:length]
                pred_scores = np.max(pred_logits, axis=-1)

                for n in range(length):
                    img_array = raw_img[n]
                    h0, w0 = img_array.shape[:2]
                    blend_mask = np.zeros((h0, w0), dtype=np.uint8)
                    for m in range(num_ins):
                        if pred_masks[:, m].max() == 0:
                            continue
                        # if pred_scores[n, m] < 0.001:
                        #     continue
                        if pred_scores[n, m] > 0.1:
                            mask = (pred_masks[n, m]).astype(np.uint8)

                            x, y, w, h = cv2.boundingRect(mask)
                            ori_x = x - w * (BBOX_SCALE - 1) / 2
                            ori_y = y - h * (BBOX_SCALE - 1) / 2
                            leftOffset = int(min(ori_x, 0))
                            topOffset = int(min(ori_y, 0))
                            x = int(ori_x - leftOffset)
                            y = int(ori_y - topOffset)
                            w = int(min(w0 - x, BBOX_SCALE * w))
                            h = int(min(h0 - y, BBOX_SCALE * h))

                            img_crop = img_array[y:y + h, x:x + w, :]
                            mask_crop = mask[y:y + h, x:x + w]
                            img_x = cv2.resize(img_crop, REFINE_SHAPE, cv2.INTER_LINEAR)
                            mask_x = cv2.resize(mask_crop, REFINE_SHAPE, cv2.INTER_NEAREST)
                            input_x = np.concatenate([img_x / 255., mask_x[:, :, np.newaxis]], axis=2)
                            input_x = torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0).float()
                            pred = refine_model(input_x.cuda())
                            pred = pred.squeeze().detach().cpu().numpy()
                            pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)

                            mask_ = np.zeros_like(mask)
                            mask_[y:y + h, x:x + w] = pred

                            mask = mask_

                            img_array[mask == 1] = img_array[mask == 1] * 0.5 + np.array(COLORS[m]) * 255 * 0.5

                            mask[mask == 1] = m + 1
                            blend_mask += mask
                            blend_mask[blend_mask > (m + 1)] = m + 1

                    blend_image = np.uint8(img_array)
                    blend_image = Image.fromarray(blend_image)

                    blend_mask = Image.fromarray(blend_mask)
                    blend_mask.putpalette(palette)

                    save_image_dir = f'./result/{videos[i]}/'
                    if not os.path.exists(save_image_dir):
                        os.makedirs(save_image_dir)
                    blend_image.save(f'{save_image_dir}/{file_names[n]}')
                    blend_mask.save(f'{save_image_dir}/{file_names[n]}'.replace('.jpg', '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
