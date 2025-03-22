import os
import sys
sys.path.append(os.getcwd())
import cv2

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import argparse

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils import DiffJPEG, USMSharp
parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", type=str, default='preset/datasets/train_datasets/training_for_seesr', help='the save path of the training dataset.')

args = parser.parse_args()

gt_path = os.path.join(args.save_dir, 'gt')
lr_path = os.path.join(args.save_dir, 'lr')
sr_bicubic_path = os.path.join(args.save_dir, 'sr_bicubic')
print(gt_path)
os.makedirs(gt_path, exist_ok=True)
os.makedirs(lr_path, exist_ok=True)
os.makedirs(sr_bicubic_path, exist_ok=True)
hr_dir = '/media/ssd8T/wyw/Data/NTIRE2025/test/hr'
lr_dir = '/media/ssd8T/wyw/Data/NTIRE2025/test/lr'
hr_files = sorted(os.listdir(hr_dir))
lr_files = sorted(os.listdir(lr_dir))
usm_sharpener = USMSharp().cuda()
step = 0
for i, (hr_file, lr_file) in enumerate(zip(hr_files, lr_files)):
    step += 1
    print('process {} images...'.format(step))
    
    with open(os.path.join(hr_dir, hr_file), 'rb') as f:
        img_bytes = f.read()
    img_gt = imfrombytes(img_bytes, float32=True)
    with open(os.path.join(lr_dir, lr_file), 'rb') as f:
        img_bytes = f.read()
    img_lr = imfrombytes(img_bytes, float32=True)
    
    h, w = img_gt.shape[0:2]
    crop_pad_size = 512
    if h < crop_pad_size or w < crop_pad_size:
        pad_h = max(0, crop_pad_size - h)
        pad_w = max(0, crop_pad_size - w)
        img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        h, w = img_gt.shape[0:2]
        top = 500
        left = 250
        img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
        img_lr = img_lr[top//4:top//4 + crop_pad_size//4, left//4:left//4 + crop_pad_size//4, ...]
        
    
    img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
    img_lr = img2tensor([img_lr], bgr2rgb=True, float32=True)[0]
    img_gt = img_gt.unsqueeze(0).cuda() 
    img_gt = usm_sharpener(img_gt).squeeze(0)
    gt = torch.clamp(img_gt, 0, 1)
    lr = torch.clamp(img_lr, 0, 1)
    
    sr_bicubic = F.interpolate(lr.unsqueeze(0), size=(gt.size(-2), gt.size(-1)), mode='bicubic',).squeeze(0)

    lr_save_path =  os.path.join(lr_path,'{}.png'.format(str(step).zfill(7)))
    gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))
    sr_bicubic_save_path =  os.path.join(sr_bicubic_path, '{}.png'.format(str(step).zfill(7)))

    cv2.imwrite(lr_save_path, 255*lr.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
    cv2.imwrite(gt_save_path, 255*gt.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
    cv2.imwrite(sr_bicubic_save_path, 255*sr_bicubic.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
