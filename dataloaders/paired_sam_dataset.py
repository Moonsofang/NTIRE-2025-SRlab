import glob
import os
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

# from .realesrgan import RealESRGAN_degradation

class SAMPairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(SAMPairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []

        root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder +'/sr_bicubic'
            tag_path = root_folder +'/tag'
            gt_path = root_folder +'/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))


        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
        
        self.lr_list.sort()
        self.gt_list.sort()
        self.tag_path_list.sort()

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        # ram_mean = [0.485, 0.456, 0.406]
        # ram_std = [0.229, 0.224, 0.225]
        # self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)
        
        self.sam_mean = [0.485, 0.456, 0.406]
        self.sam_std = [0.229, 0.224, 0.225]
        self.sam_transforms = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.Normalize(mean=self.sam_mean, std=self.sam_std)
        ])

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):

       
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[index]
            file = open(tag_path, 'r')
            tag = file.read()
            file.close()

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)#[0,1]
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0#[-1,1]
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

        lq_img = lq_img.squeeze()

        # ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
        # ram_values = ram_values.clamp(0.0, 1.0)
        # example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))
        
        
        example["sam_values"] = self.sam_transforms(lq_img)

        return example

    def __len__(self):
        return len(self.gt_list)
    
def main():
    from transformers import AutoTokenizer
    import torch
    root_folders = '/media/ssd8T/ly/SeeSR/preset/datasets/train_datasets/training_for_seesr_1'
    
    tokenizer = AutoTokenizer.from_pretrained(
        '/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base',
        subfolder="tokenizer",
        use_fast=False,
    )

    dataset = PairedCaptionDataset(
        root_folders=root_folders,
        tokenizer=tokenizer,
    )

    print(f'Dataset size: {len(dataset)}')

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f'Sample {i}:')
        print(f"  Conditioning Pixel Values Shape: {sample['conditioning_pixel_values'].shape}")#3, 512, 512
        print(f"  Pixel Values Shape: {sample['pixel_values'].shape}")#3, 512, 512
        print(f"  Input IDs Shape: {sample['input_ids'].shape}")#77
        print(f"  sam Values Shape: {sample['sam_values'].shape}")#3, 1024, 1024
        print(f"  Tag: {sample['input_ids']}") 

if __name__ == '__main__':
    main()