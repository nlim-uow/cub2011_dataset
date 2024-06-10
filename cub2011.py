import os
import pandas as pd
from torch.utils.data import Dataset
import random 
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.datasets.utils import download_url
import cv2

class CUB200Segmentation(datasets.VisionDataset):
    import pandas as pd
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"


    def __init__(
        self,
        root: str,
        image_set: str = 'Train',
        download: bool = False,
        transform = None,
        target_transform = None,
        segments: bool = False,
        blur_size: int = 11,
        blur_sigma: int = 2,
        mask_mode: str = 'mask',
        input_size: int = 224,
    ):
        super().__init__(root)

        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.segment_url = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1'
        self.root = root
        self.filename = 'CUB_200_2011.tgz'
        self.segment_filename = 'CUB_200_2011_segmentation.tgz'
        self.md5 = '97eceeb196236b17998738112f37df78'
        self.segment_md5='4d47ba1228eae64f2fa547c47bc65255'
        self.base_dir = 'CUB_200_2011/images'
        self.img_root = os.path.join(self.root, self.base_dir)
        self.segment_dir = 'CUB_200_2011/segmentations'
        self.segment_root = os.path.join(self.root, self.segment_dir)
        directory = 'CUB_200_2011'
        self.root_dir = os.path.join(self.root, directory)
        self.toTensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(256,antialias=True),
                                            transforms.CenterCrop(input_size)])

        self.input_size = input_size        
        self.transform=transform
        self.train = image_set=='Train'
        self.segments = segments
        self.blur_size = blur_size
        self.mask_mode = mask_mode
        self.blur_sigma= blur_sigma
        
        if download:
            self._download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')            

        #assert len(self.images) == len(self.targets)

    def _load_metadata(self):
        images= pd.read_csv(os.path.join(self.root,'CUB_200_2011','images.txt'), sep=' ', names=['img_id','filepath'])
        image_class_labels= pd.read_csv(os.path.join(self.root,'CUB_200_2011','image_class_labels.txt'), sep=' ', names=['img_id','target'])
        train_test_split= pd.read_csv(os.path.join(self.root,'CUB_200_2011','train_test_split.txt'),sep=' ', names=['img_id','is_training_img'])
        data=images.merge(image_class_labels, on='img_id')
        self.data=data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img==1]
        else:
            self.data = self.data[self.data.is_training_img==0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.img_root, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.md5)
        download_url(self.segment_url, self.root_dir, self.segment_filename, self.segment_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

        with tarfile.open(os.path.join(self.root_dir, self.segment_filename), "r:gz") as tar:
            tar.extractall(path=self.root_dir)


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.root, self.img_root, sample.filepath)
        target = self.data.iloc[idx].target
        mask_path = os.path.join(self.root, self.segment_dir, sample.filepath).replace('jpg','png')
        
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255
        mask = Image.open(mask_path)
        mask = np.array(mask)/255
        

        
        
        gt_mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])
        gt_mask = np.repeat(gt_mask, 3, axis=2)
        gt_mask_bool = gt_mask > 0.5
        gt_mask = img * gt_mask_bool

        if self.segments:
            segMask = gt_mask
 
            if self.mask_mode=='mask':
                img=segMask
 
            if self.mask_mode=='whiteMask':
                img=segMask
                img[segMask==0] = 1
 
            if self.mask_mode=='meanMask':
                img=segMask + np.ones(img.shape)*[0.485, 0.456, 0.406]*(segMask==0)
            if self.mask_mode=='blurMask':
                negSegMask = (segMask == 0)
                neg_img = img*negSegMask
                img=segMask+cv2.GaussianBlur(neg_img,(self.blur_size,self.blur_size),self.blur_sigma)
        
        img = self.toTensor(img)
        gt_mask = self.toTensor(gt_mask)
        gt_mask_bool = self.toTensor(gt_mask_bool)

        
        mask = self.toTensor(mask)
        if self.train:
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                gt_mask = transforms.functional.vflip(gt_mask)
                gt_mask_bool = transforms.functional.vflip(gt_mask_bool)
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                gt_mask = transforms.functional.hflip(gt_mask)
                gt_mask_bool = transforms.functional.hflip(gt_mask_bool)
        
        

        return img.float(), target, gt_mask_bool, gt_mask
