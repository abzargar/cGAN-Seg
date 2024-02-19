import cv2
from torch.utils.data import Dataset
import os
import numpy as np

def get_image_mask_pairs(base_folder):
    images_folder = os.path.join(base_folder, 'images')
    masks_folder = os.path.join(base_folder, 'masks')
    assert os.path.exists(images_folder), f'No images folder found in {base_folder}'
    assert os.path.exists(masks_folder), f'No masks folder found in {base_folder}'

    pairs = []
    for image_name in sorted(os.listdir(images_folder)):
        image_base_name, _ = os.path.splitext(image_name)

        # Find the corresponding mask
        for mask_name in sorted(os.listdir(masks_folder)):
            mask_base_name, _ = os.path.splitext(mask_name)

            if image_base_name == mask_base_name:
                image_path = os.path.join(images_folder, image_name)
                mask_path = os.path.join(masks_folder, mask_name)
                pairs.append((image_path, mask_path))
                break

    return pairs


class BasicDataset(Dataset):
    def __init__(self, data_sample_pairs, transforms=None,vanilla_aug=False,aug_iter=1,gen_nc=1):
        self.data_sample_pairs=data_sample_pairs
        self.transforms=transforms
        self.single_cell_mask_crop_bank=[]
        self.gen_nc=gen_nc
        if vanilla_aug:
            tmp=[]
            for _ in range(aug_iter):
                tmp+=self.data_sample_pairs
            self.data_sample_pairs=tmp


    def __len__(self):
        return len(self.data_sample_pairs)

    def preprocess(cls, img, mask,transforms):
        tensor_img,tensor_mask=transforms(img, mask)
        return tensor_img,tensor_mask

    def __getitem__(self, idx):
        img_file,mask_file = self.data_sample_pairs[idx]

        mask = cv2.imread(mask_file,0)>0
        mask = mask.astype('float32')
        if self.gen_nc==1:#2D
            img = cv2.imread(img_file,0).astype('float32')
        else:#3D
            img = cv2.imread(img_file).astype('float32')

        img=(255 * ((img - img.min()) / (img.ptp()+1e-6))).astype(np.uint8)

        tensor_img,tensor_mask = self.preprocess(img, mask,self.transforms)

        return {
            'image': tensor_img,
            'mask': tensor_mask,
        }


