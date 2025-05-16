import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from helper import *

class PairedImageDataset(Dataset):
 
    def __init__(self, input_dir, target_dir, input_transform=None, target_transform = None, crop=None, folder_index=None, object_index = None, basecolor_index=None, each_folder_size = 512):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.crop = crop
        self.folder_index = folder_index
        self.basecolor_index = basecolor_index
        self.object_index = object_index
        self.each_folder_size = each_folder_size
        # self.fnames = sorted(os.listdir(input_dir))

        # self.fnames = []  
        # for object_name in self.object_index:
        #     for folder_name in self.folder_index:
        #         # print("folder_name is: ", folder_name)
        #         object_folder_name = os.path.join(self.input_dir,object_name)
        #         files_in_this_folder = sorted(os.listdir(os.path.join(object_folder_name,folder_name)))
                
        #         # print("folder name is: ", folder_name)
        #         # print("files_in_this_folder number is: ", len(files_in_this_folder))
        #         full_paths = [os.path.join(folder_name, temp_fname) for temp_fname in files_in_this_folder]
        #         self.fnames.extend(full_paths)
        
        # print(len(self.fnames))


 
        self.basecolors = []
        for folder_name in folder_index:
            bc_path = os.path.join(self.target_dir, f"{folder_name}.png")
            bc_img = Image.open(bc_path).convert("RGB")
            # 这里不做 transform，只在 __getitem__ 时按需裁剪/Resize
            self.basecolors.append(bc_img)

       
        self.samples = []
        for bc_idx, folder_name in enumerate(folder_index):
            for obj_name in object_index:
                folder = os.path.join(self.input_dir, obj_name, folder_name)
                for fname in sorted(os.listdir(folder)):
                    self.samples.append(
                        (os.path.join(folder, fname), bc_idx)
                    )
 
        assert len(self.samples) == len(folder_index) * len(object_index) * each_folder_size, \
            "样本数与 each_folder_size 不符，请检查数据集。"
     
      

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):
    #     # in_name = self.fnames[idx]
    #     # in_path = os.path.join(self.input_dir, in_name)

    #     ground_truth_basecolor_index = idx // self.each_folder_size
    #     screen_shot_path = self.fnames[idx]

    #     basecolor_path = []
    #     # print("basecolor_index is: ", self.basecolor_index)
    #     for basecolor_name in self.basecolor_index:
    #         curr_basecolor_name = os.path.join(self.target_dir, f"{basecolor_name}.png")
    #         basecolor_path.append(curr_basecolor_name)


    #     # read the image
    #     input_img = Image.open(os.path.join(self.input_dir,screen_shot_path)).convert("RGB")
    #     # print("basecolor path is: ", basecolor_path[ground_truth_basecolor_index])
        
    #     target_img = Image.open(basecolor_path[ground_truth_basecolor_index]).convert("RGB")
        
    #     if self.transform:
    #         input_tensor = self.transform(input_img)
    #     if self.crop:
    #         target_tensor = self.crop(target_img)
    #     if not self.crop: 
    #         target_tensor = self.transform(target_img)
         
        
    #     return input_tensor, target_tensor


    def __getitem__(self, idx):
        img_path, bc_idx = self.samples[idx]

   
        input_img = Image.open(img_path).convert("RGB")
        input_tensor = self.input_transform(input_img) if self.input_transform else T.ToTensor()(input_img)

 
        target_img = self.basecolors[bc_idx]

        if self.crop:
            target_tensor = self.crop(target_img)
        else:
            target_tensor = self.target_transform(target_img) if self.target_transform else T.ToTensor()(target_img)

        return input_tensor, target_tensor
