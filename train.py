import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import UNet, UNet_without_overlap, Resnet50_combined, SwinTransformerCombined
from dataset import PairedImageDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from helper import *
import wandb 
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import argparse
import random
from torch.utils.data import Subset


wandb_open = True

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()    # or nn.L1Loss()
    total_loss = 0.0
    for inputs, targets in dataloader:
        #test:
        # print(targets.shape)
        # for i in range(targets.shape[0]):
        #     show_array_as_image(inputs[i], "inputs")
        #     show_array_as_image(targets[i], "basecolor")

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    print("current_epoch has dataset_num = ",len(dataloader.dataset) )
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def generate_dataset_index(one_folder_total):
    random.seed(42)

   
    total_images = one_folder_total
    indices = list(range(total_images))
    random.shuffle(indices)

    
    train_count = int(0.8 * total_images)   
    val_count = int(0.1 * total_images) 
    test_count = total_images - train_count - val_count

 
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]
    print("one possible test indice is: ", test_indices[0])

    print("train indnex number:", len(train_indices))
    print("validation index number:", len(val_indices))
    print("test index number:", len(test_indices))
    return train_indices, val_indices, test_indices


class RandomCropTransform:
        def __init__(self, size):
            self.size = size   

        def __call__(self, img):
       
            i, j, h, w = T.RandomCrop.get_params(img, output_size=self.size)
            return TF.crop(img, i, j, h, w)


def main():

    parser = argparse.ArgumentParser(description="how to define the model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate ")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--num_epochs", type=int, default=50, help="trainig epoch")
    parser.add_argument("--model_name", type=str, 
                        choices=["UNet_without_overlap",
                                    "Resnet50_combined",
                                    "SwinTransformer_combined"],
                        default="SwinTransformer_combined", help="trainig epoch")
    parser.add_argument("--wandb_run_name", type=str, default="SwinTransformer_combined_cropped_6_mat_table_roughness")
    parser.add_argument("--if_cropped", type=bool, default=False)
    parser.add_argument("--learning_rate_schedule", type=str, default="step")
    args = parser.parse_args()
    
    print("leanring rate:", args.learning_rate)
    print("batch_size:", args.batch_size)
    print("training_epoch:", args.num_epochs)
    print("model_name:", args.model_name)

    total_epoch_num = args.num_epochs
    run_name = args.wandb_run_name
    if wandb_open:
        wandb.init(
            project="basecolor_detection_project",    
            name=run_name,  
            config={"epochs": args.num_epochs, "batch_size": 8, "learning_rate":2e-4, "overlap":False, "input_size": 256 }
        )

    
    # data_path
    train_input_dir = "/home/yuzhench/Desktop/Research/ICCV/UNET/Feb_2/Paper_Dataset/rendered_cropped"
    train_target_dir = "Feb_2/ground_truth_roughness"
    # folder_index = ["wood", "stone", "stone_3", "fabric", "wood_3"]
    # basecolor_index = ["wood","stone", "stone_3", "fabric", "wood_3"]
    folder_index = ["wood_1", "stone_3", "plastic_1", "leather_1", "concrete_1", "fabric_1"]
    basecolor_index = ["wood_1", "stone_3", "plastic_1", "leather_1", "concrete_1", "fabric_1"]

    # object_index = ["table", "sofa", "chair", "vase", "door"]
    object_index = ["table"]

    # val_input_dir = "./dataset/val/input"
    # val_target_dir = "./dataset/val/target"

    #define the iinput image size:
    input_img_height = 1080
    input_img_width = 1920
    scale_factor = 2
    #create the transform s
     
    if args.model_name == "SwinTransformer_combined":
        input_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    else:
        input_transform = T.Compose([
            T.Resize((int(224), int (224))),
            T.ToTensor(),
        ])
    
    target_transform = T.Compose([
            T.Resize((int(224), int (224))),
            T.ToTensor(),
    ])

    if args.if_cropped: 
        my_crop = T.Compose([
            RandomCropTransform((224, 224)),
            T.ToTensor()
        ])
    else:
        my_crop = None
     
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use the device: ", device)

    # generate the training path 
    total_dataset = PairedImageDataset(train_input_dir, train_target_dir, input_transform, target_transform, my_crop, folder_index, object_index, basecolor_index)
    # total_dataset = PairedImageDataset(train_input_dir, train_target_dir, my_transform,None,  folder_index,basecolor_index)

    print("the length of the total_dataset is: ", len(total_dataset)) # num_class * 512 (each class sample number)
   

    #generate the training, validation, test dataset 
    num_img = len(total_dataset)
    train_size = int(0.8 * num_img)
    val_size = int(0.1 * num_img)
    test_size = num_img - train_size - val_size


    """start to seperate all dataset"""
    train_indices, val_indices, test_indices = generate_dataset_index(len(object_index)*len(folder_index) * 512)

    # print(len(train_indices))
    # exit()
    train_dataset = Subset(total_dataset, train_indices)
    validation_dataset = Subset(total_dataset, val_indices)
    test_dataset = Subset(total_dataset, test_indices)


    # """for iamge testing"""
    # for index in range (0, 800, 50): 
    #     input_tensor, basecolor_tensor = train_dataset[index]

    #     # 如果返回的是 Tensor，且 shape 为 [C, H, W]，则需要转置成 [H, W, C] 才能正确显示
    #     input_img = input_tensor.permute(1, 2, 0).numpy()
    #     basecolor_img = basecolor_tensor.permute(1, 2, 0).numpy()

    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(input_img)
    #     plt.title("Input Image")

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(basecolor_img)
    #     plt.title("Basecolor Image")

    #     plt.show()

    # exit()

    # train_dataset, validation_dataset, test_dataset = random_split(total_dataset, [train_size, val_size, test_size])


    print("the length of the train_dataset is: ", len(train_dataset)) #512 * N
    print("the length of the validation_dataset is: ", len(validation_dataset)) #512 * N
    print("the length of the test_dataset is: ", len(test_dataset)) #512 * N
    print(" ")
    print("input image size: ", train_dataset[0][0].shape) # torch.Size([3, 540, 960])
    print("output image size: ", train_dataset[0][1].shape) # torch.Size([3, 540, 960])

 
 
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


    print("Number of train data loader:", len(train_loader))
    print("Number of validation data loader:", len(val_loader))
    print("Number of test data loader:", len(test_loader))

    
    # create the model
    # model = UNet(in_channels=3, out_channels=3)

    if args.model_name == "UNet_without_overlap":
        model = UNet_without_overlap(in_channels=3, out_channels=3)

     

    if args.model_name == "Resnet50_combined":
        model = Resnet50_combined(backbone='resnet50', pretrained=True, out_channels=3)


    if args.model_name == "SwinTransformer_combined":
        model = SwinTransformerCombined(pretrained=True, out_channels=3)

     
    model.to(device)

    #monitor the weight and bias update of the model:
    if wandb_open:
        wandb.watch(model, log="all") 

    # optimizer
        
    if args.model_name == "SwinTransformer_combined":
        optimizer = optim.AdamW(model.parameters(),
                                lr=1e-4,      # Swin usually 1e-4～3e-4
                                weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=2e-4)

    #learnig rate scheduler: 
    if args.learning_rate_schedule == "step":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    elif args.learning_rate_schedule == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
    elif args.learning_rate_schedule == "monitor":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    else:
        scheduler = StepLR(optimizer, step_size=5, gamma=1)


    # loop through the dataset 
    num_epochs = total_epoch_num
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # val_loss   = validate(model, val_loader, device)

        #update learning rate 
        scheduler.step()

        #update the current_learning rate to the wandb
        current_lr = scheduler.get_last_lr()[0]
        if wandb_open: 
            wandb.log({
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "epoch": epoch + 1,
            })

            # wandb.log({
            #     "epoch": epoch + 1,
            #     # "train_loss": train_loss,
            # })
        # print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f} learning rate: {current_lr}")


        # #keep the best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), "best_unet_basecolor.pth")
        torch.save(model.state_dict(), f"model/{run_name}.pth")

if __name__ == "__main__":
    main()
