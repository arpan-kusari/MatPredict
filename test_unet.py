import argparse
import torch
import torchvision.transforms as T
from PIL import Image

from model import UNet, UNet_without_overlap, Resnet50_combined, SwinTransformerCombined
import wandb 
import glob
import os

use_wandb = False
basecolor_flag = False
def main(args):
    if use_wandb == True:
        wandb.init(
            project="basecolor_detection_project",
            id="74ke6o1a",       
            resume="allow",        
        )


    image_paths = glob.glob("Feb_2/ground_truth_normal/*.png")   
    basecolor_images = [Image.open(p).convert("RGB") for p in image_paths]
    basecolor_name = [os.path.basename(p) for p in image_paths]



    #check the device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #create the model and load the weight
    # model = UNet_without_overlap(in_channels=3, out_channels=3)  
    # model = Resnet50_combined(backbone='resnet50', pretrained=True, out_channels=3)

    model = SwinTransformerCombined(pretrained=True, out_channels=3)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model weights from {args.model}")

    #input the test iamge
    input_image = Image.open(args.input).convert("RGB")
    
    transform = T.Compose([
        T.Resize((224, 224)),  #(256, 256)
        T.ToTensor()
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device) #[B, C, H, W], and B=1

    #do the prediction
    with torch.no_grad():
        output_tensor = model(input_tensor)  # [1, 3, H, W]

    #squeeze the output and visualize it
    output_tensor = output_tensor.squeeze(0).cpu()      # shape [3, H, W]
    result_image = T.ToPILImage()(output_tensor)        # save to RGB, [0,1] to [0.255]
    result_image.save(args.output)
    print(f"Saved output to {args.output}")

    batch_images_to_log = []
    # batch_images_to_log.append(
    #     wandb.Image(input_image, caption=f"Input image is:{args.input}")
    # )
    # batch_images_to_log.append(
    #     wandb.Image(result_image, caption=f"Prediction is {args.output }")
    # )
    if use_wandb == True:

        if basecolor_flag == True:
            for basecolor_gt, name in zip(basecolor_images,basecolor_name):
                batch_images_to_log.append(
                    wandb.Image(basecolor_gt, caption=f"{name} normal gt")
                )

            #upload all the images to the run 
            wandb.log({"basecolors are": batch_images_to_log})


        batch_images_to_log = []
        batch_images_to_log.append(
            wandb.Image(input_image, caption=f"Input image is:{args.input}")
        )
        batch_images_to_log.append(
            wandb.Image(result_image, caption=f"Prediction is {args.output }")
        )
        wandb.log({"wood image & prediction": batch_images_to_log})
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, default = "/home/yuzhench/Desktop/Research/ICCV/UNET/Feb_2/Paper_Dataset/rendered_cropped/table/stone_3/10.png",
                        help="Path to the input image")
    
    parser.add_argument("--output", type=str, required= False, default="resnet_result.png",
                        help="Path to save the output image")
    
    parser.add_argument("--model", type=str, default="model/SwinTransformer_combined_cropped_6_mat_table_roughness.pth",
                        help="Path to the trained model weights (.pth file)")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on: 'cuda' or 'cpu'")
    

    args = parser.parse_args()

    main(args)
