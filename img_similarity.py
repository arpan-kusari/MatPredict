from image_similarity_measures.evaluate import evaluation

# scores = evaluation(org_img_path="/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/resized_ground_truth_224/leather_1.png", 
#            pred_img_path="/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/swintransformer/leather.png", 
#            metrics=["rmse","ssim","sam"])

# print(scores)


"""roughness"""
scores = evaluation(org_img_path="/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/resized_ground_truth_roughness_224/fabric_1.png", 
           pred_img_path="/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/swintransformer/roughness/fabric.png", 
           metrics=["rmse","ssim","sam"])

print(scores)

# from PIL import Image  
# def resize_and_save(src_path, dst_path):
#     """Resize *src_path* to 224×224 and write it under *dst_dir*.
#        Returns the path of the written file."""
#     # 1. open image
#     img = Image.open(src_path).convert("RGB")

#     # 2. resize (bilinear resampling is PIL default; change if needed)
#     img = img.resize((256, 256), Image.BILINEAR)


#     # 4. save (Pillow infers format from suffix)
#     img.save(dst_path)
#     return dst_path


# if __name__ == "__main__":

#     folder_index = ["wood_1", "stone_3", "plastic_1", "leather_1", "concrete_1", "fabric_1"]
    
#     for mat in folder_index:
#         src = f"/home/yuzhench/Desktop/Research/ICCV/UNET/Feb_2/ground_truth_roughness/{mat}.png"
#         dst_folder = f"/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/resized_ground_truth_roughness_256/{mat}.png"
#         out_path = resize_and_save(src, dst_folder)
#         print(f"Resized image written to {out_path}")