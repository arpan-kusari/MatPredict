from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os 
import torch 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import math

from torchvision.ops import box_convert



"""device selection"""
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


# """define the image """
# IMAGE_PATH ='/home/yuzhench/Desktop/Research/ICCV/UNET/Feb_2/wood/250.png'



# """sam2 model selection"""
# sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
# predictor = SAM2ImagePredictor(sam2_model)

# """DINO model selection"""

# DINO_model = load_model("/home/yuzhench/Desktop/Course/ROB498-004/Project/Final_project/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
#                         "/home/yuzhench/Desktop/Course/ROB498-004/Project/Final_project/GroundingDINO/weights/groundingdino_swint_ogc.pth")
# TEXT_PROMPT = "table"
# BOX_TRESHOLD = 0.7
# TEXT_TRESHOLD = 0.7



"""helper functions"""
def get_center(bbox, H, W):
    
    x_min, y_min, x_max, y_max = bbox
 
    cx = int(((x_min + x_max) / 2) * W )
    cy = int(((y_min + y_max) / 2) * H )
    
    return cx, cy

def get_back_ground_color(image):
    point= (10,10)
    color = image[point]
    return color 

def are_pixels_similar(pixel1, pixel2, threshold=40):
     
    c1_1, c1_2, c1_3 = pixel1
    c2_1, c2_2, c2_3 = pixel2

 
    dist = math.sqrt((c1_1 - c2_1)**2 + (c1_2 - c2_2)**2 + (c1_3 - c2_3)**2)
 
    return dist < threshold

def generate_sample_point(box, image, num_samples=5, max_retries=3):
    x_min, y_min, x_max, y_max = box
    H,W = image.shape[:2]
    x_min = x_min * W
    x_max = x_max * W 

    y_min = y_min * H 
    y_max = y_max * H 

    valid_points = []
    background_pixel = get_back_ground_color (image)
    # print("background_pixel is: ", background_pixel)

    for attempt in range(max_retries):
        sampled_points = []
        for _ in range(num_samples):
            px = random.randint(int(x_min), int(x_max))
            py = random.randint(int(y_min), int(y_max))
            sampled_points.append((px, py))

      
        new_valid = []
        for (px, py) in sampled_points:
        
            pixel_bgr = image[py, px]  
            # print(pixel_bgr )
            if not are_pixels_similar(pixel_bgr, background_pixel) :
                new_valid.append((px, py))

        if new_valid:
            valid_points.extend(new_valid)
            break
        else:
            print(f"Attempt {attempt+1} found no non-background points, retrying...")

    return valid_points

    

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def get_bounding_box_from_mask(mask):
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return (x_min, y_min, x_max, y_max)


def crop_object_with_white_bg(original_image, mask):
    bbox = get_bounding_box_from_mask(mask)
    x_min, y_min, x_max, y_max = bbox

    cropped_img = original_image[y_min:y_max+1, x_min:x_max+1]
    cropped_msk = mask[y_min:y_max+1, x_min:x_max+1]

    #create a white image: 
    h, w, c = cropped_img.shape
    white_bg = np.full((h, w, c), 255, dtype=cropped_img.dtype)

    white_bg[cropped_msk > 0] = cropped_img[cropped_msk > 0]

    return white_bg




# def text_to_mask(IMAGE_PATH, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD):
#     image_source, image = load_image(IMAGE_PATH)
#     boxes, logits, phrases = predict(
#         model=DINO_model,
#         image=image,
#         caption=TEXT_PROMPT,
#         box_threshold=BOX_TRESHOLD,
#         text_threshold=TEXT_TRESHOLD
#     )


#     predictor.set_image(image_source)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_source)
    
#     xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
#     for box in xyxy:
#         print(box)
        
#         H = image_source.shape[0]
#         W = image_source.shape[1]
#         print(f"H is{H}, W is {W}")
#         # cx, cy = get_center(box, H, W)   
#         x_min, y_min, x_max, y_max = box
#         x_min = x_min * W
#         x_max = x_max * W 

#         y_min = y_min * H 
#         y_max = y_max * H  
#         box_coords = x_min, y_min, x_max, y_max
#         # show_box(box_coords, plt.gca())

#         sample_points = generate_sample_point(box, image_source, num_samples=5, max_retries=3)

        
#         print("sample_points number is: ", len(sample_points))

#     input_point = np.array(sample_points)
#     input_label = np.ones(len(input_point), dtype=np.int32)

#     # show_points(input_point, input_label,plt.gca())

#     masks, scores, logits = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=True,
#     )
#     sorted_ind = np.argsort(scores)[::-1]
#     masks = masks[sorted_ind]
#     scores = scores[sorted_ind]
#     logits = logits[sorted_ind]

#     # print(masks.shape)
    
#     # areas = [np.sum(m > 0) for m in masks]
#     # max_index = np.argmax(areas)

#     # print("which mask choosen: ", max_index)

#     # target_mask = masks[max_index]

#     max_index = 0
#     target_mask = masks[max_index]


#     # plt.imshow(target_mask, cmap='gray')  
#     # plt.title("Binary Mask Visualization")
#     # plt.show()

#     # show_masks(image_source, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
#     target_mask_cropped = crop_object_with_white_bg(image_source, target_mask)

#     plt.imshow(target_mask_cropped, cmap='gray')  
#     plt.title("Binary Mask Visualization")
#     plt.show()

#     return target_mask_cropped



# def single_process_images(input_folder, output_folder, filename):
#     file_path = os.path.join(input_folder, filename)
#     print(file_path)
#     target_mask_cropped = text_to_mask(file_path, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)
#     print(target_mask_cropped.shape)
#     output_directory = os.path.join(output_folder, filename)
#     cv2.imwrite(output_directory, cv2.cvtColor(target_mask_cropped, cv2.COLOR_RGB2BGR))


# def batch_process_images(input_folder, output_folder):
#     for filename in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, filename)
#         print(file_path)
#         target_mask_cropped = text_to_mask(file_path, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)
#         print(target_mask_cropped.shape)
#         output_directory = os.path.join(output_folder, filename)
#         cv2.imwrite(output_directory, cv2.cvtColor(target_mask_cropped, cv2.COLOR_RGB2BGR))


def find_min_area_rect_on_white_bg(img):
 
     
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
   
    cnt = max(contours, key=cv2.contourArea)
     
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.int0(box)



def order_points_clockwise(pts):
    
    center = np.mean(pts, axis=0)

    
    angles = np.arctan2(pts[:,1] - center[1],
                        pts[:,0] - center[0])

   
    sort_idx = np.argsort(angles)
    pts_sorted = pts[sort_idx]

   
    top_idx = np.lexsort((pts_sorted[:,0], pts_sorted[:,1]))[0]
    return np.roll(pts_sorted, -top_idx, axis=0)

def crop_rotated_rect(image, box):
 
    rect = order_points_clockwise(np.array(box, dtype="float32"))
    (tl, tr, br, bl) = rect

    # print (rect)
   

    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
 
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped



def remove_bg_color(image_path):
    image = cv2.imread(image_path)
    bg_color = get_back_ground_color(image)
    
    diff = np.linalg.norm(image.astype(np.int16) - bg_color.astype(np.int16), axis=-1)

    tol = 15
   
    mask = diff < tol

    img_no_bg = image.copy()
    img_no_bg[mask] = [255,255,255]
    box = find_min_area_rect_on_white_bg(img_no_bg)
    # print(box)

    # for (x, y) in box:
    #     cv2.circle(img_no_bg, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1) 

    # img_rgb = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2RGB)

    # # 3.  
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img_rgb)
    # plt.axis('off')
    # plt.show()



    cropped_target = crop_rotated_rect(img_no_bg, box)
    cropped_target = cv2.cvtColor(cropped_target, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cropped_target)
    # plt.axis('off')
    # plt.show()
 
 
    # 注意 NumPy slice: image[y_min:y_max, x_min:x_max]
    return cropped_target
  

def batch_process_cropped_image(input_folder, output_folder):
    for filename in sorted(os.listdir(input_folder)):

        
        file_path = os.path.join(input_folder, filename)
        print(file_path)
        target_mask_cropped = remove_bg_color(file_path)
        # print(target_mask_cropped.shape)
        output_directory = os.path.join(output_folder, filename)
        cv2.imwrite(output_directory, cv2.cvtColor(target_mask_cropped, cv2.COLOR_RGB2BGR))


def main():
    object_name = "swivel_chair"
    # material_name = ["concrete_1", "fabric_1", "fabric_2", "fabric_3", "fabric_4",  "leather_1", "plastic_1", "stone_1", "stone_2", "stone_3", "wood_1", "wood_2", "wood_3", "wood_4"]
    # material_name = ["concrete_1", "fabric_1", "leather_1", "plastic_1", "stone_3", "wood_1"]

    material_name = ["fabric_1", "fabric_2", "fabric_3", "leather_1", "stone_3", "wood_1"]


    # material_name = ["wood_1"]

    for mat_name in material_name:
        print(f" {mat_name} start !!!")
        # input_folder = f"/home/yuzhench/Desktop/Research/ICCV/UNET/Feb_2/Paper_Dataset/rendered/{object_name}/{mat_name}"
        # output_folder = f"/home/yuzhench/Desktop/Research/ICCV/UNET/Feb_2/Paper_Dataset/rendered_cropped/{object_name}/{mat_name}"

        input_folder = "/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/different_uv"
        output_folder = "/home/yuzhench/Desktop/Research/ICCV/UNET/test_resource/test_result/different_uv"


        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            print(f"created the directory {output_folder}")
        else:
            print(f"directory already existed: {output_folder}")

        # single_process_images(input_folder, output_folder, filename = "453.png")
        
        # remove_bg_color(input_folder)
        print("start the preprocessing")
        batch_process_cropped_image(input_folder, output_folder)

        break


        # batch_process_images(input_folder, output_folder)

        # text_to_mask(IMAGE_PATH, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)

if __name__ == "__main__":
    main()

     
