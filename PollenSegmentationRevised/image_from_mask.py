import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import tv_tensors
from train_model import *
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F


def segment_objects(path_to_model: str, output_dir: str, image_path: str, score_threshold: float, mask_threshold: float,  name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model_instance_segmentation(2, False)
    model.to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = tv_tensors.Image(image)

    eval_transform = get_transform(train=False)
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    masks = pred['masks'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    for i in range(len(masks)):
        if scores[i] >= score_threshold:
            mask = masks[i, 0]
            box = boxes[i]
            object_img = extract_object_by_mask(mask, image, mask_threshold, box)

            output_path = os.path.join(output_dir, f"{name.split('.')[0]}_{i}.png")
            object_img.save(output_path)


def extract_object_by_mask(mask: np.array, image: np.array, mask_thershold=0.5, box=None) -> Image:
    if type(box) != np.array:
        masks = np.array([mask])
        box = masks_to_boxes(torch.from_numpy(masks))[0]

    extracted_obj = cv2.bitwise_and(image, image, mask=(mask > mask_thershold if mask_thershold else mask).astype(np.uint8))
    extracted_obj = extracted_obj[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    extracted_obj = clean_background(extracted_obj)

    return Image.fromarray(extracted_obj)


def generate_samples(save_dir, amount):
    dataset = AugmentImageDataset(pollen_dir, noise_dir, back_usages=10, objects_per_image=(1, 5), transform=get_transform(False),transform_object=get_transform_object())
    idx = 0
    for img, _ in dataset:
        img = F.to_pil_image(img)
        img.save(f'{save_dir}/sample_{idx}.png')
        idx += 1
        if idx == amount:
            break
    
if __name__ == '__main__':
    path_to_model = 'Save/models/InSegModel_new_wtl'
    input_dir = 'input'
    output_dir = 'output'

    # Generating synthetic images as sample
    # generate_samples(input_dir, 100)

    for name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, name)
        
        score_threshold = 0.7
        mask_threshold = 0.5
        segment_objects(path_to_model, output_dir, image_path, score_threshold, mask_threshold, name)
        