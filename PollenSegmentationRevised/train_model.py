import os
import cv2
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from copy import deepcopy
from engine import train_one_epoch, evaluate
from torchvision.io import read_image
from random import randint, randrange
import utils
from tqdm import tqdm
from torchvision.transforms import v2 as T
from torchvision import datasets, tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import functional as F
from datumaro.components.dataset import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def clean_background(src):
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst


def cvatParse(path_to_xml: str) -> dict:
    dataset = Dataset.import_from(path=path_to_xml, format='cvat')

    namespace = {}
    for item in dataset:
        content = []
        for ann in item.annotations:
            obj = {}

            if ann.type.name == 'polygon':
                mask = Image.new('L', item.media.size[::-1], 0)
                draw = ImageDraw.Draw(mask)

                polygon_points = ann.points
                draw.polygon(polygon_points, outline=1, fill=1)

                mask_array = np.array(mask)
                obj.update({'mask': mask_array})
                obj.update({'specie': ann.attributes.get('Specie', 'Unknown')})  

            content.append(obj)
        
        image_name = item.id.split('/')[1]
        namespace.update({image_name: content})
    
    return namespace


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images/Test/Sample_1'))))
        self.namespace = cvatParse(path_to_xml=os.path.join(root, 'annotations.xml'))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images/Test/Sample_1', self.imgs[idx])
        img = read_image(img_path).float() / 255.0
        
        img_name = self.imgs[idx].split('.')[0]
        masks = [item['mask'] for item in self.namespace[img_name]]
        
        if len(masks) == 0:
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            num_objs = 0
        else:
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            num_objs = len(masks)
            boxes = masks_to_boxes(masks)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class AugmentImageDataset(datasets.DatasetFolder):
    def __init__(self, object_dir: str, back_dir: str, back_usages=10, objects_per_image=(1, 2),
                 transform=None, transform_object=None):
        self.img_dir = object_dir
        self.back_dir = back_dir
        self.objects_per_image = objects_per_image
        self.back_usages = back_usages
        self.transform = transform
        self.transform_object = transform_object
        self.objects = []
        self.backgrounds = []

        for folder_name in os.listdir(object_dir):
            folder_path = os.path.join(object_dir, folder_name)
            for image_name in os.listdir(folder_path):
                if image_name.endswith('.png'):
                    image = Image.open(os.path.join(folder_path, image_name)).convert('RGB')
                    self.objects.append(image)

        for image_name in os.listdir(back_dir):
            image = F.to_pil_image(read_image(os.path.join(back_dir, image_name)))
            self.backgrounds.append(image)

    def __len__(self):
        return len(self.backgrounds) * self.back_usages

    def __getitem__(self, idx):
        idx //= self.back_usages
        back = deepcopy(self.backgrounds[idx])
        objects_per_image = randint(*self.objects_per_image)

        target = {}
        masks = []
        itr = 0
        fitted = 0
        while fitted < objects_per_image and itr < objects_per_image * 3:
            itr += 1
            object = deepcopy(self.objects[randrange(len(self.objects))])
            object_image = F.to_pil_image(clean_background(np.array(object)))

            if self.transform_object is not None:
                object_image = self.transform_object(object_image)

            pos_x = randrange(0, back.size[0] - object_image.size[0])
            pos_y = randrange(0, back.size[1] - object_image.size[1])
            
            mask_background = F.to_pil_image(np.zeros((back.size[1], back.size[0], 1)))
            mask_background.paste(object_image, (pos_x, pos_y), object_image)
            mask = np.array(mask_background)
            mask[mask != 0] = 1
            
            obj_size = cv2.countNonZero(mask)
            for fitted_mask in masks:
                fitted_obj_size = cv2.countNonZero(fitted_mask)
                if cv2.countNonZero(cv2.bitwise_and(fitted_mask, mask)) > 0.05 * min(obj_size, fitted_obj_size):
                    break
            else:
                masks += [mask] 
                fitted += 1
                back.paste(object_image, (pos_x, pos_y), object_image)

        img = tv_tensors.Image(back)
        masks = torch.from_numpy(np.array(masks)).to(dtype=torch.uint8)
        labels = torch.ones((fitted,), dtype=torch.int64)
        ispollen = torch.zeros((fitted,), dtype=torch.int64)
        boxes = masks_to_boxes(masks)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target['masks'] = tv_tensors.Mask(masks)
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(back))
        target['labels'] = labels
        target['iscrowd'] = ispollen # easier to use iscrowd
        target['area'] = area
        target['image_id'] = idx
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target
        

def get_model_instance_segmentation(num_classes, pre_trained: bool):
    if pre_trained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1')
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2()

    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 128
    
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomPhotometricDistort(p=0.5))
        transforms.append(T.RandomAutocontrast(p=0.5))
        transforms.append(T.RandomAdjustSharpness(p=0.5, sharpness_factor=2))
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_transform_object():
    transforms = []
    transforms.append(T.RandomRotation(degrees=180))
    return T.Compose(transforms)


def get_image(model, device, image, epoch, save_dir):
    eval_transform = get_transform(train=False)
    
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
    model.train()
    
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    orig = deepcopy(image)
    
    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="yellow")

    pred_labels = [f"{score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(output_image, pred_boxes, pred_labels, 
                                       colors="red", font="/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf", font_size=32)

    plt.figure(figsize=(12, 12))
    plt.imshow(orig.permute(1, 2, 0))
    plt.savefig(f'{save_dir}/orig[{epoch}].png')
    
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(f'{save_dir}/img[{epoch}].png')


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True


def train_model(model, model_save_dir:str, image_save_dir:str, data_loader: torch.utils.data.DataLoader, data_loader_val: torch.utils.data.DataLoader,
                review_dataset:datasets.DatasetFolder, device, num_epochs:int, freezed_epochs: int, log_dir:str, log_freq: int):
    writer = SummaryWriter(log_dir)

    if freezed_epochs:
        freeze_backbone(model)

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([
        {
            'params': head_params,
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'nesterov': True,
        },
        {
            'params': model.backbone.parameters(),
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'nesterov': True,
        }
    ])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15], 
        gamma=0.5,
    )
    
    best_mAP = 0
    for epoch in tqdm(range(1, num_epochs+1)):
        if freezed_epochs and (epoch - 1 == freezed_epochs):
            unfreeze_backbone(model)

        train_one_epoch(model, optimizer, data_loader, writer, device, epoch, print_freq=log_freq)

        lr_scheduler.step()
        
        writer.flush()

        image = review_dataset[randrange(len(review_dataset))][0]
        get_image(model, device, image, epoch, image_save_dir)


        if epoch % 2 == 0:
            evaluator = evaluate(model, data_loader_val, device)
            segm_metric = evaluator.coco_eval['segm'].stats[0]
            bbox_metric = evaluator.coco_eval['bbox'].stats[0]

            log_val = {}
            log_val.update({'segm': float(segm_metric)})
            log_val.update({'bbox': float(bbox_metric)})

            #writer.add_scalars(f'{name} sample', log_train, (epoch-1)*len(iterable) + i)
            writer.add_scalars('Validation Sample', log_val, epoch)

            current_mAP = segm_metric 
            
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                torch.save(model.state_dict(), model_save_dir)

    writer.close()


if __name__ == '__main__':
    # train/save dirs
    noise_dir = 'Data/Noise_40X_95'
    pollen_dir = 'Data/pollen_dataset_2024_08_14_objects_clean'
    PATH_TO_MODEL = 'Save/models/InSegModel_last_wtl'
    log_dir = 'Save/metrics/log_last_wtl'
    image_save_dir = 'Save/images/images_last_wtl'
    test_root = r'Data/CVAT_dataset'

    #pollen_classes = [folder for folder in os.listdir(pollen_dir) if folder[0] != '.']
    num_classes = 2  # pollen as one class + background (noise)
    freezed_epochs = 10 # number of epochs when only roi head will get updates

    # synthetic dataset definition
    dataset = AugmentImageDataset(pollen_dir, noise_dir, back_usages=3, objects_per_image=(1, 6), transform=get_transform(True),
                        transform_object=get_transform_object())

    review_dataset = AugmentImageDataset(pollen_dir, noise_dir, back_usages=1, objects_per_image=(1, 6), transform=get_transform(False),transform_object=get_transform_object())

    # define training data loader
    data_loader_train = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn,
    )

    real_dataset = SampleDataset(root=test_root)

    seed = 28
    generator = torch.Generator().manual_seed(seed)

    val_dataset, test_dataset = torch.utils.data.random_split(
        real_dataset,
        [124, len(real_dataset)-124],
        generator=generator 
    )

    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_instance_segmentation(num_classes, pre_trained=True)
    model.to(device)

    num_epochs = 30
    log_freq = 50

    train_model(model, PATH_TO_MODEL, image_save_dir, data_loader_train, data_loader_val, review_dataset, 
                device, num_epochs, freezed_epochs, log_dir,  log_freq)
