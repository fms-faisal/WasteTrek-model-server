import os
import json
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt  # Only needed if you have other plotting requirements
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F
from collections import defaultdict, deque
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm  # For progress bars
import collections  # For MetricLogger
import time  # For MetricLogger
import math  # For subset creation
import random  # For subset shuffling
from torchvision.ops import nms  # For custom NMS
import tempfile  # For handling temporary files
import pandas as pd  # For saving metrics to Excel

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')


# ------------- Custom Dataset Class -------------

class GarbageDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None, resize=(720, 720)):
        """
        Args:
            images_dir (str): Path to images directory.
            annotation_file (str): Path to the COCO format annotation file.
            transforms (callable, optional): Optional transform to be applied on a sample.
            resize (tuple): Desired image size (height, width).
        """
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.resize = resize

        # Create a mapping from image ID to annotations
        self.img_id_to_ann = defaultdict(list)
        for ann in self.coco.loadAnns(self.coco.getAnnIds()):
            self.img_id_to_ann[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape [3, H, W]
            target: Dict containing bounding boxes and labels
        """
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Original dimensions
        orig_width, orig_height = image.size

        # Resize image
        if self.resize:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS  # For older Pillow versions
            image = image.resize((self.resize[1], self.resize[0]), resample)
            scale_x = self.resize[1] / orig_width
            scale_y = self.resize[0] / orig_height
        else:
            scale_x = scale_y = 1.0

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO format: [x, y, width, height]
            bbox = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = bbox[0] * scale_x
            y_min = bbox[1] * scale_y
            width = bbox[2] * scale_x
            height = bbox[3] * scale_y
            x_max = x_min + width
            y_max = y_min + height

            # Ensure bounding boxes have positive width and height
            if width <= 0 or height <= 0:
                continue  # Skip invalid boxes

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # Assuming only one class (Garbage)
            areas.append(width * height)
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


# ------------- Data Transformations -------------

def get_transform(train):
    transforms_list = []
    # Removed data augmentation
    transforms_list.append(ToTensor())
    transforms_list.append(ResizeTransform((720, 720)))  # Ensure images are resized
    return Compose(transforms_list)


# Define custom transformations using torchvision's functional API
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ResizeTransform(object):
    def __init__(self, size):
        """
        Args:
            size (tuple): Desired output size as (height, width).
        """
        self.size = size

    def __call__(self, image, target):
        # Image is already resized in the dataset class
        return image, target


# ------------- Utility Functions -------------

# Define a utility function for collating batches
class utils:
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# ------------- MetricLogger and SmoothedValue Classes -------------

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def add_meter(self, name, meter):
        """Add a new meter to track additional metrics."""
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, float) or isinstance(v, int):
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'MetricLogger' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                elapsed = time.time() - start_time
                eta = elapsed / i * (len(iterable) - i)
                print(f"{header} [{i}/{len(iterable)}]\t" +
                      f"eta: {eta:.2f}s\t" +
                      f"{str(self)}")
        total_time = time.time() - start_time
        print(f"{header} [{i}/{len(iterable)}]\t" +
              f"Total time: {total_time:.2f}s")


# ------------- Model Definition -------------

def get_model(num_classes):
    # Specify the weights to use
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Adjust NMS parameters in roi_heads
    model.roi_heads.nms_thresh = 0.7  # Adjust NMS threshold as needed
    model.roi_heads.score_thresh = 0.05  # Adjust score threshold as needed
    model.roi_heads.detections_per_img = 100  # Adjust detections per image as needed

    return model


# Number of classes (including background)
num_classes = 2  # 1 class (Garbage) + background

# Get the model
model = get_model(num_classes)
model.to(device)

# ------------- Optimizer and Learning Rate Scheduler -------------

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# ------------- Training and Evaluation Functions -------------

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    # Wrap the data loader with tqdm for progress bar
    with tqdm(total=len(data_loader), desc=header) as pbar:
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            # Convert targets from list of dicts to list of dicts on device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # Reduce losses over all GPUs for logging purposes
            # If using multiple GPUs, this would be necessary
            loss_dict_reduced = loss_dict  # Simplified for single GPU
            losses_reduced = losses  # Simplified for single GPU

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            pbar.set_postfix(loss=metric_logger.meters['loss'].global_avg,
                             lr=metric_logger.meters['lr'].global_avg)
            pbar.update(1)


def custom_nms(outputs, iou_threshold=0.5):
    """
    Applies custom NMS to the model's outputs.

    Args:
        outputs (list of dict): Model outputs for each image.
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        list of dict: Outputs after applying custom NMS.
    """
    processed_outputs = []
    for output in outputs:
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']

        # Apply NMS
        keep = nms(boxes, scores, iou_threshold)

        # Filter the results
        output = {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        }
        processed_outputs.append(output)
    return processed_outputs


def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    coco_gt = get_coco_api_from_dataset(data_loader.dataset)

    coco_results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            outputs = custom_nms(outputs, iou_threshold=iou_threshold)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    if score < 0.05:
                        continue
                    box = box.cpu().numpy()
                    bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': bbox,
                        'score': score.item()
                    })

    if len(coco_results) == 0:
        print("No results to evaluate.")
        return None

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


# Additional utility functions required for evaluation
def get_coco_api_from_dataset(dataset):
    """
    Converts the dataset annotations to COCO format and returns a COCO object.

    Args:
        dataset (Dataset): The dataset to convert.

    Returns:
        COCO: A COCO object with the resized annotations.
    """
    # Create a new COCO object with resized annotations
    coco_json = {
        'images': [],
        'categories': [{'id': 1, 'name': 'Garbage', 'supercategory': 'Garbage'}],
        'annotations': []
    }
    ann_id = 1
    for img_id in dataset.coco.imgs:
        img_info = dataset.coco.loadImgs(img_id)[0]
        coco_json['images'].append({
            'id': img_info['id'],
            'width': 720,
            'height': 720,
            'file_name': img_info['file_name']
        })
        anns = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            # Adjust bbox for resized images
            bbox = ann['bbox']
            orig_width = dataset.coco.imgs[img_id]['width']
            orig_height = dataset.coco.imgs[img_id]['height']
            scale_x = 720 / orig_width
            scale_y = 720 / orig_height
            resized_bbox = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ]
            coco_json['annotations'].append({
                'id': ann_id,
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': resized_bbox,
                'area': ann['area'] * (scale_x * scale_y),
                'iscrowd': ann.get('iscrowd', 0)
            })
            ann_id += 1

    # Write the resized annotations to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(coco_json, tmp_file)
        tmp_file_path = tmp_file.name

    # Initialize COCO with the temporary annotations file
    coco = COCO(tmp_file_path)
    return coco


def _get_iou_types(model):
    """
    Determines the IoU types based on the model type.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        list: A list of IoU types.
    """
    model_type = type(model)
    if model_type == torchvision.models.detection.MaskRCNN:
        return ['bbox', 'segm']
    elif model_type == torchvision.models.detection.KeypointRCNN:
        return ['bbox', 'keypoints']
    return ['bbox']


# ------------- Training Loop -------------

# Paths (Update these paths as per your directory structure)
images_dir = 'C:/Users/lib612/Desktop/litter dataset/images/'
train_ann_file = 'garbage_annotations_train.json'
val_ann_file = 'garbage_annotations_val.json'
test_ann_file = 'garbage_annotations_test.json'

# Create dataset instances
dataset = GarbageDataset(images_dir, train_ann_file, transforms=get_transform(train=True))
dataset_val = GarbageDataset(images_dir, val_ann_file, transforms=get_transform(train=False))
dataset_test = GarbageDataset(images_dir, test_ann_file, transforms=get_transform(train=False))

# Use full training data
subset_train = dataset

print(f"Total training samples: {len(dataset)}")
print(f"Training subset samples: {len(subset_train)}")

# Define data loaders with num_workers set to 0
data_loader = DataLoader(
    subset_train, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn
)

data_loader_val = DataLoader(
    dataset_val, batch_size=4, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn
)

data_loader_test = DataLoader(
    dataset_test, batch_size=4, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn
)

# Create necessary directories
os.makedirs("val_pred", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# Initialize a list to store metrics
metrics_list = []

num_epochs = 10

for epoch in range(num_epochs):
    # Train for one epoch, with tqdm progress bar
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # Update the learning rate
    lr_scheduler.step()
    # Evaluate on the validation dataset
    coco_eval = evaluate(model, data_loader_val, device=device)

    if coco_eval is not None:
        # Extract metrics
        metrics = {
            'epoch': epoch,
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'APs': coco_eval.stats[3],
            'APm': coco_eval.stats[4],
            'APl': coco_eval.stats[5],
            'AR1': coco_eval.stats[6],
            'AR10': coco_eval.stats[7],
            'AR100': coco_eval.stats[8],
            'ARs': coco_eval.stats[9],
            'ARm': coco_eval.stats[10],
            'ARl': coco_eval.stats[11],
        }
        metrics_list.append(metrics)

    # Save model checkpoint
    checkpoint_path = os.path.join("trained_models", f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Create a directory for visualizations of this epoch
    val_pred_epoch_dir = os.path.join("val_pred", f"epoch_{epoch}")
    os.makedirs(val_pred_epoch_dir, exist_ok=True)


    # Visualize predictions and save images
    def visualize_predictions(model, dataset, device, save_dir, iou_threshold=0.5):
        model.eval()
        for i in range(len(dataset)):
            img, target = dataset[i]
            with torch.no_grad():
                prediction = model([img.to(device)])

            # Apply custom NMS
            prediction = custom_nms(prediction, iou_threshold=iou_threshold)

            # Convert the image tensor to PIL Image
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype('uint8'))
            draw = ImageDraw.Draw(img_pil)

            # Plot ground truth boxes in green
            for box in target["boxes"]:
                box = box.cpu().numpy()
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=2)

            # Plot predicted boxes in red
            for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
                if score > 0.5:
                    box = box.cpu().numpy()
                    x_min, y_min, x_max, y_max = box
                    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
                    # Optionally, add text labels
                    # Note: PIL's ImageDraw doesn't support text wrapping and fonts easily.
                    # For simplicity, we skip adding text.

            # Save the image
            image_filename = f"image_{i}_epoch_{epoch}.png"
            save_path = os.path.join(save_dir, image_filename)
            img_pil.save(save_path)


    visualize_predictions(model, dataset_val, device, val_pred_epoch_dir, iou_threshold=0.5)
    print(f"Saved visualizations to {val_pred_epoch_dir}")

# ------------- Saving Metrics to Excel -------------

if metrics_list:
    metrics_df = pd.DataFrame(metrics_list)
    metrics_excel_path = os.path.join("metrics", "training_metrics.xlsx")
    metrics_df.to_excel(metrics_excel_path, index=False)
    print(f"Metrics saved to {metrics_excel_path}")

# ------------- Testing -------------

# After training, evaluate on the test set
print("Evaluating on the test set...")
test_coco_eval = evaluate(model, data_loader_test, device=device)

# ------------- Saving the Final Model -------------

torch.save(model.state_dict(), "garbage_detection_model_final.pth")
print("Final model saved to garbage_detection_model_final.pth")

# ------------- Visualization on Test Set (Optional) -------------

# Create a directory for test visualizations
test_pred_dir = os.path.join("val_pred", "test_predictions")
os.makedirs(test_pred_dir, exist_ok=True)


# Visualize predictions on the test set
def visualize_test_predictions(model, dataset, device, save_dir, iou_threshold=0.5):
    model.eval()
    for i in range(len(dataset)):
        img, target = dataset[i]
        with torch.no_grad():
            prediction = model([img.to(device)])

        # Apply custom NMS
        prediction = custom_nms(prediction, iou_threshold=iou_threshold)

        # Convert the image tensor to PIL Image
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
        draw = ImageDraw.Draw(img_pil)

        # Plot ground truth boxes in green
        for box in target["boxes"]:
            box = box.cpu().numpy()
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=2)

        # Plot predicted boxes in red
        for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
            if score > 0.5:
                box = box.cpu().numpy()
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
                # Optionally, add text labels
                # Note: PIL's ImageDraw doesn't support text wrapping and fonts easily.
                # For simplicity, we skip adding text.

        # Save the image
        image_filename = f"test_image_{i}.png"
        save_path = os.path.join(save_dir, image_filename)
        img_pil.save(save_path)


visualize_test_predictions(model, dataset_test, device, test_pred_dir, iou_threshold=0.5)
print(f"Saved test visualizations to {test_pred_dir}")
