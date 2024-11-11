import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Image_dataset/maps/train"
VAL_DIR = "Image_dataset/maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
VAL_BATCH_SIZE = 5
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "discv5.pth.tar"
CHECKPOINT_GEN = "genv5.pth.tar"
 
# Stlye loss
ALPHA = 1
BETA = 0.01

both_transform = A.Compose(
    [A.Resize(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Resize(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_inter = A.Compose(
    [
        A.Resize(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Resize(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)