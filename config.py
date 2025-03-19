import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

def is_running_on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_A = "data/horse2zebra/trainA"
TRAIN_DIR_B = "data/horse2zebra/trainB"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True

KAGGLE_STR = "/kaggle/working/CycleGAN/" if is_running_on_kaggle() else ""

CHECKPOINT_GEN_A = KAGGLE_STR + "gena.pth.tar"
CHECKPOINT_GEN_B = KAGGLE_STR + "genb.pth.tar"
CHECKPOINT_DISC_A = KAGGLE_STR + "disca.pth.tar"
CHECKPOINT_DISC_B = KAGGLE_STR + "discb.pth.tar"


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)