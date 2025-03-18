from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config

class CycleGANDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_B = root_B
        self.root_A = root_A
        self.transform = transform

        self.B_images = os.listdir(root_B)
        self.A_images = os.listdir(root_A)
        self.length_dataset = max(len(self.B_images), len(self.A_images)) # 1000, 1500
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        B_img = self.B_images[index % self.B_len]
        A_img = self.A_images[index % self.A_len]

        B_path = os.path.join(self.root_B, B_img)
        A_path = os.path.join(self.root_A, A_img)

        B_img = np.array(Image.open(B_path).convert("RGB"))
        A_img = np.array(Image.open(A_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=B_img, image0=A_img)
            B_img = augmentations["image"]
            A_img = augmentations["image0"]

        return A_img, B_img

## Testing
if __name__ == "__main__":
    ds = CycleGANDataset(root_A=config.TRAIN_DIR_A, root_B=config.TRAIN_DIR_B, transform=config.transforms)
    dl = DataLoader(ds, batch_size=2)
    images = next(iter(dl))
    print(images[0].shape)
    print(images[1].shape)