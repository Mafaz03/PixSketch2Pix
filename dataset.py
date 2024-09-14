import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config

class Image_dataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()

        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        # print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_path = self.list_files[index]
        file_path = os.path.join(self.root_dir, image_path)
        image = np.array(Image.open(file_path))
        input_image = image[:, :256, :]
        inter_image = image[:, 256:512, :]
        target_image = image[:, 512:, :]

        input_image = config.transform_only_input(image=input_image)["image"]
        inter_image = config.transform_only_input(image=inter_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, inter_image, target_image