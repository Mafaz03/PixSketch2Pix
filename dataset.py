import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config
from utils import to_grayscale


class Image_dataset(Dataset):
    def __init__(self, root_dir, inter_images=1, binarize_output=False, grayscale_all=False):
    
        super().__init__()

        self.binarize = binarize_output
        self.grayscale_all = grayscale_all
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        self.inter_images = inter_images 
        # print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_path = self.list_files[index]
        file_path = os.path.join(self.root_dir, image_path)
        image = np.array(Image.open(file_path))
        input_image = image[:, :config.IMAGE_RESIZED, :]

        inter_image_dict = {}
        for idx in range(self.inter_images):
            start_idx = config.IMAGE_RESIZED * (idx + 1)
            end_idx = config.IMAGE_RESIZED * (idx + 2)
            inter_image_dict[idx] = image[:, start_idx:end_idx, :]

        target_image = image[:, config.IMAGE_RESIZED * (self.inter_images + 1):, :]
        input_image = config.transform_only_input(image=input_image)["image"]

        for idx, inter_image in enumerate(inter_image_dict.values()):
            inter_image_dict[idx] = config.transform_only_input(image=inter_image)["image"]

        if self.binarize:  
            target_image = config.transform_only_mask_binarize(image=target_image)["image"]
        else:
            target_image = config.transform_only_mask(image=target_image)["image"]
        
        if self.grayscale_all: 
            input_image = to_grayscale(input_image)
            inter_image_dict = {key: to_grayscale(value) for key, value in inter_image_dict.items()}
            target_image = to_grayscale(target_image)
        
        return input_image, *list(inter_image_dict.values()), target_image
    
## Testing
if __name__ == "__main__":
    ds = Image_dataset(root_dir="image_dataset/landslide/Train", inter_images=0, binarize_output = True, grayscale_all=True)
    dl = DataLoader(ds)

    x, z1, z2, z3, z4, y = next(iter(dl))
    print("Input image: ", x.shape)
    print("Inter1 image: ", z1.shape)
    print("Inter2 image: ", z2.shape)
    print("Inter3 image: ", z3.shape)
    print("Inter4 image: ", z4.shape)
    print("Output/Target image: ", y.shape)