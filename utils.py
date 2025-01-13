import torch
import config
from torchvision.utils import save_image
import cv2
import numpy as np
import wandb
# from sentinelhub import SHConfig
from dotenv import load_dotenv
import os
import numpy as np
import ee
import geemap
import cv2

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


def save_some_examples(gen, val_loader, epoch, folder):
    x,y = next(iter(val_loader))
    x,y = x.to(config.DEVICE), y.to(config.DEVICE) #z1.to(config.DEVICE), z2.to(config.DEVICE), z3.to(config.DEVICE), z4.to(config.DEVICE)
    
    gen.eval()
    with torch.no_grad():
        y_fake =  gen(x,)# z1=z1, z2=z2, z3=z3, z4=z4)
        # y_fake = (y_fake > 0.5).float() 
        # y = (y > 0.5).float()

        # x, z1, z2, z3, z4 = x*0.5+0.5, z1*0.5+0.5, z2*0.5+0.5, z3*0.5+0.5, z4*0.5+0.5 # Denormalise
        x = x*0.5+0.5

        stacked_images = torch.cat((x,y,y_fake), dim=2)
        save_image(stacked_images, folder + f"/y_gen_{epoch}.png")
        save_image(x, folder + f"/input_{epoch}.png")
        wandb.log({
            "Generated Images": [wandb.Image(f"/content/evaluation/y_gen_{epoch}.png", caption=f"Epoch {epoch} - Generated")]
        })
        if epoch == 1 or epoch == 0:
            save_image(y, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def image_to_line_art(image, thickenss=7):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)  # Larger kernel size for smoother lines
    edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)
    
    dilated_edges = cv2.dilate(edges, kernel=np.ones((thickenss, thickenss), np.uint8), iterations=1) # Thicker
    
    smooth_edges = cv2.GaussianBlur(dilated_edges, (5, 5), 0) # Smoother
    line_art = cv2.bitwise_not(smooth_edges)
    line_art_rgb = cv2.cvtColor(line_art, cv2.COLOR_GRAY2RGB)

    return line_art_rgb

def to_grayscale(image_tensor: torch.Tensor):
    """
    Converts an RGB image tensor to grayscale with a single channel.
    Args:
        image_tensor (torch.Tensor): Tensor of shape (C, H, W) or (N, C, H, W)
    Returns:
        torch.Tensor: Grayscale tensor of shape (1, H, W) or (N, 1, H, W)
    """
    if len(image_tensor.shape) == 4:  # Batch of images
        r, g, b = image_tensor[:, 0, :, :], image_tensor[:, 1, :, :], image_tensor[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(1)  # Add channel dim
    
    elif len(image_tensor.shape) == 3:  # Single image
        if not isinstance(image_tensor, torch.Tensor): 
            r, g, b = image_tensor[:, :, 0], image_tensor[:, :, 1], image_tensor[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        
        r, g, b = image_tensor[0, :, :], image_tensor[1, :, :], image_tensor[2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(0)  # Add channel dim
    
    else:
        raise ValueError("Expected input to have 3 or 4 dimensions.")
    

evalscript_true_color = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04"]
        }],
        output: {
            bands: 3
        }
    };
}

function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""


evalscript_ndvi = """
    //VERSION=3

    function setup() {
        return {
            input: [{bands: ["B08", "B04"]}],
            output: {bands: 1, sampleType: "FLOAT32"}
        };
    }

    function evaluatePixel(sample) {
        return [(sample.B08 - sample.B04) / (sample.B08 + sample.B04)];
    }
"""

evalscript_ndwi = """
    //VERSION=3

    function setup() {
        return {
            input: [{bands: ["B03", "B08"]}],
            output: {bands: 1, sampleType: "FLOAT32"}
        };
    }

    function evaluatePixel(sample) {
        return [(sample.B03 - sample.B08) / (sample.B03 + sample.B08)];
    }
"""

def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def generated_lsm_mask(gen, image_seq, inter_images_num=4, target=False, already_grayscale=False):
    inter_image_dict = {}
    image_seq_ndim = len(image_seq.shape)

    # Handling 2D and 3D image sequences
    if image_seq_ndim == 2:  # 2D input
        input_image = image_seq[:config.IMAGE_SIZE, :config.IMAGE_SIZE]
        for idx in range(inter_images_num):
            start_idx = config.IMAGE_SIZE * (idx + 1)
            end_idx = config.IMAGE_SIZE * (idx + 2)
            inter_image_dict[idx] = image_seq[:config.IMAGE_SIZE, start_idx:end_idx, ]
        if target: target_image = image_seq[config.IMAGE_SIZE * (inter_images_num + 1):, :]
    elif image_seq_ndim == 3:  # 3D input
        input_image = image_seq[:, :config.IMAGE_SIZE, :]
        for idx in range(inter_images_num):
            start_idx = config.IMAGE_SIZE * (idx + 1)
            end_idx = config.IMAGE_SIZE * (idx + 2)
            inter_image_dict[idx] = image_seq[:, start_idx:end_idx, :]
        if target: target_image = image_seq[:, config.IMAGE_SIZE * (inter_images_num + 1):, :]
    else:
        raise ValueError("image_seq must be either 2D or 3D.")

    # Transform intermediate images
    for idx, inter_image in inter_image_dict.items():
        inter_image_dict[idx] = config.transform_only_input(image=inter_image)["image"]

    # Transform input and target images
    if target: target_image = config.transform_only_mask_binarize(image=target_image)["image"]
    input_image = config.transform_only_input(image=input_image)["image"]

    # Convert images to grayscale
    if not already_grayscale: 
        input_image = to_grayscale(input_image)
        inter_image_dict = {key: to_grayscale(value) for key, value in inter_image_dict.items()}
    if target: target_image = to_grayscale(target_image)

    # Generate output using the generator model
    generated_output = gen(
        input_image.unsqueeze(0),
        z1=inter_image_dict[0].unsqueeze(0),
        z2=inter_image_dict[1].unsqueeze(0),
        z3=inter_image_dict[2].unsqueeze(0),
        z4=inter_image_dict[3].unsqueeze(0),
    )

    # Binarize the generated output and target image
    generated_binary = (generated_output[0].permute(1, 2, 0).detach().numpy() > 0.5).astype(float)
    if target: target_binary = (target_image.permute(1, 2, 0).detach().numpy() > 0.5).astype(float)

    if target: return generated_binary, target_binary
    return generated_binary


def get_map(cords, start_date, end_date, evalscript_true_color, evalscript_ndvi, evalscript_ndwi, resolution=0.8, correct_shape= False):
    """
    Correct shape - Crops to make a sqaure and brings all maps to a same shape
    """
    ee.Authenticate()
    ee.Initialize()

    config_sentinel = SHConfig(sh_client_id=os.environ.get("sh_client_id"), sh_client_secret=os.environ.get("sh_client_secret"))
    config_sentinel.sh_client_id

    bbox = BBox(bbox=cords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config_sentinel,
    )
    rgb_response = request_true_color.get_data()
    
    request_ndvi = SentinelHubRequest(
        evalscript=evalscript_ndvi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config_sentinel,
    )
    ndvi_response = request_ndvi.get_data()[0]

    request_ndwi = SentinelHubRequest(
        evalscript=evalscript_ndwi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config_sentinel,
    )
    ndwi_response = request_ndwi.get_data()[0]

    # Load the SRTM DEM dataset
    srtm = ee.Image("USGS/SRTMGL1_003")

    # Calculate slope using the `terrain` function
    terrain = ee.Algorithms.Terrain(srtm)

    # Extract elevation and slope
    elevation = srtm.select('elevation')  # Elevation data
    slope = terrain.select('slope') 

    srtm = ee.Image("USGS/SRTMGL1_003")
    terrain = ee.Algorithms.Terrain(srtm)
    elevation = srtm.select('elevation')  # Elevation data
    slope = terrain.select('slope') 

    roi = ee.Geometry.Rectangle(*cords) 

    Map = geemap.Map()
    Map.centerObject(roi, 16)

    # Add elevation and slope layers
    Map.addLayer(elevation.clip(roi), {'min': 0, 'max': 3000, 'palette': ['black', 'white', 'gray']}, 'Elevation')
    Map.addLayer(slope.clip(roi), {'min': 0, 'max': 60, 'palette': ['black', 'white', 'gray']}, 'Slope')

    elevation_arr = geemap.ee_to_numpy(elevation.resample('bicubic'), region=roi)
    slope_arr = geemap.ee_to_numpy(slope, region=roi)

    rgb = (rgb_response[0]*3.5).clip(0,255)
    ndvi = (ndvi_response*3.5).clip(0,255)
    ndwi = (ndwi_response*3.5).clip(0,255)
    elevation = (elevation_arr*3.5).clip(0,255)
    slope = (slope_arr*3.5).clip(0,255)

    rgb = center_crop(rgb)
    ndvi = center_crop(ndvi)
    ndwi = center_crop(ndwi)
    elevation = center_crop(cv2.resize(elevation, (rgb.shape[1], rgb.shape[0])))
    slope = center_crop(cv2.resize(slope, (rgb.shape[1], rgb.shape[0])))    

    return rgb.astype(np.uint8), ndvi, elevation, slope, ndwi

