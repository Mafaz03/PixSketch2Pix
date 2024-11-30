import torch
import config
from torchvision.utils import save_image
import cv2
import numpy as np
import wandb

def save_some_examples(gen, val_loader, epoch, folder):
    x,z1,z2,z3,z4,y = next(iter(val_loader))
    x,z1,z2,z3,z4,y = x.to(config.DEVICE), z1.to(config.DEVICE), z2.to(config.DEVICE), z3.to(config.DEVICE), z4.to(config.DEVICE), y.to(config.DEVICE)
    
    gen.eval()
    with torch.no_grad():
        y_fake =  gen(x, z1=z1, z2=z2, z3=z3, z4=z4)
        y_fake = (y_fake > 0.5).float() 
        y = (y > 0.5).float()

        x, z1, z2, z3, z4 = x*0.5+0.5, z1*0.5+0.5, z2*0.5+0.5, z3*0.5+0.5, z4*0.5+0.5 # Denormalise

        stacked_images = torch.cat((x,z1,z2,z3,z4,y,y_fake), dim=2)
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

def to_grayscale(image_tensor):
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
        r, g, b = image_tensor[0, :, :], image_tensor[1, :, :], image_tensor[2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(0)  # Add channel dim
    
    else:
        raise ValueError("Expected input to have 3 or 4 dimensions.")