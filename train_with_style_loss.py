import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Image_dataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import wandb
from VGG import VGG
vgg_model = VGG().to(config.DEVICE).eval()

torch.backends.cudnn.benchmark = True

def calc_style_loss(y_fake, y):
    s_loss = 0
    generated_features = vgg_model(y_fake * 0.5 + 0.5) # Remove unsqueeze when actuall training
    style_image_features = vgg_model(y)

    for generated_feature, style_image_feature in zip(generated_features, style_image_features):
        batch_size, channel, height, width = generated_feature.shape
        
        generated_feature = generated_feature.view(batch_size, channel, height * width)
        style_image_feature = style_image_feature.view(batch_size, channel, height * width)

        generated_feature = torch.nn.functional.normalize(generated_feature, p=2, dim=-1)
        style_image_feature = torch.nn.functional.normalize(style_image_feature, p=2, dim=-1)

        # Compute Gram matrices for each image in the batch
        generated_gram_matrix = torch.bmm(generated_feature, generated_feature.transpose(1, 2))
        style_gram_matrix = torch.bmm(style_image_feature, style_image_feature.transpose(1, 2))
      
        # Compute style loss for the whole batch
        # generated_gram_matrix = generated_gram_matrix.to('cpu')
        # style_gram_matrix = style_gram_matrix.to('cpu')

        diff = generated_gram_matrix-style_gram_matrix
        
        s_loss += torch.mean(((diff)**2).float()).half() # HALF PRECISION WAS CAUSING NAN EROR (2 days to fix)

    # import pdb; pdb.set_trace()
    return s_loss



def train_fn(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop = tqdm(train_loader, leave=True)

    for idx, (x,z,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        z = z.to(config.DEVICE)
        y = y.to(config.DEVICE)


        # Train Discriminator
        with torch.amp.autocast("cuda"):
            y_fake = gen(x, z)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_real))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.amp.autocast("cuda"):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
            style_loss_G = calc_style_loss(y_fake, y)
            total_loss = config.ALPHA * G_loss + config.BETA * style_loss_G

        gen.zero_grad()
        g_scaler.scale(total_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_loss=G_loss.item(),
                style_loss_G=style_loss_G.item(),
                total_loss=total_loss.item()
            )

def main():
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator(in_channels=3, features=64).to(config.DEVICE)

    opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, generator, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, discriminator, opt_disc, config.LEARNING_RATE,
        )
    
    train_dataset = Image_dataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")
    
    val_dataset = Image_dataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False)
    wandb.init()
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: ", epoch)
        train_fn(
            discriminator, generator, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(generator, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(discriminator, opt_disc, filename=config.CHECKPOINT_DISC)
        save_some_examples(generator, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()

