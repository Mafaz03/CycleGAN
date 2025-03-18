import wandb
import os
import torch
import torch.amp
from dataset import CycleGANDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator import Discriminator
from Generator import Generator
from config import KAGGLE_STR
from torch.cuda.amp import autocast, GradScaler

def train(epoch, disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):

    loop = tqdm(loader, leave=True)

    for idx, (a, b) in enumerate(loop):
        a = a.to(config.DEVICE)
        b = b.to(config.DEVICE)

        with torch.autocast("cuda"):
        # with autocast():
            # Discriminator A
            fake_A = gen_A(b)
            disc_real_A = disc_A(a)
            disc_fake_A = disc_A(fake_A.detach())
            disc_real_loss_A = mse(disc_real_A, torch.ones_like(disc_real_A))
            disc_fake_loss_A = mse(disc_fake_A, torch.zeros_like(disc_fake_A))
            disc_loss_A = disc_real_loss_A + disc_fake_loss_A

            # Discriminator B
            fake_B = gen_B(a)
            disc_real_B = disc_B(b)
            disc_fake_B = disc_B(fake_B.detach())
            disc_real_loss_B = mse(disc_real_B, torch.ones_like(disc_real_B))
            disc_fake_loss_B = mse(disc_fake_B, torch.zeros_like(disc_fake_B))
            disc_loss_B = disc_real_loss_B + disc_fake_loss_B

            D_loss = (disc_loss_A + disc_loss_B) / 2
        
        # Backprop Discriminator loss
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Generator loss
        with torch.autocast("cuda"):
            # Adversarial loss for both generators
            disc_fake_A = disc_A(fake_A)
            disc_fake_B = disc_B(fake_B)
            loss_gen_A = mse(disc_fake_A, torch.ones_like(disc_fake_A))
            loss_gen_B = mse(disc_fake_B, torch.ones_like(disc_fake_B))

            # Cycle loss
            cycle_A = gen_A(fake_B)
            cycle_B = gen_B(fake_A)
            cycle_loss_A = l1(a, cycle_A)
            cycle_loss_B = l1(b, cycle_B)

            # Identity loss
            if config.LAMBDA_IDENTITY:
                identity_A = gen_A(a)
                identity_B = gen_B(b)
                identity_loss_A = l1(a, identity_A)
                identity_loss_B = l1(b, identity_B)

                G_loss = (
                    loss_gen_A + loss_gen_B 
                    + cycle_loss_A * config.LAMBDA_CYCLE
                    + cycle_loss_B * config.LAMBDA_CYCLE
                    + identity_loss_A * config.LAMBDA_IDENTITY
                    + identity_loss_B * config.LAMBDA_IDENTITY
                )
            else:
                G_loss = (
                    loss_gen_A + loss_gen_B 
                    + cycle_loss_A * config.LAMBDA_CYCLE
                    + cycle_loss_B * config.LAMBDA_CYCLE
                )

        # Backprop Generator loss
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Logging to WandB and saving images
        if idx % 200 == 0:
            save_image(fake_A * 0.5 + 0.5, KAGGLE_STR + f"saved_images/A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, KAGGLE_STR + f"saved_images/B_{idx}.png")
            wandb.log({
                "Generated A Images": [wandb.Image(KAGGLE_STR + f"saved_images/A_{idx}.png", caption=f"Epoch {epoch} - A -> B Generated")],
                "Generated B Images": [wandb.Image(KAGGLE_STR + f"saved_images/B_{idx}.png", caption=f"Epoch {epoch} - B -> A Generated")]
            })

        loop.set_postfix(G_loss=G_loss.item())

        # Log generator and discriminator losses to WandB
        wandb.log({
            "G_loss": G_loss.item(),
            "D_loss": D_loss.item()
        })
def main(wandb_api_key, project):
    wandb.login(key=wandb_api_key)
    wandb.init(project=project)
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)

    gen_A = Generator(img_channels=3, num_features=9) # Generates A
    gen_B = Generator(img_channels=3, num_features=9) # Generates B

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A,
            gen_A,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_gen_B,
            gen_B,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_A,
            disc_A,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_B,
            disc_B,
            opt_disc,
            config.LEARNING_RATE,
        )
    dataset = CycleGANDataset(root_A=config.TRAIN_DIR_A, root_B=config.TRAIN_DIR_B, transform=config.transforms)
    loader = DataLoader(
        dataset=dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    disc_A = disc_A.to(config.DEVICE)
    disc_B = disc_B.to(config.DEVICE)
    gen_A = gen_A.to(config.DEVICE)
    gen_B = gen_B.to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train(
            epoch,
            disc_A,
            disc_B,
            gen_B,
            gen_A,
            loader,
            opt_disc,
            opt_gen,
            L1_loss,
            MSE_loss,
            d_scaler,
            g_scaler,
        )

        if epoch % (config.NUM_EPOCHS // 5) == 0 or epoch == config.NUM_EPOCHS - 1 and config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_DISC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_DISC_B)

if __name__ == "__main__":
    os.makedirs("saved_images", exist_ok=True)
    main("<wandb api key here>", "CycleGAN")