import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import Subset

import lpips
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from src.models.GCRB_model import Encoder, Decoder, PatchGAN
from download_data import get_train_loader

trainloader = get_train_loader(data_dir="data/PetImages", batch_size=128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAINING_ITERATIONS = 10000
START_DISC = 4000
DISPLAY_ITERATIONS = 100
DISC_WEIGHT = 0.05
LPIPS_WEIGHT = 0.7

def calculate_compression_ratio(input_tensor, latent_tensor):
    input_size = torch.numel(input_tensor)
    latent_size = torch.numel(latent_tensor)
    return input_size / latent_size

def main():
    e = Encoder().to(DEVICE)
    d = Decoder().to(DEVICE)
    model = nn.Sequential(e, d)

    discriminator = PatchGAN().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    lpips_loss_fn = lpips.LPIPS(net="vgg").eval().to(DEVICE)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    SSIM = []
    PSNR = []

    completed_steps = 0
    pbar = tqdm(range(TRAINING_ITERATIONS))
    train = True

    while train:
        for i, (images, _) in enumerate(trainloader):
            model_toggle = (completed_steps % 2) == 0
            train_disc = (completed_steps >= START_DISC)
            generator_step = model_toggle or not train_disc

            images = images.to(DEVICE)
            latent_space = e(images)
            reconstructions = d(latent_space)

            if generator_step:
                optimizer.zero_grad()

                loss = F.l1_loss(images, reconstructions)
                lpips_loss = lpips_loss_fn(reconstructions, images).mean()
                loss = loss + lpips_loss * LPIPS_WEIGHT

                if train_disc:
                    generator_loss = -1 * discriminator(reconstructions).mean()

                    last_layer = d.out_conv.weight
                    norm_grad_perceptual = torch.autograd.grad(loss, last_layer, retain_graph=True)[0].detach().norm(2)
                    norm_grad_gen = torch.autograd.grad(generator_loss, last_layer, retain_graph=True)[0].detach().norm(2)

                    adaptive_weight = (norm_grad_perceptual / norm_grad_gen.clamp(min=1e-8)).clamp(max=1e4)
                    loss = loss + adaptive_weight * generator_loss * DISC_WEIGHT

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            else:
                disc_optimizer.zero_grad()

                real_pred = discriminator(images)
                fake_pred = discriminator(reconstructions.detach())

                discriminator_loss = (F.relu(1 + fake_pred) + F.relu(1 - real_pred)).mean()

                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                disc_optimizer.step()

            if completed_steps % DISPLAY_ITERATIONS == 0:
                with torch.no_grad():
                    reconstructions_disp = (reconstructions + 1) / 2
                    images_disp = (images + 1) / 2

                    ssim_score = ssim_metric(reconstructions_disp, images_disp).item()
                    psnr_score = psnr_metric(reconstructions_disp, images_disp).item()

                    SSIM.append(ssim_score)
                    PSNR.append(psnr_score)

                    print(f"Step {completed_steps} | PSNR: {psnr_score:.2f} | SSIM: {ssim_score:.4f}")

                    fig, ax = plt.subplots(2, min(images.shape[0], 5), figsize=(15, 5))

                    for idx in range(min(images.shape[0], 5)):
                        orig = images_disp[idx].cpu()
                        recon = reconstructions_disp[idx].cpu()

                        if orig.shape[0] == 1:
                            ax[0, idx].imshow(orig.squeeze(), cmap='gray')
                        else:
                            ax[0, idx].imshow(orig.permute(1, 2, 0))
                        ax[0, idx].axis('off')

                        if recon.shape[0] == 1:
                            ax[1, idx].imshow(recon.squeeze(), cmap='gray')
                        else:
                            ax[1, idx].imshow(recon.permute(1, 2, 0))
                        ax[1, idx].axis('off')

                    fig.text(0.5, 0.95, f"Iteration #{completed_steps}", fontsize=16, ha='center')
                    fig.text(0.5, 0.90, "Original", fontsize=14, ha='center')
                    fig.text(0.5, 0.47, "Reconstructions", fontsize=14, ha='center')
                    plt.show()

            if completed_steps >= TRAINING_ITERATIONS:
                print("Completed Training")
                train = False

                os.makedirs("metrics", exist_ok=True)
                pd.DataFrame({
                    "Step": list(range(0, completed_steps + 1, DISPLAY_ITERATIONS)),
                    "PSNR": PSNR,
                    "SSIM": SSIM
                }).to_csv("metrics/gancaerb_metrics.csv", index=False)
                print("Saved metrics to metrics/gancaerb_metrics.csv")

                os.makedirs("CAE_outputs/samples", exist_ok=True)
                cat_subset = Subset(trainloader.dataset, list(range(128)))
                dog_subset = Subset(trainloader.dataset, list(range(12800, 12800 + 128)))

                cat_images, _ = next(iter(torch.utils.data.DataLoader(cat_subset, batch_size=128)))
                dog_images, _ = next(iter(torch.utils.data.DataLoader(dog_subset, batch_size=128)))
                combined_images = torch.cat([cat_images, dog_images], dim=0).to(DEVICE)

                with torch.no_grad():
                    e.eval()
                    d.eval()

                    latents = e(combined_images)
                    reconstructions = d(latents)

                    originals = (combined_images + 1) / 2
                    reconstructions = (reconstructions + 1) / 2

                    latents_np = latents.view(latents.size(0), -1).cpu().numpy()
                    latents_2d = PCA(n_components=2).fit_transform(latents_np)

                    plt.figure(figsize=(8, 6))
                    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c='blue', edgecolor='k', alpha=0.7)
                    plt.title("PCA of Latent Space (128 Cats + 128 Dogs)")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.grid(True)
                    plt.savefig("CAE_outputs/samples/pca_latent_space.png")
                    plt.close()

                break

            completed_steps += 1
            pbar.update(1)

if __name__ == "__main__":
    main()
