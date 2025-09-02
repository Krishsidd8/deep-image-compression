import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import Subset
from torchvision.utils import save_image

from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from src.models.CAE_model import Encoder, Decoder
from download_data import get_train_loader

trainloader = get_train_loader(data_dir="data/PetImages", batch_size=128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_ITERATIONS = 10000
DISPLAY_ITERATIONS = 100

def calculate_compression_ratio(input_tensor, latent_tensor):
    input_size = torch.numel(input_tensor)
    latent_size = torch.numel(latent_tensor)
    return input_size / latent_size

def main():
    e = Encoder().to(DEVICE)
    d = Decoder().to(DEVICE)
    model = nn.Sequential(e, d)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0001, 
        betas=(0.9, 0.999), 
        weight_decay=0.005
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    SSIM = []
    PSNR = []

    completed_steps = 0
    pbar = tqdm(range(TRAINING_ITERATIONS))

    train = True
    while train:
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(DEVICE)
            latent_space = e(images)
            reconstructions = d(latent_space)

            optimizer.zero_grad()
            loss = F.mse_loss(images, reconstructions)
            loss.backward()
            optimizer.step()

            if completed_steps % DISPLAY_ITERATIONS == 0:
                with torch.no_grad():
                    images_batch = (images + 1) / 2
                    reconstructions_batch = (reconstructions + 1) / 2

                    ssim_score = ssim_metric(reconstructions_batch, images_batch).item()
                    psnr_score = psnr_metric(reconstructions_batch, images_batch).item()

                    SSIM.append(ssim_score)
                    PSNR.append(psnr_score)

                    print(f"Step {completed_steps} | PSNR: {psnr_score:.2f} | SSIM: {ssim_score:.4f}")

                    fig, ax = plt.subplots(2, min(images.shape[0], 5), figsize=(15, 5))

                    for img_idx in range(min(images.shape[0], 5)):
                        orig = images_batch[img_idx].cpu()
                        recon = reconstructions_batch[img_idx].cpu()

                        if orig.shape[0] == 1:
                            ax[0, img_idx].imshow(orig.squeeze(), cmap='gray')
                        else:
                            ax[0, img_idx].imshow(orig.permute(1, 2, 0))
                        ax[0, img_idx].axis('off')

                        if recon.shape[0] == 1:
                            ax[1, img_idx].imshow(recon.squeeze(), cmap='gray')
                        else:
                            ax[1, img_idx].imshow(recon.permute(1, 2, 0))
                        ax[1, img_idx].axis('off')

                    fig.text(0.5, 0.95, f"Iteration #{completed_steps}", fontsize=16, ha='center')
                    fig.text(0.5, 0.90, "Original", fontsize=14, ha='center')
                    fig.text(0.5, 0.47, "Reconstructions", fontsize=14, ha='center')
                    plt.show()

            if completed_steps >= TRAINING_ITERATIONS:
                train = False
                print("Completed Training")

                output_dir = "CAE_outputs/samples"
                os.makedirs(output_dir, exist_ok=True)

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
                    if latents_np.shape[1] > 2:
                        latents_2d = PCA(n_components=2).fit_transform(latents_np)
                    else:
                        latents_2d = latents_np

                    plt.figure(figsize=(8, 6))
                    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c='blue', edgecolor='k', alpha=0.7)
                    plt.title("PCA of Latent Space (128 Cats + 128 Dogs)")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.grid(True)
                    plt.savefig(os.path.join(output_dir, "latent_pca.png"))
                    plt.close()

                break

            completed_steps += 1
            pbar.update(1)

if __name__ == "__main__":
    main()
