from datetime import datetime
from pathlib import Path

import cv2
import mlflow
from PIL import Image
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from tqdm import tqdm
import wandb

from model import ConvDenoiser

NUM_EPOCHS = 100
BATCH_SIZE = 16

LOSS_STEP = 50

wandb.init(
    project="bm3d-denoiser",
    config={
        "learning_rate": 1e-3,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss_step": LOSS_STEP
    }
)


def torch_psnr(images_a, images_b):
    mse = mse_loss(images_a, images_b, reduction='none').mean(dim=(1, 2, 3))
    return 20 * torch.log10(1 / torch.sqrt(mse))


class DenoiserDataset(Dataset):
    def __init__(self, noisy_root: Path, bm3d_root: Path, transform):
        super().__init__()

        self.noisy_root = noisy_root
        self.bm3d_root = bm3d_root
        self.transform = transform
        self.image_paths = list(noisy_root.glob("*.png"))


    def __getitem__(self, index):
        img = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_UNCHANGED)
        img_tensor = torch.from_numpy(img[None, ...])

        bm3d_img = cv2.imread(str(self.bm3d_root / self.image_paths[index].name), cv2.IMREAD_UNCHANGED)
        bm3d_tensor = torch.from_numpy(bm3d_img[None, ...])

        stacked_tensor = torch.cat([img_tensor, bm3d_tensor], dim=0)
        stacked_tensor = self.transform(stacked_tensor)

        img_tensor = stacked_tensor[0:1, ...]
        bm3d_tensor = stacked_tensor[1:2, ...]
        return img_tensor, bm3d_tensor
    
    def __len__(self):
        return len(self.image_paths)


def train_one_epoch(model, dataloader, epoch, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dataloader):
        (noisy_image, bm3d_image) = data
        (noisy_image, bm3d_image) = (noisy_image.to(device, non_blocking=True), bm3d_image.to(device, non_blocking=True))

        optimizer.zero_grad()
        output_image = model(noisy_image)

        loss = loss_fn(output_image, bm3d_image)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % LOSS_STEP == (LOSS_STEP - 1):
            last_loss = running_loss / LOSS_STEP
            print(f'Batch {i} loss = {last_loss}')
            wandb.log({"train/batch/loss": last_loss}, step=(i + epoch * len(dataloader)))

            running_loss = 0.

    wandb.log({"train/epoch/loss": last_loss}, step=epoch)

    return last_loss


def val_one_epoch(model, dataloader, epoch, loss_fn, device):
    model.eval()
    running_loss = 0.
    running_psnr = 0.

    last_loss = 0.
    last_psnr = 0.

    for i, data in tqdm(enumerate(dataloader)):
        (noisy_image, bm3d_image) = data
        (noisy_image, bm3d_image) = (noisy_image.to(device, non_blocking=True), bm3d_image.to(device, non_blocking=True))

        output_image = model(noisy_image)
        loss = loss_fn(output_image, bm3d_image)
        running_loss += loss.item()

        running_psnr += torch_psnr(output_image, bm3d_image).mean()

    last_loss = running_loss / len(dataloader)
    last_psnr = running_psnr / len(dataloader)
    print(f'Validation loss of epoch {epoch} = {last_loss}')
    print(f'Validation PSNR of epoch {epoch} = {last_psnr}')

    wandb.log({"val/epoch/loss": last_loss}, step=epoch)
    wandb.log({"val/epoch/psnr": last_psnr}, step=epoch)

    return last_psnr


def train_loop(num_epochs, model, train_loader, val_loader, loss_fn, optimizer, runs_dir, device):
    best_psnr = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} starting")
        print("Training")
        train_one_epoch(model, train_loader, epoch, loss_fn, optimizer, device)
        print("Validation")
        val_psnr = val_one_epoch(model, val_loader, epoch, loss_fn, device)

        if val_psnr > best_psnr:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, runs_dir / "best.pt")


def main():

    runs_dir = Path("runs") 
    runs_dir.mkdir(exist_ok=True)
    
    runs_dir /= datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_dir.mkdir(exist_ok=False)

    train_transforms = T.Compose([
        T.RandomCrop((128, 128)),
        T.Lambda(lambda x: x.float() / 255.0)
    ])

    val_transforms = T.Compose([
        T.CenterCrop((128, 128)),
        T.Lambda(lambda x: x.float() / 255.0),
    ])

    train_dataset = DenoiserDataset(Path("data/noisy"), Path("data/bm3d"), train_transforms)
    val_dataset = DenoiserDataset(Path("data/noisy"), Path("data/bm3d"), val_transforms)

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    indices = torch.randperm(len(train_dataset))
    
    train_dataset = Subset(train_dataset, indices[:train_size])
    val_dataset = Subset(val_dataset, indices[train_size:])

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)

    loss = nn.MSELoss()

    device = torch.device("mps")
    model = ConvDenoiser().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)

    train_loop(NUM_EPOCHS, model, train_loader, val_loader, loss, optimizer, runs_dir, device)

    wandb.finish()



if __name__ == "__main__":
    main()