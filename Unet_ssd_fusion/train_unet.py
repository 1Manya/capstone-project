from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from config import IMAGE_ROOT, MASK_ROOT, UNET_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED
from common import load_labels, get_split, resolve_image_path, relative_image_path, seed_everything
from unet_model import UNet

CHECKPOINT = UNET_DIR / "checkpoint_last.pth"
SAVE_EVERY_STEPS = 100
EPOCHS = 30

class MaskDataset(Dataset):
    def __init__(self, frame):
        self.df = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_image_path(row["path"], row["variation"])
        rel = relative_image_path(row["path"], row["variation"])
        mask_path = MASK_ROOT / rel.with_suffix(".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read image/mask: {img_path} / {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask = torch.from_numpy((mask > 127).astype(np.float32)[None, ...])
        return image, mask

def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()

def save_checkpoint(model, optimizer, scaler, epoch, next_batch, global_step):
    UNET_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, CHECKPOINT)

def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_labels()
    train_df, val_df = get_split(df, "train"), get_split(df, "val")
    train_ds, val_ds = MaskDataset(train_df), MaskDataset(val_df)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    start_epoch, resume_batch, global_step = 0, 0, 0

    if CHECKPOINT.exists():
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = int(ckpt["epoch"])
        resume_batch = int(ckpt.get("next_batch", 0))
        global_step = int(ckpt.get("global_step", 0))
        if ckpt.get("torch_rng") is not None:
            torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print(f"Resuming at epoch {start_epoch+1}, batch {resume_batch}, step {global_step}")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    for epoch in range(start_epoch, EPOCHS):
        # A deterministic order per epoch allows a mid-epoch restart to skip completed batches.
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )
        model.train()
        running = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"U-Net epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (images, masks) in pbar:
            if epoch == start_epoch and batch_idx < resume_batch:
                continue
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if global_step % SAVE_EVERY_STEPS == 0:
                save_checkpoint(model, optimizer, scaler, epoch, batch_idx + 1, global_step)

        # Validation and epoch-complete checkpoint
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += (nn.functional.binary_cross_entropy_with_logits(logits, masks) +
                             dice_loss(logits, masks)).item()
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}: train_loss={running/max(1, len(train_loader)-resume_batch):.4f}, val_loss={val_loss:.4f}")
        save_checkpoint(model, optimizer, scaler, epoch + 1, 0, global_step)
        resume_batch = 0

    torch.save(model.state_dict(), UNET_DIR / "unet_final.pth")
    print(f"Saved: {UNET_DIR / 'unet_final.pth'}")

if __name__ == "__main__":
    main()
