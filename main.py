import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from facb.model import FACB
from facb.losses import NTXentLoss
from facb.data_utils import get_dataloader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  

    # Load Data
    print("Loading data...")
    train_loader = get_dataloader("dataset/unlabeled", batch_size=64, num_workers=4)

    if len(train_loader) == 0:
        print("ERROR: Dataloader is empty. Check dataset path!")
        return

    print("Data loaded successfully!")

    # Initialize Model
    model = FACB().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    epochs = 100
    loss_history = []
    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting Self-Supervised Training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (x, x_fft, y, y_fft) in progress_bar:
            x, x_fft, y, y_fft = x.to(device), x_fft.to(device), y.to(device), y_fft.to(device)
            optimizer.zero_grad()

            # Check for NaNs before forward pass
            if any(torch.isnan(t).any() for t in [x, x_fft, y, y_fft]):
                print(f"NaN detected in batch {batch_idx}, skipping...")
                continue

            z_x, z_y = model(x, x_fft, y, y_fft)

            # Ensure no NaNs in the output
            if torch.isnan(z_x).any() or torch.isnan(z_y).any():
                print(f"NaN detected in model output, skipping batch {batch_idx}...")
                continue

            loss = criterion(z_x, z_y)

            if torch.isnan(loss):
                print("NaN detected in loss, skipping batch...")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = total_loss / len(train_loader)
        loss_history.append(epoch_loss)

        torch.save(model.state_dict(), f"checkpoints/facb_epoch_{epoch}.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "checkpoints/facb_best.pth")

        print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    # Plot loss after all epochs
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("checkpoints/loss_curve.png")
    plt.close()

    print("Self-Supervised Training Complete!")

if __name__ == "__main__":
    train()