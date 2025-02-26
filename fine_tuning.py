import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from facb.model import FACB  # Import the FACB model

def get_fine_tune_dataloader(data_dir, batch_size=32):
    """Create a dataloader for fine-tuning using ImageFolder format."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return dataloader, dataset

def fine_tune():
    """Fine-tunes only the classification head while keeping ResNet-50 frozen."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Load fine-tuning dataset
    data_dir = "dataset/labeled/Finetune"
    train_loader, dataset = get_fine_tune_dataloader(data_dir, batch_size=64)

    if len(train_loader) == 0:
        print("ERROR: No data found in dataloader!")
        return

    print(f"Loaded {len(dataset)} images from {data_dir}")

    # ✅ Load pre-trained FACB model (ResNet-50 backbone)
    model = FACB().to(device)
    model.load_state_dict(torch.load("checkpoints/facb_best.pth", map_location=device))

    # ✅ Freeze ResNet-50 backbone (encoder)
    for param in model.encoder.parameters():
        param.requires_grad = False  

    # ✅ Replace projection head with a classifier for 3 classes
    model.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 3)  # 3 output classes: Malignant, Benign, Normal
    ).to(device)

    # ✅ Compute class weights for handling imbalanced data
    all_labels = np.array(dataset.targets)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)  # Only train classifier

    # ✅ Fine-tuning settings
    epochs = 30
    best_acc = 0
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting Fine-Tuning...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels, all_preds = [], []

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (x, labels) in progress_bar:
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            features = model.encoder(x)  # Extract features using frozen ResNet-50
            outputs = model.classifier(features)  # Pass features through classifier
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({"Batch Loss": loss.item()})

        # ✅ Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

        print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # ✅ Save the best fine-tuned model (includes entire architecture: ResNet + classifier)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, "checkpoints/best_fine_tuned.pth")  # Saves the full model
            print("✅ Model saved as best_fine_tuned.pth")

if __name__ == "__main__":
    fine_tune()