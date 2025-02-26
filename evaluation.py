import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from facb.model import FACB  # Import FACB model

def evaluate_model(model, dataloader, device):
    """Runs evaluation and returns predictions & labels."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Extract features from FACB encoder
            features = model.encoder(images)

            # Pass through classifier
            outputs = model.classifier(features)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def print_metrics(preds, labels, dataset_name):
    """Computes accuracy, precision, recall, and F1-score."""
    accuracy = accuracy_score(labels, preds)
    num_classes = len(set(labels))
    average_type = 'binary' if num_classes == 2 else 'macro'

    precision = precision_score(labels, preds, average=average_type, zero_division=1)
    recall = recall_score(labels, preds, average=average_type, zero_division=1)
    f1 = f1_score(labels, preds, average=average_type, zero_division=1)

    print(f"\nResults for {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 40)

def main():
    """Loads the model, evaluates on multiple datasets, and prints results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Define test transformations
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Ensure compatibility with ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ✅ Define test datasets
    test_datasets = {
        "BUSBRA": datasets.ImageFolder('dataset/labeled/Test/BUSBRA', transform=test_transforms),
        "BUSI": datasets.ImageFolder('dataset/labeled/Test/BUSI', transform=test_transforms),
        "LIVERLES": datasets.ImageFolder('dataset/labeled/Test/LIVERLISS', transform=test_transforms)
    }

    # ✅ Load fine-tuned FACB model
    model = FACB().to(device)
    checkpoint_path = "checkpoints/best_fine_tuned.pth"

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No fine-tuned model found at {checkpoint_path}")
        return

    model = torch.load(checkpoint_path, map_location=device)
    model.to(device)
    model.classifier = nn.Identity()  # Remove any previous classifier
    model.eval()

    # ✅ Evaluate model on each dataset
    for dataset_name, dataset in test_datasets.items():
        num_classes = len(dataset.classes)

        # ✅ Define classifier dynamically for the dataset
        model.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        ).to(device)

        test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        preds, labels = evaluate_model(model, test_loader, device)
        print_metrics(preds, labels, dataset_name)

if __name__ == "__main__":
    main()