import warnings
warnings.simplefilter("ignore")

import os
import gc
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
#adapted from https://opacus.ai/tutorials/building_image_classifier
MAX_GRAD_NORM = 1.0
EPSILON_VALUES = [4.0, 7.0, 10.0]
DELTA = 1e-5
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 128
MAX_PHYSICAL_BATCH_SIZE = 128

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

MODELS = {
    "resnet34": lambda num_classes: torchvision.models.resnet34(num_classes=num_classes),
    "densenet121": lambda num_classes: torchvision.models.densenet121(num_classes=num_classes),
    "efficientnet_v2_s": lambda num_classes: torchvision.models.efficientnet_v2_s(num_classes=num_classes),
}

DATASETS = {
    "cifar10": {
        "dataset": lambda root, transform: torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        ),
        "test_dataset": lambda root, transform: torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        ),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "classes": 10,
    },
    
}

def accuracy(preds, labels):
    return (preds == labels).mean()

def train(model, train_loader, optimizer, epoch, privacy_engine, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

    avg_loss = np.mean(losses)
    avg_acc = np.mean(top1_acc) * 100
    return avg_loss, avg_acc

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(top1_acc) * 100
    print(f"Test Set: Loss: {avg_loss:.6f} Acc: {avg_acc:.6f}")
    return avg_acc

def main():
    for dataset_name, dataset_info in DATASETS.items():
        for model_name, model_func in MODELS.items():
            for epsilon in EPSILON_VALUES:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_info["mean"], dataset_info["std"]),
                ])
                train_dataset = dataset_info["dataset"](root=f"../{dataset_name}", transform=transform)
                test_dataset = dataset_info["test_dataset"](root=f"../{dataset_name}", transform=transform)

                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                model = model_func(num_classes=dataset_info["classes"])
                model = ModuleValidator.fix(model)
                model = model.to(device)

                optimizer = optim.SGD(model.parameters(), lr=LR)
                privacy_engine = PrivacyEngine()
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    epochs=EPOCHS,
                    target_epsilon=epsilon,
                    target_delta=DELTA,
                    max_grad_norm=MAX_GRAD_NORM,
                )

                best_acc = 0
                for epoch in range(EPOCHS):
                    train(model, train_loader, optimizer, epoch + 1, privacy_engine, device)
                test_acc = test(model, test_loader, device)

                model_filename = f"{model_name}-{dataset_name}-eps{epsilon}-{test_acc:.2f}.pth"
                torch.save(model, model_filename)

                del model, optimizer, train_loader, test_loader
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

