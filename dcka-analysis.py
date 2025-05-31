import PyRKHSstats
from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.hsic import perform_gamma_approximation_hsic_independence_testing, perform_permutation_hsic_independence_testing
import os
import pandas as pd
import torch
import csv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from captum.attr import LayerGradientXActivation, LayerActivation
from torchvision import models, datasets
from opacus.validators import ModuleValidator
import numpy as np
from functools import partial
from simtorch.model.sim_model import SimilarityModel
from simtorch.similarity import CKA, DeltaCKA
device = torch.device("cpu")


SEED = 42
#taken from https://opacus.ai/tutorials/building_image_classifier
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)


torch.manual_seed(SEED)
np.random.seed(SEED)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

test_dataset = datasets.CIFAR10(root="../cifar10", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)

base_model_paths = ["..."]
private_model_paths = ["...", "..." ,"..."]

base_model = models.resnet34(pretrained=True)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, 10)
base_model = ModuleValidator.fix(base_model)
full_model = torch.load(base_model_paths[0], map_location=device)
state_dict = full_model.state_dict()
new_state_dict = {key.replace("_module.", ""): value for key, value in state_dict.items()}
base_model.load_state_dict(new_state_dict)
base_model.to(device)
base_model.eval()


for private_model_path in private_model_paths:
    private_model = models.resnet34(pretrained=True)
    num_ftrs = private_model.fc.in_features
    private_model.fc = nn.Linear(num_ftrs, 10)
    private_model = ModuleValidator.fix(private_model)
    full_model = torch.load(private_model_path, map_location=device)
    state_dict = full_model.state_dict()
    new_state_dict = {key.replace("_module.", ""): value for key, value in state_dict.items()}
    private_model.load_state_dict(new_state_dict)
    private_model.to(device)
    private_model.eval()

    private_model_name = os.path.splitext(os.path.basename(private_model_path))[0]
    """
    For the uploaded base model, we compare the representational similarity at all corresponding activation layers 
    between the base model and each private model. In DenseNet and ResNet architectures, these layers use ReLU activations, 
    while in EfficientNet, they primarily use SiLU.
    """
    sim_model1 = SimilarityModel(
        base_model,
        model_name="non-pvt",
        layers_to_include=["relu"] 
    )

    sim_model2 = SimilarityModel(
        private_model,
        model_name=private_model_name,
        layers_to_include=["relu"]
    )

    # Compute DeltaCKA
    sim_dcka = DeltaCKA(sim_model1, sim_model2, device=device)
    dcka_matrix = sim_dcka.compute(test_loader)

    dcka_output_filename = os.path.join(output_folder, f'dcka_matrix_{private_model_name}.npy')
    np.save(dcka_output_filename, dcka_matrix) # Mainly the principal diagonal is required for our analysis between corresponding layers.


    del private_model
    torch.cuda.empty_cache()

del base_model
torch.cuda.empty_cache()
