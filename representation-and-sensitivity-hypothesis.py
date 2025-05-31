import PyRKHSstats
from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.hsic import perform_gamma_approximation_hsic_independence_testing, perform_permutation_hsic_independence_testing
from sklearn.metrics.pairwise import linear_kernel
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

output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)

base_model_paths = ["..."] #Saved non-private model with ".pth" extension for our case.
private_model_paths = ["...", "..." ,"..."] #Set of saved private models, parameterized with epsilon & delta with ".pth" extension for our case.
csv_filename = os.path.join(output_folder, 'grad-act-hypothesis.csv')
csv_header = ['Model', 'Layer', 'act_Independence', 'grad_Independence']
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    for base_model_path in base_model_paths:
        """
        Opacus-trained models require some preprocessing before loading their weights into a standard model. First, we use `ModuleValidator.fix(...)` to replace unsupported layers (e.g., BatchNorm → GroupNorm). Since both the private and public models share the same architecture and normalization strategy (GroupNorm), this ensures structural compatibility.
        Additionally, Opacus sometimes attaches internal attributes (e.g., `_module`) to the state_dict,
        which must be stripped out before loading the weights.
        """
        base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = base_model.classifier.in_features
        base_model.classifier = nn.Linear(num_ftrs, 10)
        base_model = ModuleValidator.fix(base_model)
        full_model = torch.load(base_model_path, map_location=device)
        state_dict = full_model.state_dict()
        #new_state_dict = {key.replace("_module.", ""): value for key, value in state_dict.items()}
        base_model.load_state_dict(new_state_dict)
        base_model.to(device)
        base_model.eval()

        with torch.no_grad():
            base_outputs = base_model(images.to(device))
            base_labels = base_outputs.argmax(dim=1)

        for private_model_path in private_model_paths:
            private_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            num_ftrs = private_model.classifier.in_features
            private_model.classifier = nn.Linear(num_ftrs, 10)
            private_model = ModuleValidator.fix(private_model)
            full_model = torch.load(private_model_path, map_location = device)
            state_dict = full_model.state_dict()
            new_state_dict = {key.replace("_module.", ""): value for key, value in state_dict.items()} #Unlike Base Model, this is an essential step for private models.
            private_model.load_state_dict(new_state_dict)
            private_model.to(device)
            private_model.eval()

            with torch.no_grad():
                private_outputs = private_model(images.to(device))
                private_labels = private_outputs.argmax(dim=1)

            common_mask = base_labels == private_labels
            common_images = images[common_mask]
            common_labels = base_labels[common_mask]

    #activation layers for models. For resnet and efficinetNet, please access the activation layer ids from here: https://bit.ly/dp_x_xai
            layer_paths = [
base_model.features.relu0,
base_model.features[4].denselayer1.relu1 ,
base_model.features[4].denselayer1.relu2 ,
base_model.features[4].denselayer2.relu1 ,
base_model.features[4].denselayer2.relu2 ,
base_model.features[4].denselayer3.relu1 ,
base_model.features[4].denselayer3.relu2 ,
base_model.features[4].denselayer4.relu1 ,
base_model.features[4].denselayer4.relu2 ,
base_model.features[4].denselayer5.relu1 ,
base_model.features[4].denselayer5.relu2 ,
base_model.features[4].denselayer6.relu1 ,
base_model.features[4].denselayer6.relu2 ,
base_model.features[5].relu ,
base_model.features[6].denselayer1.relu1 ,
base_model.features[6].denselayer1.relu2 ,
base_model.features[6].denselayer2.relu1 ,
base_model.features[6].denselayer2.relu2 ,
base_model.features[6].denselayer3.relu1 ,
base_model.features[6].denselayer3.relu2 ,
base_model.features[6].denselayer4.relu1 ,
base_model.features[6].denselayer4.relu2 ,
base_model.features[6].denselayer5.relu1 ,
base_model.features[6].denselayer5.relu2 ,
base_model.features[6].denselayer6.relu1 ,
base_model.features[6].denselayer6.relu2 ,
base_model.features[6].denselayer7.relu1 ,
base_model.features[6].denselayer7.relu2 ,
base_model.features[6].denselayer8.relu1 ,
base_model.features[6].denselayer8.relu2 ,
base_model.features[6].denselayer9.relu1 ,
base_model.features[6].denselayer9.relu2 ,
base_model.features[6].denselayer10.relu1 ,
base_model.features[6].denselayer10.relu2 ,
base_model.features[6].denselayer11.relu1 ,
base_model.features[6].denselayer11.relu2 ,
base_model.features[6].denselayer12.relu1 ,
base_model.features[6].denselayer12.relu2 ,
base_model.features[7].relu ,
base_model.features[8].denselayer1.relu1 ,
base_model.features[8].denselayer1.relu2 ,
base_model.features[8].denselayer2.relu1 ,
base_model.features[8].denselayer2.relu2 ,
base_model.features[8].denselayer3.relu1 ,
base_model.features[8].denselayer3.relu2 ,
base_model.features[8].denselayer4.relu1 ,
base_model.features[8].denselayer4.relu2 ,
base_model.features[8].denselayer5.relu1 ,
base_model.features[8].denselayer5.relu2 ,
base_model.features[8].denselayer6.relu1 ,
base_model.features[8].denselayer6.relu2 ,
base_model.features[8].denselayer7.relu1 ,
base_model.features[8].denselayer7.relu2 ,
base_model.features[8].denselayer8.relu1 ,
base_model.features[8].denselayer8.relu2 ,
base_model.features[8].denselayer9.relu1 ,
base_model.features[8].denselayer9.relu2 ,
base_model.features[8].denselayer10.relu1 ,
base_model.features[8].denselayer10.relu2 ,
base_model.features[8].denselayer11.relu1 ,
base_model.features[8].denselayer11.relu2 ,
base_model.features[8].denselayer12.relu1 ,
base_model.features[8].denselayer12.relu2 ,
base_model.features[8].denselayer13.relu1 ,
base_model.features[8].denselayer13.relu2 ,
base_model.features[8].denselayer14.relu1 ,
base_model.features[8].denselayer14.relu2 ,
base_model.features[8].denselayer15.relu1 ,
base_model.features[8].denselayer15.relu2 ,
base_model.features[8].denselayer16.relu1 ,
base_model.features[8].denselayer16.relu2 ,
base_model.features[8].denselayer17.relu1 ,
base_model.features[8].denselayer17.relu2 ,
base_model.features[8].denselayer18.relu1 ,
base_model.features[8].denselayer18.relu2 ,
base_model.features[8].denselayer19.relu1 ,
base_model.features[8].denselayer19.relu2 ,
base_model.features[8].denselayer20.relu1 ,
base_model.features[8].denselayer20.relu2 ,
base_model.features[8].denselayer21.relu1 ,
base_model.features[8].denselayer21.relu2 ,
base_model.features[8].denselayer22.relu1 ,
base_model.features[8].denselayer22.relu2 ,
base_model.features[8].denselayer23.relu1 ,
base_model.features[8].denselayer23.relu2 ,
base_model.features[8].denselayer24.relu1 ,
base_model.features[8].denselayer24.relu2 ,
base_model.features[9].relu ,
base_model.features[10].denselayer1.relu1 ,
base_model.features[10].denselayer1.relu2 ,
base_model.features[10].denselayer2.relu1 ,
base_model.features[10].denselayer2.relu2 ,
base_model.features[10].denselayer3.relu1 ,
base_model.features[10].denselayer3.relu2 ,
base_model.features[10].denselayer4.relu1 ,
base_model.features[10].denselayer4.relu2 ,
base_model.features[10].denselayer5.relu1 ,
base_model.features[10].denselayer5.relu2 ,
base_model.features[10].denselayer6.relu1 ,
base_model.features[10].denselayer6.relu2 ,
base_model.features[10].denselayer7.relu1 ,
base_model.features[10].denselayer7.relu2 ,
base_model.features[10].denselayer8.relu1 ,
base_model.features[10].denselayer8.relu2 ,
base_model.features[10].denselayer9.relu1 ,
base_model.features[10].denselayer9.relu2 ,
base_model.features[10].denselayer10.relu1 ,
base_model.features[10].denselayer10.relu2 ,
base_model.features[10].denselayer11.relu1 ,
base_model.features[10].denselayer11.relu2 ,
base_model.features[10].denselayer12.relu1 ,
base_model.features[10].denselayer12.relu2 ,
base_model.features[10].denselayer13.relu1 ,
base_model.features[10].denselayer13.relu2 ,
base_model.features[10].denselayer14.relu1 ,
base_model.features[10].denselayer14.relu2 ,
base_model.features[10].denselayer15.relu1 ,
base_model.features[10].denselayer15.relu2 ,
base_model.features[10].denselayer16.relu1 ,
base_model.features[10].denselayer16.relu2
            ]
            layer_paths_ = [
private_model.features.relu0,
private_model.features[4].denselayer1.relu1 ,
private_model.features[4].denselayer1.relu2 ,
private_model.features[4].denselayer2.relu1 ,
private_model.features[4].denselayer2.relu2 ,
private_model.features[4].denselayer3.relu1 ,
private_model.features[4].denselayer3.relu2 ,
private_model.features[4].denselayer4.relu1 ,
private_model.features[4].denselayer4.relu2 ,
private_model.features[4].denselayer5.relu1 ,
private_model.features[4].denselayer5.relu2 ,
private_model.features[4].denselayer6.relu1 ,
private_model.features[4].denselayer6.relu2 ,
private_model.features[5].relu ,
private_model.features[6].denselayer1.relu1 ,
private_model.features[6].denselayer1.relu2 ,
private_model.features[6].denselayer2.relu1 ,
private_model.features[6].denselayer2.relu2 ,
private_model.features[6].denselayer3.relu1 ,
private_model.features[6].denselayer3.relu2 ,
private_model.features[6].denselayer4.relu1 ,
private_model.features[6].denselayer4.relu2 ,
private_model.features[6].denselayer5.relu1 ,
private_model.features[6].denselayer5.relu2 ,
private_model.features[6].denselayer6.relu1 ,
private_model.features[6].denselayer6.relu2 ,
private_model.features[6].denselayer7.relu1 ,
private_model.features[6].denselayer7.relu2 ,
private_model.features[6].denselayer8.relu1 ,
private_model.features[6].denselayer8.relu2 ,
private_model.features[6].denselayer9.relu1 ,
private_model.features[6].denselayer9.relu2 ,
private_model.features[6].denselayer10.relu1 ,
private_model.features[6].denselayer10.relu2 ,
private_model.features[6].denselayer11.relu1 ,
private_model.features[6].denselayer11.relu2 ,
private_model.features[6].denselayer12.relu1 ,
private_model.features[6].denselayer12.relu2 ,
private_model.features[7].relu ,
private_model.features[8].denselayer1.relu1 ,
private_model.features[8].denselayer1.relu2 ,
private_model.features[8].denselayer2.relu1 ,
private_model.features[8].denselayer2.relu2 ,
private_model.features[8].denselayer3.relu1 ,
private_model.features[8].denselayer3.relu2 ,
private_model.features[8].denselayer4.relu1 ,
private_model.features[8].denselayer4.relu2 ,
private_model.features[8].denselayer5.relu1 ,
private_model.features[8].denselayer5.relu2 ,
private_model.features[8].denselayer6.relu1 ,
private_model.features[8].denselayer6.relu2 ,
private_model.features[8].denselayer7.relu1 ,
private_model.features[8].denselayer7.relu2 ,
private_model.features[8].denselayer8.relu1 ,
private_model.features[8].denselayer8.relu2 ,
private_model.features[8].denselayer9.relu1 ,
private_model.features[8].denselayer9.relu2 ,
private_model.features[8].denselayer10.relu1 ,
private_model.features[8].denselayer10.relu2 ,
private_model.features[8].denselayer11.relu1 ,
private_model.features[8].denselayer11.relu2 ,
private_model.features[8].denselayer12.relu1 ,
private_model.features[8].denselayer12.relu2 ,
private_model.features[8].denselayer13.relu1 ,
private_model.features[8].denselayer13.relu2 ,
private_model.features[8].denselayer14.relu1 ,
private_model.features[8].denselayer14.relu2 ,
private_model.features[8].denselayer15.relu1 ,
private_model.features[8].denselayer15.relu2 ,
private_model.features[8].denselayer16.relu1 ,
private_model.features[8].denselayer16.relu2 ,
private_model.features[8].denselayer17.relu1 ,
private_model.features[8].denselayer17.relu2 ,
private_model.features[8].denselayer18.relu1 ,
private_model.features[8].denselayer18.relu2 ,
private_model.features[8].denselayer19.relu1 ,
private_model.features[8].denselayer19.relu2 ,
private_model.features[8].denselayer20.relu1 ,
private_model.features[8].denselayer20.relu2 ,
private_model.features[8].denselayer21.relu1 ,
private_model.features[8].denselayer21.relu2 ,
private_model.features[8].denselayer22.relu1 ,
private_model.features[8].denselayer22.relu2 ,
private_model.features[8].denselayer23.relu1 ,
private_model.features[8].denselayer23.relu2 ,
private_model.features[8].denselayer24.relu1 ,
private_model.features[8].denselayer24.relu2 ,
private_model.features[9].relu ,
private_model.features[10].denselayer1.relu1 ,
private_model.features[10].denselayer1.relu2 ,
private_model.features[10].denselayer2.relu1 ,
private_model.features[10].denselayer2.relu2 ,
private_model.features[10].denselayer3.relu1 ,
private_model.features[10].denselayer3.relu2 ,
private_model.features[10].denselayer4.relu1 ,
private_model.features[10].denselayer4.relu2 ,
private_model.features[10].denselayer5.relu1 ,
private_model.features[10].denselayer5.relu2 ,
private_model.features[10].denselayer6.relu1 ,
private_model.features[10].denselayer6.relu2 ,
private_model.features[10].denselayer7.relu1 ,
private_model.features[10].denselayer7.relu2 ,
private_model.features[10].denselayer8.relu1 ,
private_model.features[10].denselayer8.relu2 ,
private_model.features[10].denselayer9.relu1 ,
private_model.features[10].denselayer9.relu2 ,
private_model.features[10].denselayer10.relu1 ,
private_model.features[10].denselayer10.relu2 ,
private_model.features[10].denselayer11.relu1 ,
private_model.features[10].denselayer11.relu2 ,
private_model.features[10].denselayer12.relu1 ,
private_model.features[10].denselayer12.relu2 ,
private_model.features[10].denselayer13.relu1 ,
private_model.features[10].denselayer13.relu2 ,
private_model.features[10].denselayer14.relu1 ,
private_model.features[10].denselayer14.relu2 ,
private_model.features[10].denselayer15.relu1 ,
private_model.features[10].denselayer15.relu2 ,
private_model.features[10].denselayer16.relu1 ,
private_model.features[10].denselayer16.relu2
            ]

            for layer_idx, (base_layer, private_layer) in enumerate(zip(layer_paths, layer_paths_)):
                #routine for calculating `sensitivity` of activation layers, for both the private and non-private models
                layer_ga_base = LayerGradientXActivation(base_model, base_layer, multiply_by_inputs=0)
                layer_ga_pvt = LayerGradientXActivation(private_model, private_layer, multiply_by_inputs=0)

                #routine for calculating activation of activation layers, for both the private and non-private models
                layer_act_base = LayerActivation(base_model, base_layer)
                layer_act_pvt = LayerActivation(private_model, private_layer)

                act_base = layer_act_base.attribute(images).view(images.size(0), -1).to(device)
                act_pvt = layer_act_pvt.attribute(images).view(images.size(0), -1).to(device)

                attribution_base = layer_ga_base.attribute(common_images, common_labels).view(common_images.size(0), -1).to(device)
                attribution_private = layer_ga_pvt.attribute(common_images, common_labels).view(common_images.size(0), -1).to(device)

                #significance test for activations using HSIC with gamma approximation at 5% significance level
                act_test_result_gamma = perform_gamma_approximation_hsic_independence_testing(
                                data_x=act_base.cpu().numpy(),
                                data_y=act_pvt.cpu().numpy(),
                                kernel_kx=KernelWrapper(...), #we employed linear and RBF kernels
                                kernel_ky=KernelWrapper(...),
                                test_level=0.05  
                            )
                #significance test for sensitivity using HSIC with gamma approximation at 5% significance level
                attr_test_result_gamma = perform_gamma_approximation_hsic_independence_testing(
                                data_x=attribution_base.cpu().numpy(),
                                data_y=attribution_private.cpu().numpy(),
                                kernel_kx=KernelWrapper(...), #we employed linear and RBF kernels
                                kernel_ky=KernelWrapper(...),
                                test_level=0.05  
                            )            
            

                csv_writer.writerow([
                    os.path.basename(private_model_path),
                    f'Layer_{layer_idx}',
                    act_test_result_gamma['Reject H0 (H0 : X _||_ Y | Z)'],
                    attr_test_result_gamma['Reject H0 (H0 : X _||_ Y | Z)']
                    
                ])



