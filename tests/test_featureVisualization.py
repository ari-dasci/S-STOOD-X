import pytest
import torch
import os
import json
import numpy as np
from openood.evaluation_api import Evaluator
from matplotlib import pyplot as plt
from openood.networks import ResNet18_32x32
import pandas as pd

from STOODX.STOODXPostprocessor import STOODXPostprocessor
from STOODX.featureVisualization import FeatureExplanation
from crp.visualization import FeatureVisualization
from crp.image import plot_grid
from crp.helper import get_layer_names
import torchvision

cmaps = ["Purples", "Blues", "Greens", "Oranges", "Reds", "Greys"]


from torchvision.models.vision_transformer import VisionTransformer

class CLSExtractor(torch.nn.Module):
    # Use a torch function to extract the CLS token from the output of the encoder
    
    def forward(self, x):
        feature_vector = torch.index_select(x, 1, torch.tensor([0], device=x.device)).squeeze(1)
        
        feature_vector = feature_vector
        return feature_vector
class ViT_B_16(VisionTransformer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=768,
                 mlp_dim=3072,
                 num_classes=1000):
        super(ViT_B_16, self).__init__(image_size=image_size,
                                       patch_size=patch_size,
                                       num_layers=num_layers,
                                       num_heads=num_heads,
                                       hidden_dim=hidden_dim,
                                       mlp_dim=mlp_dim,
                                       num_classes=num_classes)
        self.feature_size = hidden_dim
        # self.CLS_token must be defined as an nn.Module that recibes an input x of shape (batch_size, hidden_dim,...) and returns the [:,0] component of the input.
        self.CLS_token = CLSExtractor()


    def forward(self, x, return_feature=False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]


        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = self.CLS_token(x)

        if return_feature:
            return self.heads(x), x
        else:
            return self.heads(x)

    def forward_threshold(self, x, threshold):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = self.CLS_token(x)

        feature = x.clip(max=threshold)
        logits_cls = self.heads(feature)

        return logits_cls

    def get_fc(self):
        fc = self.heads[0]
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.heads[0]


def save_tensor_as_image(tensor, file_path):
    image = tensor.to("cpu").numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    plt.imshow(image)#, cmap="gray")
    plt.axis("off")
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_heatmap_as_image(heatmap, file_path,max_value=None,cmap_index=0):
    #Dimension of the heatmap is (1, H, W)
    
    heatmap = heatmap.to("cpu").numpy()
    heatmap = heatmap[0]
    if max_value != None:
        heatmap = heatmap / max_value
    else:
        heatmap = heatmap/np.abs(heatmap).max()
    # cmap must be blue when -1 and red when 1. Must me black when 0
    # I must use the cmap
    
    plt.imshow(heatmap, cmap=cmaps[cmap_index], interpolation="nearest", vmin=-1, vmax=1)
    plt.axis("off")
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_sum_tensor_and_heatmap(filepath,tensor, heatmap,cmap_index=0):
    tensor = tensor.to("cpu").detach().numpy()
    heatmap = heatmap.to("cpu").detach().numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)
    tensor = tensor * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    tensor = np.clip(tensor, 0, 1)
    heatmap_plot = heatmap
    heatmap_plot = heatmap_plot[0]
    heatmap_plot = heatmap_plot / np.abs(heatmap_plot).max()
    heatmap_plot = plt.cm.get_cmap(cmaps[cmap_index])(heatmap_plot)

    # Transform tensor (224,224,3) to (224,224,4) where the last channel is the heatmap
    
    
    
    
    tensor_with_alpha = np.zeros((224, 224, 4))
    tensor_with_alpha[:, :, 0:3] = tensor
    tensor_with_alpha[:, :, 3] = np.abs(heatmap[0])/np.abs(heatmap[0]).max()
    
    image = 0.5 * tensor_with_alpha + 0.5 * heatmap_plot
    plt.imshow(image)
    plt.axis("off")
    plt.imsave(filepath, image)


num_classes = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}


dists = {
    "cosine": lambda x, y: 1 - torch.cosine_similarity(x, y, dim=1),
    "normCosine": lambda x, y: (1 - torch.cosine_similarity(x, y, dim=1))
    * (torch.min(torch.norm(x, dim=1), torch.norm(y, dim=1)))
    / torch.max(torch.norm(x, dim=1), torch.norm(y, dim=1)),
}
statistics = {
    "min": lambda x: x["p_value"].min(),
    "max": lambda x: x["p_value"].max(),
    "mean": lambda x: x["p_value"].mean(),
    "median": lambda x: x["p_value"].median(),
    "first": lambda x: x["p_value"].iloc[0],
    "last": lambda x: x["p_value"].iloc[-1],
}
@pytest.mark.parametrize(
    ("model_name", "weights", "feature_name"),
    [
        # [
        #    "resnet18",
        #    "./pretrained_models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt",
        #    "avgpool",
        # ],
        [
            "resnet50",
            torchvision.models.ResNet50_Weights.DEFAULT,
            "avgpool",
        ],
        [
            "vit_b_16",
            torchvision.models.ViT_B_16_Weights.DEFAULT,
            "CLS_token",
        ],
    ],
)
@pytest.mark.parametrize("id_val", [i for i in range(20)])
@pytest.mark.parametrize("id_set", ["imagenet"])
@pytest.mark.parametrize(
    "ood_type,ood_dataset",
    [
        ("near", "ssb_hard"),
        ("near", "ninco"),
        # ("near", "tin"),
        ("far", "inaturalist"),
        ("far", "openimage_o"),
        ("far", "textures"),
    ],
)
@pytest.mark.parametrize("K", [500])
def test_FeatureExplanation(model_name, weights, id_set,feature_name,id_val,ood_type,ood_dataset,K):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load the ViT model
    net = torchvision.models.__dict__[model_name](num_classes=num_classes[id_set],weights=weights)
    if model_name == "vit_b_16":
        state_dict = net.state_dict()
        net = ViT_B_16(num_classes=num_classes[id_set])
        net.load_state_dict(state_dict)
    net = net.to(device)

    net.eval()

    config = {
        "K": K,
        "distance": dists["cosine"],
        "feature_name": feature_name,
        "intraclass": True,
        "quantil": 0.75,
        "atribut": False,
        "relative": False,
        "p_value_statistic": statistics["mean"],
        "partition": "train",
        "whole_test": True,
        "model_name": model_name,
        "id_name": id_set,
    }
    postprocessorOOD = STOODXPostprocessor(config)

    evaluator = Evaluator(
        net=net,
        id_name=id_set,
        data_root="./utils/data",
        postprocessor=postprocessorOOD,
        batch_size=124,
        num_workers=2,
    )
    dataloader1 = evaluator.dataloader_dict["id"]["train"]
    dataloader2 = evaluator.dataloader_dict["id"]["val"]
    dataloader_ood = evaluator.dataloader_dict["ood"][ood_type][ood_dataset]

    # Make dataloader3 a dataloader of just one example of the dataloader2
    dataloader3 = torch.utils.data.DataLoader(
        [dataloader2.dataset[id_val]],
        batch_size=1,
        shuffle=False,
    )

    feature_explainator = FeatureExplanation(net, postprocessorOOD, dataloader1)
    conf = feature_explainator.ood_score(dataloader3)[0].item()
    print("K: ", K)
    print("Confidence: ", conf)

    x_ = dataloader3.dataset[0]["data"].to(device)
    nearest_examples_indexes = feature_explainator.closest_examples(x_, k=5)
    print("Nearest index example: ", nearest_examples_indexes[0])
    nearest_examples = []
    for index in nearest_examples_indexes:
        nearest_examples.append(dataloader1.dataset[index]["data"].to(device))
    print("Saving nearest examples")
    os.makedirs(f"visualization/ID/{id_val}/{model_name}", exist_ok=True)
    save_tensor_as_image(x_, f"visualization/ID/{id_val}/{model_name}/example_{conf}.png")
    for i, example in enumerate(nearest_examples):
        save_tensor_as_image(
            example, f"visualization/ID/{id_val}/{model_name}/NN_{i}.png"
        )
    n_features = 5
    main_features = feature_explainator.features_presence(x_.unsqueeze(0), k=n_features)

    max_heatmap = feature_explainator.presence_of_feature(x_, main_features[0:n_features])
    print("Save all-the-features map")
    save_heatmap_as_image(max_heatmap, f"visualization/ID/{id_val}/{model_name}/all_features_heatmap.png")
    save_sum_tensor_and_heatmap(f"visualization/ID/{id_val}/{model_name}/all_features_sum.png",x_,max_heatmap)
    for i in range(n_features):
        heatmap = feature_explainator.presence_of_feature(x_, main_features[i:i+1])

        print(f"Heatmap of main feature {i}")
        save_heatmap_as_image(heatmap, f"visualization/ID/{id_val}/{model_name}/feat_{i}_heatmap.png",cmap_index=i)
        save_sum_tensor_and_heatmap(f"visualization/ID/{id_val}/{model_name}/feat_{i}_sum.png",x_,heatmap,cmap_index=i)
        print(f"Heatmap on the nearest examples of main feature {i}")
        for j, example in enumerate(nearest_examples):
            heatmap = feature_explainator.presence_of_feature(example, main_features[i:i+1])
            save_heatmap_as_image(
                heatmap,
                f"visualization/ID/{id_val}/{model_name}/feat_{i}_NN_{j}_heatmap.png",
                cmap_index =i,
            )
            save_sum_tensor_and_heatmap(f"visualization/ID/{id_val}/{model_name}/feat_{i}_NN_{j}_sum.png",example,heatmap,cmap_index=i)

    dataloader_ood2 = torch.utils.data.DataLoader(
        [dataloader_ood.dataset[id_val]],
        batch_size=1,
        shuffle=False,
    )

    feature_explainator = FeatureExplanation(net, postprocessorOOD, dataloader1)
    conf = feature_explainator.ood_score(dataloader_ood2)[0].item()
    print("Confidence: ", conf)
    x_ = dataloader_ood2.dataset[0]["data"].to(device)
    nearest_examples_indexes = feature_explainator.closest_examples(x_, k=5)
    print("Nearest index example: ", nearest_examples_indexes[0])
    nearest_examples = []
    for index in nearest_examples_indexes:
        nearest_examples.append(dataloader1.dataset[index]["data"].to(device))
    print("Saving nearest examples")

    os.makedirs(f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}", exist_ok=True)
    save_tensor_as_image(x_, f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/example_{conf}.png")
    for i, example in enumerate(nearest_examples):
        save_tensor_as_image(
            example, f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/NN_{i}.png"
        )
    main_features = feature_explainator.features_presence(x_.unsqueeze(0), k=n_features)
    max_heatmap = (
        feature_explainator.presence_of_feature(x_, main_features[0:n_features])
    )
    print("Save all-the-features map")
    save_heatmap_as_image(
        max_heatmap,
        f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/all_features_heatmap.png",
    )
    save_sum_tensor_and_heatmap(f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/all_features_sum.png",x_,max_heatmap)
    for i in range(n_features):
        heatmap = feature_explainator.presence_of_feature(x_, main_features[i : i + 1])
        print(f"Heatmap of main feature {i}")
        save_heatmap_as_image(
            heatmap, f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/feat_{i}_heatmap.png",cmap_index=i,
        )
        save_sum_tensor_and_heatmap(f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/feat_{i}_sum.png",x_,heatmap,cmap_index=i)
        print(f"Heatmap on the nearest examples of main feature {i}")
        for j, example in enumerate(nearest_examples):
            heatmap = feature_explainator.presence_of_feature(
                example, main_features[i : i + 1]
            )
            save_heatmap_as_image(
                heatmap,
                f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/feat_{i}_NN_{j}_heatmap.png",
                cmap_index=i,
            )
            save_sum_tensor_and_heatmap(f"visualization/OOD/{ood_type}_{ood_dataset}/{id_val}/{model_name}/feat_{i}_NN_{j}_sum.png",example,heatmap,cmap_index=i)

    assert True
