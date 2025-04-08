import pytest
import torch
import os
import json
import numpy as np
from openood.evaluation_api import Evaluator
from openood.networks.resnet50 import ResNet50
from openood.networks import ResNet18_32x32
from openood.networks.vit_b_16 import ViT_B_16
from openood.networks.swin_t import Swin_T
from openood.networks.regnet_y_16gf import RegNet_Y_16GF


import pandas as pd

from STOODX.STOODXPostprocessor import STOODXPostprocessor
import torchvision


def convert_numpy_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    else:
        return obj


num_classes = {"cifar10":10, "cifar100":100, "imagenet200":200,"imagenet":1000}

dists = {"cosine":lambda x,y:1 - torch.cosine_similarity(x, y,dim=1), 
        "normCosine":lambda x,y: (1-torch.cosine_similarity(x, y,dim=1))*(torch.min(torch.norm(x,dim=1),torch.norm(y,dim=1)))/torch.max(torch.norm(x,dim=1),torch.norm(y,dim=1))}
statistics = {"min":lambda x:x["p_value"].min(),"max":lambda x:x["p_value"].max(),
              "mean":lambda x:x["p_value"].mean(),"median":lambda x:x["p_value"].median(),
              "first":lambda x:x["p_value"].iloc[0],
              "last":lambda x:x["p_value"].iloc[-1]}


@pytest.mark.parametrize("id_set", ["cifar10"])
@pytest.mark.parametrize("atribut", [False])
@pytest.mark.parametrize("relative", [False])
@pytest.mark.parametrize("fsood", ["ood"])
@pytest.mark.parametrize(
    ("model_name", "weights", "feature_name"),
    [
        [
            "resnet18",
            "./pretrained_models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt",
            "avgpool",
        ],
    ],
)
@pytest.mark.parametrize(
    "K,partition",
    [
        (500, "train"),
    ],
)
@pytest.mark.parametrize("intraclass", [True])
@pytest.mark.parametrize("dist", ["cosine"])
@pytest.mark.parametrize("p_value_statistic", ["mean"])
@pytest.mark.parametrize("whole_test", [True])
@pytest.mark.parametrize(
    "quantile", [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
)
def test_id_STOODX_cifar10_postProcessor(
    id_set,
    model_name,
    weights,
    feature_name,
    K,
    partition,
    intraclass,
    atribut,
    fsood,
    relative,
    dist,
    p_value_statistic,
    whole_test,
    quantile,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_file = (
        f"XAI_{model_name}_{id_set}_{feature_name}_NN_{K}_{fsood}_WT_{whole_test}"
    )

    result_file = result_file + f"_{partition}_Q{quantile}"

    if os.path.exists(f"./results/{result_file}/metrics.csv"):
        assert True
        print(f"Reading results from {result_file}/metrics.csv")
        df = pd.read_csv(f"./results/{result_file}/metrics.csv")
        print(df)
        return
    elif not os.path.exists(f"./results/{result_file}"):
        os.makedirs(f"./results/{result_file}",exist_ok=True)
    print(f"Calculating results in {result_file}:")
    net = ResNet18_32x32(num_classes=10).to(device)
    net.load_state_dict(torch.load(weights, map_location=device))

    net.to(device)

    net.eval()
    config = {
        "K": K,
        "distance": dists[dist],
        "feature_name": feature_name,
        "intraclass": intraclass,
        "quantil": quantile,
        "atribut": atribut,
        "relative": relative,
        "p_value_statistic": statistics[p_value_statistic],
        "partition": partition,
        "whole_test": whole_test,
        "model_name": model_name,
        "id_name": id_set,
    }

    postprocessorOOD = STOODXPostprocessor(config)

    evaluator = Evaluator(
        net=net,
        id_name=id_set,
        data_root="./utils/data",
        postprocessor=postprocessorOOD,
        batch_size=16,
        num_workers=2,
    )

    metrics = evaluator.eval_ood(fsood=(fsood == "fsood"))

    with open(f"./results/{result_file}/metrics.csv", "w") as f:
        # Imprimir los resultados en el archivo
        metrics.to_csv(f, sep="\t")
    with open(f"./results/{result_file}/results.json", "w") as f:
        scores = evaluator.scores

        json.dump(convert_numpy_to_list(scores), f, indent=2)
    assert True


@pytest.mark.parametrize("id_set", ["cifar10"])
@pytest.mark.parametrize(
    "model_name,weights",
    [
        (
            "resnet18",
            "./pretrained_models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt",
        ),
    ],
)
@pytest.mark.parametrize("fsood", [False])
@pytest.mark.parametrize(
    "baseline_postprocessor",
    [
        "mds",
        "rmds",
        "vim",
        "knn",
        "scale",
        "ash",
        "react",
        "odin",
        "godin",
        "gradnorm",
        "nnguide",
    ],
)
def test_Baseline_cifar10_postProcessor(
    id_set, model_name, weights, fsood, baseline_postprocessor
):
    result_file = (
        f"{baseline_postprocessor}_{model_name}_{id_set}_{'fsood' if fsood else 'ood'}"
    )
    if os.path.exists(f"./results/{result_file}/metrics.csv"):
        assert True
        print(f"Reading results from {result_file}/metrics.csv")
        df = pd.read_csv(f"./results/{result_file}/metrics.csv")
        print(df)
        return
    else:
        os.makedirs(f"./results/{result_file}",exist_ok=True)
    print(f"Calculating results in {result_file}:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ResNet18_32x32(num_classes=num_classes[id_set]).to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()
    evaluator = Evaluator(
        net=net,
        id_name=id_set,
        data_root="./utils/data",
        postprocessor_name=baseline_postprocessor,
        batch_size=16,
        num_workers=2,
    )

    metrics = evaluator.eval_ood(fsood=False)

    with open(f"./results/{result_file}/metrics.csv", "w") as f:
        metrics.to_csv(f, sep="\t")
    with open(f"./results/{result_file}/results.json", "w") as f:
        scores = evaluator.scores
        json.dump(convert_numpy_to_list(scores), f, indent=2)
    assert True
