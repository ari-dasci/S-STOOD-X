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


@pytest.mark.parametrize("id_set", ["imagenet"])
@pytest.mark.parametrize("atribut", [True])
@pytest.mark.parametrize("relative", [False])
@pytest.mark.parametrize("fsood", ["ood"])
@pytest.mark.parametrize(
    ("model_name", "weights", "feature_name"),
    [
        ["resnet50", torchvision.models.ResNet50_Weights.DEFAULT, "avgpool"],
        ["vit_b_16", torchvision.models.ViT_B_16_Weights.DEFAULT, "encoder"],
        # ["swin_t", torchvision.models.Swin_T_Weights.DEFAULT, "features"],
    ],
)
@pytest.mark.parametrize(
    "K,partition,quantile",
    [
        #(9, "train",0.625),
        #(18, "train",0.625),
        #(36, "train",0.625),
        #(72, "train",0.625),
        #(144, "train",0.625),
        #(288, "train",0.625),
        (500, "train",0.625),
        #(-1, "train",0.625),
        (500, "train",0.0),
        (500, "train",0.125),
        (500, "train",0.25),
        (500, "train",0.375),
        (500, "train",0.5),
        (500, "train",0.75),
        (500, "train",0.875),
        (500, "train_val",0.625)
    ],
)
@pytest.mark.parametrize("NN_K", [1000])
@pytest.mark.parametrize("intraclass", [True])
@pytest.mark.parametrize("dist", ["cosine"])
@pytest.mark.parametrize("p_value_statistic", ["mean"])
@pytest.mark.parametrize("whole_test", [True])
def test_id_XAI_postProcessor(
    id_set, model_name,
    weights, feature_name,
    K, NN_K,partition,quantile,
    intraclass,
    atribut, fsood, relative,
    dist, p_value_statistic,
    whole_test,
):
    result_file = f"XAI_{model_name}_{id_set}_{feature_name}_NN_{K}_NNK_{NN_K}_{fsood}_WT_{whole_test}"
    result_file = result_file + f"_{partition}_Q{quantile}_atri_{atribut}"

    if os.path.exists(f"./results/{result_file}/metrics.csv"):
        assert True
        print(f"Reading results from {result_file}/metrics.csv")
        df = pd.read_csv(f"./results/{result_file}/metrics.csv")
        print(df)
        return
    elif not os.path.exists(f"./results/{result_file}"):
        os.makedirs(f"./results/{result_file}")
    print(f"Calculating results in {result_file}:")
    net = torchvision.models.__dict__[model_name](
        weights=weights, num_classes=num_classes[id_set]
    )
    net.eval()
    config = {
        "K": K,"NN_K":NN_K,"distance": dists[dist],
        "feature_name": feature_name,"intraclass": intraclass,
        "quantil": quantile,"atribut": atribut,"relative": relative,
        "p_value_statistic": statistics[p_value_statistic],"partition": partition,
        "whole_test": whole_test,"model_name":model_name,
    }

    postprocessorOOD = STOODXPostprocessor(config)
    
    evaluator = Evaluator(
        net=net,id_name=id_set,
        data_root="./utils/data",
        postprocessor=postprocessorOOD,
        batch_size=64,num_workers=2,
    )

    metrics = evaluator.eval_ood(fsood=(fsood == "fsood"))

    with open(f"./results/{result_file}/metrics.csv", "w") as f:
        metrics.to_csv(f, sep="\t")
    with open(f"./results/{result_file}/results.json", "w") as f:
        scores = evaluator.scores

        json.dump(convert_numpy_to_list(scores), f, indent=2)
    assert True


@pytest.mark.parametrize("id_set", ["imagenet"])
@pytest.mark.parametrize(
    "model_name,weights",
    [
        ("vit_b_16", torchvision.models.ViT_B_16_Weights.DEFAULT),
        ("resnet50", torchvision.models.ResNet50_Weights.DEFAULT),
        #("swin_t",torchvision.models.Swin_T_Weights.DEFAULT),
    ],
)
@pytest.mark.parametrize("fsood", [False])
@pytest.mark.parametrize("baseline_postprocessor", ["mds","rmds","vim",
                                                    "knn","scale","ash",
                                                    "react","odin",
                                                    "godin","gradnorm",
                                                    "nnguide",
    ])
def test_Baseline_postProcessor(id_set, model_name, weights, fsood,baseline_postprocessor):
    result_file = f"{baseline_postprocessor}_{model_name}_{id_set}_{'fsood' if fsood else 'ood'}"
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
    
    net = torchvision.models.__dict__[model_name](
        weights=weights, num_classes=num_classes[id_set]
    ).to(device)

    if "resnet50" in model_name:
        wraped_net = ResNet50().to(device)
        net_state_dict = net.state_dict()
        wraped_net.load_state_dict(net_state_dict)
        net = wraped_net
    elif "vit_b_16" in model_name:
        wraped_net = ViT_B_16().to(device)
        net_state_dict = net.state_dict()
        wraped_net.load_state_dict(net_state_dict)
        net = wraped_net
    else:
        assert False
    net.to(device)
    net.eval()
    evaluator = Evaluator(net=net,
        id_name=id_set,data_root="./utils/data",
        postprocessor_name=baseline_postprocessor,
        batch_size=16,num_workers=2,
    )
    metrics = evaluator.eval_ood(fsood=fsood)

    with open(f"./results/{result_file}/metrics.csv", "w") as f:
        metrics.to_csv(f, sep="\t")
    with open(f"./results/{result_file}/results.json", "w") as f:
        scores = evaluator.scores
        json.dump(convert_numpy_to_list(scores), f, indent=2)
    assert True
