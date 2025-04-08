import pytest
import torch
import torchvision
from STOODX.featureStractor import FeatureStractor


@pytest.mark.parametrize(
    ("model_name", "weights", "feature_name"),
    [
        ["efficientnet_b2", "DEFAULT", "avgpool"],
        ["efficientnet_b3", "DEFAULT", "avgpool"],
        ["efficientnet_v2_s", "DEFAULT", "avgpool"],
        ["vit_b_16", "DEFAULT", "encoder"],
        ["vit_b_16", "IMAGENET1K_SWAG_E2E_V1", "encoder"],
        ["regnet_y_16gf", "IMAGENET1K_SWAG_E2E_V1", "avgpool"],
    ],
)
@pytest.mark.parametrize("relative", [True, False])
def test_models(model_name:str,weights:str,feature_name:str,relative:bool):
    input_dimension = 224 
    if model_name == "vit_b_16" and weights == "IMAGENET1K_SWAG_E2E_V1":
        input_dimension = 384
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.__dict__[model_name](weights=weights).to(device)
    feature_estractor = FeatureStractor(model = model, device=device, feature_name=feature_name,atribut=False,relative=relative)
    activations = feature_estractor.feature_activations(torch.randn(1,3,input_dimension,input_dimension))
    relevances = feature_estractor.atribute(torch.randn(1,3,input_dimension,input_dimension))
    print(activations)
    print(relevances)
    print(activations.shape)
    print(relevances.shape)
    
    assert True
