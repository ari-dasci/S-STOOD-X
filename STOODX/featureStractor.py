import torch
from crp.attribution import CondAttribution
from zennit.composites import Composite
from typing import List

class FeatureStractor(torch.nn.Module):
    '''
    Class for Feature Extraction
    
    A Feature Estractor object is used to extract features and their relevance from a specific layer of a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to extract features from.
    device : torch.device
        The device to use for the model.
    feature_name : str
        The name of layer representing the feature to extract.
    composite : zennit.composites.Composite, optional
        The composite to use for the attribution calculation. If not provided, the default composite will be used.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 feature_name:str,
                 atribut:bool,
                 composite:Composite = None,
                 relative=True
                 ):
        super(FeatureStractor, self).__init__()
        self.model = model.to(device)
        self.feature_name = feature_name
        self.atribution = CondAttribution(self.model,no_param_grad=True)
        self.composite = composite
        self.device = device
        self.atribut = atribut
        self.relative = relative
        

    '''
    Method to extract features and their relevance.
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to extract features from.
        
    Returns
    -------
    features : torch.Tensor
        The extracted features.
    relevance : torch.Tensor
        The relevance of the extracted features.
    '''
    def _atribution_calc(self,x:torch.Tensor):
        _x = x.to(self.device)
        _x.requires_grad = True
        x_class = self.model(_x)

        x_class = torch.argmax(x_class,dim=1).item()
        
        attr = self.atribution(
            _x,
            conditions=[{'y':[x_class]}],
            composite=self.composite,
            record_layer=[self.feature_name],
            init_rel=-1,
        )
        
        return attr

    '''
    Forward method for the model wraped
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to extract features from.
        
    Returns
    -------
    model_rediction : torch.Tensor
        The model prediction.
    '''
    def forward(self,x:torch.Tensor):
        return self.model(x)

    '''
    Call method for the model prediction
    '''
    def __call__(self,x:torch.Tensor):
        return self.forward(x)

    '''
    Method to extract the relative activations multiplied by the relevance  
    of the model for a specific input
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to extract features from.
        
    Returns
    -------
    activations : torch.Tensor
        The relative activations of the model for the input.
    
    '''
    def features(self,x:torch.Tensor):
        if self.atribut:
            return self.atribute(x)
        else:
            return self.feature_activations(x)
    def feature_activations(self,x:torch.Tensor):
        attr = self._atribution_calc(x)
        importance_matrix = attr.activations[self.feature_name]
        if self.feature_name == "encoder" or self.feature_name == "features":
            # This is how ViT works for their feature space, I don't know why it is only used the first feature vector
            # of the final features. Reference in the forward function of the VisionTransformer class, line 12:
            # https://github.com/pytorch/vision/blob/ed55b0309fc3ed7d8abc4e4172b8a3c9852ef454/torchvision/models/vision_transformer.py#L301C20-L302C1
            importance_matrix = importance_matrix[:,0]
        if self.relative:
            importance_matrix = importance_matrix/torch.max(torch.abs(importance_matrix))
        return importance_matrix

    '''
    Method to extract the relative activations of the model for a specific input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to extract features from.

        Returns
        -------
        activations : torch.Tensor
            The relative activations of the model for the input.
    '''
    def atribute(self,x:torch.Tensor):
        attr = self._atribution_calc(x)
        importance = attr.relevances[self.feature_name] #* attr.activations[self.feature_name]
        if self.feature_name == "encoder":
            # This is how ViT works for their feature space, I don't know why they only use the first feature vector
            # of the final features. Reference in the forward function of the VisionTransformer class, line 12:
            # https://github.com/pytorch/vision/blob/ed55b0309fc3ed7d8abc4e4172b8a3c9852ef454/torchvision/models/vision_transformer.py#L301C20-L302C1
            importance = importance[:,0]
        if self.relative:
            importance = importance/torch.max(torch.abs(importance))
        
        return importance
