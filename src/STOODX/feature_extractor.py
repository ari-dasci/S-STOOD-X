import torch
from crp.attribution import CondAttribution
from zennit.composites import Composite


class _BatchAttribution:
    """Attributions for a batch, concatenated along the batch dimension."""

    def __init__(self, activations: dict, relevances: dict):
        self.activations = activations
        self.relevances = relevances


class FeatureExtractor(torch.nn.Module):
    """
    Class for Feature Extraction

    A Feature Extractor object is used to extract features and their relevance from a specific layer of a model.

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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_name: str,
        atribut: bool,
        composite: Composite = None,
        relative=True,
    ):
        super(FeatureExtractor, self).__init__()
        self.model = model.to(device)
        self.feature_name = feature_name
        self.atribution = CondAttribution(self.model, no_param_grad=True)
        self.composite = composite
        self.device = device
        self.atribut = atribut
        self.relative = relative

    def _atribution_calc(self, x: torch.Tensor):
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[0] == 1:
            return self._atribution_calc_single(x)

        # Relevance propagation resolves one target class per call, so a batch is attributed
        # sample by sample and the results are concatenated along the batch dimension.
        activations, relevances = [], []
        for index in range(x.shape[0]):
            attribution = self._atribution_calc_single(x[index : index + 1])
            activations.append(attribution.activations)
            relevances.append(attribution.relevances)

        return _BatchAttribution(
            {name: torch.cat([item[name] for item in activations], dim=0) for name in activations[0]},
            {name: torch.cat([item[name] for item in relevances], dim=0) for name in relevances[0]},
        )

    def _atribution_calc_single(self, x: torch.Tensor):
        _x = x.to(self.device)
        _x.requires_grad = True
        x_class = self.model(_x)

        x_class = torch.argmax(x_class, dim=1).item()

        attr = self.atribution(
            _x,
            conditions=[{"y": [x_class]}],
            composite=self.composite,
            record_layer=[self.feature_name],
            init_rel=-1,
        )

        return attr

    def forward(self, x: torch.Tensor):
        """
        Forward method for the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The model prediction.
        """
        return self.model(x)

    def features(self, x: torch.Tensor):
        """
        Extract features from the input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to extract features from.

        Returns
        -------
        torch.Tensor
            The extracted features.
        """
        if self.atribut:
            return self.atribute(x)
        else:
            return self.feature_activations(x)

    def feature_activations(self, x: torch.Tensor):
        """
        Extract feature activations from the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The feature activations.
        """
        attr = self._atribution_calc(x)
        importance_matrix = attr.activations[self.feature_name]
        if self.feature_name == "encoder" or self.feature_name == "features":
            # ViT: use only the first feature vector (CLS token)
            # Reference: https://github.com/pytorch/vision/blob/ed55b0309fc3ed7d8abc4e4172b8a3c9852ef454/torchvision/models/vision_transformer.py#L301
            importance_matrix = importance_matrix[:, 0]
        if self.relative:
            importance_matrix = importance_matrix / torch.max(torch.abs(importance_matrix))
        return importance_matrix

    def atribute(self, x: torch.Tensor):
        """
        Extract feature attributions using relevance propagation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The feature relevance/attributions.
        """
        attr = self._atribution_calc(x)
        importance = attr.relevances[self.feature_name]
        if self.feature_name == "encoder":
            # ViT: use only the first feature vector (CLS token)
            importance = importance[:, 0]
        if self.relative:
            importance = importance / torch.max(torch.abs(importance))
        return importance
