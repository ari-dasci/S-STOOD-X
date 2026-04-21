import torch
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept

from STOODX.STOODXPostprocessor import STOODXPostprocessor
class FeatureExplanation(torch.nn.Module):

    def __init__(self, net: torch.nn.Module, stoodx_postprocessor: STOODXPostprocessor, dataset):
        super(FeatureExplanation, self).__init__()
        self.net = net.to(stoodx_postprocessor.device)
        self.STOODXPostprocessor = stoodx_postprocessor
        self.dataloader = dataset

    def setup(self):
        self.STOODXPostprocessor.setup(self.net, self.dataloader, self.dataloader)

    def complete_explanation(self, x_: torch.Tensor):
        """
        Explain the decisions of the model on x_.

        Steps:
        1. Is x_ an ID sample? Test with STOODXPostprocessor.
        2. If x_ is an ID sample, find K similar examples from the train set.
        3. Identify which features make them similar to x_ using CondAttribution.
        4. Locate specific features in the example using Composite class.
        5. Locate features in each K similar example.
        """
        pass

    def ood_score(self, dataloader) -> torch.Tensor:
        """Calculate OOD scores for the given dataloader."""
        preds, confs, labels = self.STOODXPostprocessor.inference(self.net, dataloader)
        return confs

    def closest_examples(self, x_, k: int = 3) -> torch.Tensor:
        """
        Find the K closest examples to x_ in the dataset of the same class.

        Parameters
        ----------
        x_ : torch.Tensor
            The example to find the closest examples.
        k : int
            The number of closest examples to find.

        Returns
        -------
        torch.Tensor
            The indexes of the closest examples.
        """
        x_ = x_.to(self.STOODXPostprocessor.device)
        x_features = self.STOODXPostprocessor.oodTest.features(x_.unsqueeze(0)).squeeze(0)
        x_class = torch.argmax(self.STOODXPostprocessor.oodTest(x_.unsqueeze(0)).squeeze(0), dim=0)

        # Indexes of the examples of the same class
        same_class_indexes = torch.where(self.STOODXPostprocessor.oodTest.classes == x_class)

        feat_subset_class = self.STOODXPostprocessor.oodTest.feats[same_class_indexes]
        sum_abs_features = torch.sum(torch.abs(feat_subset_class), dim=0)

        IQR = torch.quantile(feat_subset_class, 0.75, dim=0) - torch.quantile(feat_subset_class, 0.25, dim=0)
        Q3 = torch.quantile(feat_subset_class, 0.75, dim=0)

        n_features = torch.sum(sum_abs_features >= Q3 + self.STOODXPostprocessor.oodTest.quantile * IQR).item()

        least_present_features = torch.argsort(sum_abs_features, descending=True)[n_features:]
        feat_subset_class[:, least_present_features] = 0
        x_features[least_present_features] = 0

        x_features = x_features.squeeze().unsqueeze(0)

        x_distances = self.STOODXPostprocessor.oodTest.distance(x_features, feat_subset_class)

        closest_k_distances = torch.argsort(x_distances)[:k]

        return same_class_indexes[0][closest_k_distances]

    def indexes_to_dataset_examples(self, example_indexes: torch.Tensor) -> torch.Tensor:
        """
        Show the examples in the dataset with the indexes in example_indexes.

        Parameters
        ----------
        example_indexes : torch.Tensor
            The indexes of the examples to show.

        Returns
        -------
        torch.Tensor
            The examples in the dataset with the indexes in example_indexes.
        """
        return self.dataloader[example_indexes]

    def features_presence(self, X: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Show the most important k features in the examples in the set X.

        Parameters
        ----------
        X : torch.Tensor
            The examples to find the most important features.
        k : int
            The number of features to find. Default is 1.

        Returns
        -------
        torch.Tensor
            The indexes of the most important features.
        """
        X_features = self.STOODXPostprocessor.oodTest.features(X)

        sum_abs_features = torch.sum(torch.abs(X_features), dim=0)
        return torch.argsort(sum_abs_features, dim=0, descending=True)[:k]

    def presence_of_feature(self, example: torch.Tensor, feature_indexes: torch.Tensor) -> torch.Tensor:
        """
        Calculate the presence of the feature in the example.

        The presence is calculated with the Composite class using the feature_indexes
        with the zennit-crp library.

        Parameters
        ----------
        example : torch.Tensor
            The example to calculate the presence of the feature.
        feature_indexes : torch.Tensor
            The indexes of the features to calculate the presence.

        Returns
        -------
        torch.Tensor
            Heatmap with the presence of the feature.
        """
        class_index = torch.argmax(
            self.STOODXPostprocessor.oodTest(example.unsqueeze(0)).squeeze(0), dim=0
        )

        cc = ChannelConcept()
        example.requires_grad = True

        composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
        atribution = CondAttribution(self.net, no_param_grad=True)
        feature_indexes = feature_indexes.squeeze().tolist()

        conditions = [
            {
                "y": class_index.item(),
                self.STOODXPostprocessor.feature_name: feature_indexes,
            }
        ]

        attr = atribution(example.unsqueeze(0), conditions, composite)

        return attr.heatmap
