import torch
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from crp.helper import get_layer_names

from STOODX.STOODXPostprocessor import STOODXPostprocessor
class FeatureExplanation(torch.nn.Module):

    def __init__(self, net:torch.nn.Module, STOODXPostprocessor:STOODXPostprocessor,dataset):
        super(FeatureExplanation, self).__init__()
        self.net = net.to(STOODXPostprocessor.device)
        self.STOODXPostprocessor = STOODXPostprocessor
        self.dataloader = dataset
    def setup(self):
        self.STOODXPostprocessor.setup(self.net, self.dataloader,self.dataloader)

    def complete_explanation(self,x_:torch.Tensor):
        # Explain x_ the decisions of the model on x_
        # 1. Is x_ an ID sample?
        #     -> Test with the test on XAIOODTestPostprocessor
        # 2. If x_ is an ID sample, which examples of the train set with the same class are similar to x_?
        #    -> Propose K similar examples
        # 3. Which features of the examples are the ones that make them similar to x_?
        #    -> Use the CondAttribution class to explain the features on the examples and select the most important ones.
        #    -> You can also use the vector vector of more similarity to all the features.
        # 4. Where can I find the specific feature in the example?
        #   -> Use the Composite class to find the feature in the example as a backpropagation. In case of using the vector of more similarity, use the composite class to find the feature in the example.
        # 5. And where can I find the feature in each K similar example?

        pass
    def ood_score(self,dataloader)->torch.Tensor:

        preds, confs, labels = self.STOODXPostprocessor.inference(self.net,dataloader)
        return confs

    def closest_examples(self,x_,k:int=3)-> torch.Tensor:
        # sphinx documantation
        '''
        Find the K closest examples to x_ in the dataset of the same class. The return must be a tensor of the indexes of the examples.
        The distance between x_ and the examples must be calculated using the distance defined in the XAIODDTest in the XAIODDTestPostprocessor.
        
        Parameters:
            x_: torch.Tensor
                The example to find the closest examples.
            k: int
                The number of closest examples to find.
        
        Returns:
            torch.Tensor
                The indexes of the closest
        
        '''
        x_.to(self.STOODXPostprocessor.device)
        x_features = self.STOODXPostprocessor.oodTest.features(x_.unsqueeze(0)).squeeze(0)
        x_class = torch.argmax(self.STOODXPostprocessor.oodTest(x_.unsqueeze(0)).squeeze(0),dim=0)

        # Indexes of the examples of the same class

        same_class_indexes = torch.where(self.STOODXPostprocessor.oodTest.classes == x_class)

        feat_subset_class = self.STOODXPostprocessor.oodTest.feats[same_class_indexes]
        sum_abs_features = torch.sum(torch.abs(feat_subset_class),dim=0)

        IQR = torch.quantile(feat_subset_class,0.75,dim=0) - torch.quantile(feat_subset_class,0.25,dim=0)
        Q3 = torch.quantile(feat_subset_class,0.75,dim=0)

        n_features = torch.sum(sum_abs_features >= Q3 + self.STOODXPostprocessor.oodTest.quantile*IQR).item()

        least_present_features = torch.argsort(sum_abs_features,descending=True)[n_features:]
        feat_subset_class[:,least_present_features] = 0
        x_features[least_present_features] = 0

        x_features = x_features.squeeze()
        x_features = x_features.unsqueeze(0)

        x_distances = self.XAIPostprocesor.oodTest.distance(x_features,feat_subset_class)

        closest_k_distances = torch.argsort(x_distances)[:k]

        return same_class_indexes[0][closest_k_distances]

    def indexes_to_dataset_examples(self,example_indexes:torch.Tensor)->torch.Tensor:
        '''
        Show the examples in the dataset with the indexes in example_indexes.
        The indexes must be the indexes of the examples in the dataset.
        
        Parameters:
            example_indexes: torch.Tensor
                The indexes of the examples to show.

        Returns:
            torch.Tensor
                The examples in the dataset with the indexes in example_indexes.
        '''
        return self.dataloader[example_indexes]

    def features_presence(self,X:torch.Tensor,k:int=1)->torch.Tensor:
        '''
        Show the most important $k$ features in the examples in the set X of examples.
        The return must be a tensor of the indexes of the features.
        
        Parameters:
            X: torch.Tensor
                The examples to find the most important features.
            k: int
                The number of features to find. 1 by default.
        Returns:
            torch.Tensor
                The indexes of the most important features. 
        '''

        X_features = self.XAIPostprocesor.oodTest.features(X)

        sum_abs_features = torch.sum(torch.abs(X_features),dim=0)
        '''print(len(sum_abs_features[sum_abs_features != 0]))
        print(sum_abs_features[sum_abs_features != 0])
        print(sum_abs_features[torch.argsort(sum_abs_features,dim=0, descending=True)])'''
        return torch.argsort(sum_abs_features,dim=0,descending=True)[:k]

    def presence_of_feature(self,example:torch.Tensor,feature_indexes:torch.Tensor)->torch.Tensor:
        '''
        Calculate the presence of the feature in the example.
        The return must be a tensor of the same size of the example with the presence of the feature.
        
        The presence of the feature is calculated with the Composite class using the feature_indexes with the zennit crp library.
        Parameters:
            example: torch.Tensor
                The example to calculate the presence of the feature.
            feature_indexes: torch.Tensor
                The indexes of the features to calculate the presence.
        '''

        class_index = torch.argmax(self.XAIPostprocesor.oodTest(example.unsqueeze(0)).squeeze(0),dim=0)

        cc = ChannelConcept()

        device = example.device
        example.requires_grad = True

        composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
        atribution = CondAttribution(self.net, no_param_grad=True)
        feature_indexes = feature_indexes.squeeze().tolist()
        conditions = [
            {
                "y": class_index.item(),
                self.XAIPostprocesor.feature_name: feature_indexes,
            }
        ]

        attr = atribution(example.unsqueeze(0), conditions,composite)

        return attr.heatmap
