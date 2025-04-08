import torch
import pandas as pd
from baycomp import two_on_multiple
from STOODX.featureStractor import FeatureStractor
from typing import Callable
from scipy import stats
import numpy as np


class STOODX:
    '''
    class for OOD Test detector. 

    Parameters
    ----------
    model : FeatureEstractor
        The model to test.
    distance :
        The distance function to use betwen the validation features and the test features. It must be a function that takes two torch.Tensors 
        and returns a torch.Tensor of shape (1,).
    ''' 
    def __init__(self, model:FeatureStractor,
                 distance:Callable = lambda x,y:torch.norm(x-y,dim=1),
                 k_neighbors:int = 50,
                 k_NNs:int = 50,
                 quantile:float = 0.99,
                 whole_test:bool = True,
                 ):
        self.model          = model
        self.distance       = distance
        self.k_neighbors    = k_neighbors
        self.k_NNs          = k_NNs
        self.quantile       = quantile
        self.whole_test     = whole_test
        
        self.feats          = None
        self.classes        = None
        self.feats_list = []
        self.classes_list = []

    def __call__(self,x:torch.Tensor)->torch.Tensor:
        '''
            Forward method for the model wraped
            
            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from.
                
            Returns
            -------
            model_prediction : torch.Tensor
                The model prediction.
        '''
        return self.forward(x)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
            Forward method for the model wraped
            
            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from.
                
            Returns
            -------
            model_prediction : torch.Tensor
                The model prediction.
        '''
        return self.model(x)

    def addFeatures(self,x:torch.Tensor):
        '''
            Method to add features of x estracted from the model to features of the OODTest object. If the features are
            not initialized, the method will initialize them with the features of x. If not, it will add them.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from. Shape must be (B,...), where B is the batch size and 
                ... is the shape of the input accepted by the model.
            
            Returns
            -------
            None
        '''
        feats = self.model.features(x).squeeze()
        classes = self.forward(x).argmax(1).detach()
        self.feats_list.append(feats)
        self.classes_list.append(classes)

    def finalizeFeatures(self):
        '''
            Method to finalize the features and classes of the OODTest object by concatenating all the collected features and classes.

            Parameters
            ----------
            None
            
            Returns
            -------
            None
        '''
        if self.feats_list:
            self.feats = torch.cat(self.feats_list, dim=0)
            self.classes = torch.cat(self.classes_list, dim=0)
            self.feats_list = []
            self.classes_list = []

    def restartFeatures(self):
        '''
            Method to restart the features and classes of the OODTest object to None.

            Parameters
            ----------
            None
            
            Returns
            -------
            None
        '''
        self.feats = None
        self.classes = None
        self.feats_list = []
        self.classes_list = []

    def features(self,x:torch.Tensor)->torch.Tensor:
        '''
            Method to extract features of x from the model.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from. Shape must be (B,...), where B is the batch size and 
                ... is the shape of the input accepted by the model.
            
            Returns
            -------
            features : torch.Tensor
                The extracted features. Shape must be (B,F), where B is the batch size and F is the number of features.
        '''
        return self.model.features(x)

    def test(self,x:torch.Tensor,intraclass:bool=True)->pd.DataFrame:
        """
        Method to test if the input x is OOD.

        The algorithm will use the following steps:
        1. Calculate the class and features of the input x.
        2. Consider the subset of the validation features:
            1. If intraclass is True, consider the subset of the same class as x.
            2. Else, consider the whole validation set.
        3. If n_features given, process the features of the input x and the validation features subset:
            1. Obtain the features ordered in the validation features subset and choose the main n_features.
            The presence is calculated as the sum of features in the validation subset:
                $$
                presence(i) = \sum_{j=1}^{#{ValSet}} f(x_j)_i
                $$
            2. The less present features are changed to 0.
        4. Obtain the distances between the input x and the validation features subset with the distance function
            provided and choose the K nearest neighbors, $V = \{v_1,...,v_K\}$.
        5. For each of the K nearest neighbors, calculate the distance between this v_i and the rest of the validation
            features subset, $i \neq j$ and choose other its K nearest neighbors, $V_i = \{v_{i1},...,v_{iK}\}$.
        6. Calculate the following matrix:

            | d(v,v_1) | d(v,v_2) | ... | d(v,v_K) |
            | d(v_1,v^1_1) | d(v_1,v^1_2) | ... | d(v_1,v^1_K) |
            | ... | ... | ... | ... |
            | d(v_K,v^1_1) | d(v_K,v^1_2) | ... | d(v_K,v^1_K) |

            where v is the features of the input x, v_i is the validation features of the $i$-th nearest neighbor
            and v^i_j is the validation features of the $j$-th nearest neighbor of the $i$-th nearest neighbor of x.
        7. For each K-nearest neighbor, perform the Bayes test between the distance between x and the validation subset (first row
            of the matrix) and the distance between x_i and the rest of the validation subset (i-th row of the matrix).
            The Bayes test is performed with the implementation of the article :cite:`benavoli2017time`. The Bayes test needs a Region of Practical Equivalence (ROPE) and we set it to the 75% quantile of the differences between the distances of the validation subset.


        Parameters
        ----------
        x : torch.Tensor
            The input tensor to test. Shape must be (B,...), where B is the batch size and
            ... is the shape of the input accepted by the model.
        n_features : int, optional
            The number of features to use for the test. If not provided, all features will be used.
            The features used will be the ones with the most presence in the features set considered.
            Default is None.
        intraclass : bool, optional
            If True, the test will be performed on the subset of the validation features that belong to the same class
            as x is predicted to belong to by the model. If False, the test will be performed on the whole validation set.
            Default is True.

        Returns
        -------
        ood_scores : pd.DataFrame
            A DataFrame with the following columns:
            - "d(x,VAL)>d(x_i,VAL)": Bayes test probability that distance between x and the validation subset is greater
                than the distance between x_i and the rest of the validation subset.
            - "d(x,VAL)~d(x_i,VAL)": Bayes test probability that distance between x and the validation subset is equivalent
                to the distance between x_i and the rest of the validation subset.
            - "d(x,VAL)<d(x_i,VAL)": Bayes test probability that distance between x and the validation subset is less
                than the distance between x_i and the rest of the validation subset.
            The Datafrane will have as many rows as the number of K-nearest neighbors considered.

        """

        x_features  = self.features(x.unsqueeze(0)).squeeze(0).detach().flatten()
        x_class     = torch.argmax(self.model(x.unsqueeze(0)).squeeze(0),dim=0)

        if intraclass:
            feat_subset = self.feats[self.classes == x_class]
        else:
            feat_subset = self.feats

        sum_abs_features = torch.sum(torch.abs(feat_subset),dim=0)

        
        Quantil = torch.quantile(sum_abs_features,self.quantile).item()

        n_features = torch.sum(sum_abs_features >= Quantil).item()

        # Guardar n_features en un archivo de logs llamado n_features.log
        # with open(f"XAI_features/features_NN{self.k_neighbors}.log","a") as f:
        #    f.write(f"len()={len(sum_abs_features)} min()={torch.min(sum_abs_features).item()} max()={torch.max(sum_abs_features)} Selected={n_features}\n")
        least_present_idx = torch.argsort(sum_abs_features,descending=True)[n_features:]

        feat_subset[:,least_present_idx] = 0

        x_features[least_present_idx] = 0

        x_distances = self.distance(x_features,feat_subset)
        
        if self.k_neighbors == -1:
            sorted_x_distances_idx = torch.argsort(x_distances)
        else:
            sorted_x_distances_idx = torch.argsort(x_distances)[:self.k_neighbors]

        distances_knns = torch.zeros(
            (len(sorted_x_distances_idx), len(feat_subset)), device=feat_subset.device
        )
        for i in range(len(sorted_x_distances_idx)):
            distances_knns[i] = self.distance(
                feat_subset[sorted_x_distances_idx[i]], feat_subset
            )

        df_p_values = pd.DataFrame(columns=["p_value"],index= range(len(sorted_x_distances_idx)))
        distances_top_ks = torch.sort(distances_knns, dim=0).values[0 : len(sorted_x_distances_idx)]

        x_distances = x_distances[sorted_x_distances_idx][:len(sorted_x_distances_idx)].detach().cpu().numpy()

        for i in range(len(sorted_x_distances_idx[:self.k_NNs])):
            distances_i = distances_top_ks[:,i].detach().cpu().numpy()
            # rope = torch.quantile(distances_i, self.quantile).item() - torch.quantile(distances_i, 1-self.quantile).item()

            if self.whole_test:
                if np.sum(np.abs(x_distances - distances_i)) > 0:
                    bayes_test = stats.wilcoxon(
                        x_distances, 
                        distances_i,
                        alternative="greater",
                        nan_policy="omit",
                    )
                else:
                    # Devuelve un bayes test ficticio para que no de error
                    # con un p_valor de 1
                    
                    bayes_test = stats.wilcoxon(
                        x_distances, 
                        distances_i+1,
                        alternative="greater",
                        nan_policy="omit",
                    )
                    
            else:
                bayes_test = stats.wilcoxon(
                    x_distances-distances_i,
                    alternative="greater",
                    nan_policy="omit",
                )

            df_p_values.iloc[i] = bayes_test.pvalue

        return df_p_values
