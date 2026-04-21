import torch
from typing import Any
from tqdm import tqdm
from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm
import pandas as pd
import os

from STOODX.featureStractor import FeatureStractor
from STOODX.STOODX import STOODX

class STOODXPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(STOODXPostprocessor, self).__init__(config)
        self.K = self.config['K']
        self.NNK = self.config.get('NN_K', 5)
        self.distance = self.config['distance']
        self.feature_name = self.config['feature_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intraclass = self.config['intraclass']
        self.quantile = self.config['quantil']
        self.atribut = self.config['atribut']
        self.partition = self.config.get('partition', 'train')
        self.whole_test = self.config.get('whole_test', False)
        self.id_name = self.config.get("id_name")
        self.model_name = self.config.get("model_name")
        self.APS_mode = False
        self.oodTest = None

        def p_value_statistic(df: pd.DataFrame) -> float:
            return df["p_value"].mean()

        self.p_value_statistic = config.get("p_value_statistic", p_value_statistic)

    def deleteIrrelevantFeatures(self, q: int = -1):
        if q != -1:
            new_feats = []
            classes_list = []

            for classes in torch.unique(self.oodTest.classes):
                feats = self.oodTest.feats[self.oodTest.classes == classes]
                # Shuffle feats
                feats = feats[torch.randperm(len(feats))][:q]
                new_feats.append(feats)
                classes_list.append(torch.tensor([classes for i in range(len(feats))]))

            self.oodTest.feats = torch.cat(new_feats).to(self.device)
            self.oodTest.classes = torch.cat(classes_list).to(self.device)

    def setup(self, net: torch.nn.Module, id_loader_dict, ood_loader_dict):
        # Create the feature extractor
        net = net.to(self.device)
        feature_extractor = FeatureStractor(model=net, device=self.device, feature_name=self.feature_name, atribut=self.atribut).to(self.device)
        self.oodTest = STOODX(
            model=feature_extractor, distance=self.distance, quantile=self.quantile,
            whole_test=self.whole_test,k_neighbors=self.K,k_NNs=self.NNK
        )

        if os.path.exists(
            f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}.pth"
        ) and os.path.exists(
            f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}_classes.pth"
        ):

            self.oodTest.feats = torch.load(
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}.pth",
                weights_only=True,
                map_location=self.device,
            )
            self.oodTest.classes = torch.load(
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}_classes.pth",
                weights_only=True,
                map_location=self.device,
            )
            print("Features loaded from cache")
        elif self.oodTest.feats is None:
            if len(self.partition.split("_")) == 1:
                loader_dict = id_loader_dict[self.partition]
            else:
                # Une los loaders
                combined_dataset = [id_loader_dict[self.partition.split("_")[0]].dataset,
                                    id_loader_dict[self.partition.split("_")[1]].dataset]
                combined_dataset = torch.utils.data.ConcatDataset(combined_dataset)
                loader_dict = torch.utils.data.DataLoader(
                    combined_dataset, batch_size=32, shuffle=False
                )
            for batch in tqdm(loader_dict, desc="Adding features..."):
                data = batch['data'].to(self.device)

                self.oodTest.addFeatures(data)

            self.oodTest.finalizeFeatures()

            os.makedirs(f"./utils/features/",exist_ok=True)
            torch.save(
                self.oodTest.feats,
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}.pth",
            )
            torch.save(
                self.oodTest.classes,
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}_classes.pth",
            )

    def postprocess(self, net: torch.nn.Module, data: Any):
        pred_list = []
        confs = []

        for batch in tqdm(data, desc="Calculating OOD conf..."):
            data = batch['data'].to(self.device)
            pred = self.oodTest(data)
            pred_list.append(pred)

            for element in data:
                df_wilcoxon = self.oodTest.test(element, self.intraclass)
                wilcoxon_stat = self.p_value_statistic(df_wilcoxon)
                confs.append(wilcoxon_stat)

        preds = torch.cat(pred_list).argmax(dim=1).cpu().numpy().astype(int)
        confs = torch.tensor(confs).cpu().numpy()
        return preds, confs

    def inference(self, net, data_loader, progress=True):
        label_list = []
        preds, confs = self.postprocess(net, data_loader)

        for batch in tqdm(data_loader, desc="Calculating inference...",
                          disable=not progress or not comm.is_main_process()):
            label = batch["label"].to(self.device)
            label_list.append(label)

        labels = torch.cat(label_list).cpu().numpy().astype(int)
        return preds, confs, labels
