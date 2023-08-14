import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from meta_model import TransformerRegressor
from base_models.NHGNN_DTA.build_vocab import WordVocab
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as gLoader
from tqdm import tqdm

atom_addition = ["Gd", "Co", "Al", "Rb", "Mg", "He", "Sn", "Zr", "Li", "Ga", "Na", "In", "Ti", "Ag", "V", "Lu", "Hg", "Au", "Xe", "Tl", "La", "Ra", "Ga", "Y", "2H", "H+"]

class DrugScreening():
    def __init__(self, dataset_name, pred_dataset):
        # dataset_name: the dataset that the model trained, eg. kiba
        # pred_dataset: the dataset that are going to predict, eg. AD
        self.drug_vocab = WordVocab.load_vocab('base_models/Vocab/smiles_vocab.pkl')
        self.target_vocab = WordVocab.load_vocab('base_models/Vocab/protein_vocab.pkl')
        a = self.drug_vocab.stoi
        for atom in atom_addition:
            self.drug_vocab.stoi[atom] = 44

        self.model = TransformerRegressor(save_path=f"saved_models/final_model_{dataset_name}.pt",
                                          drug_vocab=self.drug_vocab, target_vocab=self.target_vocab,
                                          dataset_name=dataset_name, auto_dataset=False)
        self.model.load_state_dict(torch.load(f"saved_models/final_model_{dataset_name}.pt"))
        # mse, ci = self.model.val(
        #     test_sets=[self.model.model_Mgraph.test_set, self.model.model_TFusion.test_set, self.model.model_NHGNN.test_set],
        #     graph_data=[True, False, True]
        # )
        # print(f"mse={mse}, ci={ci}")

        self.dataset_name = dataset_name
        self.pred_data_path = f"data_input/{pred_dataset}/raw/data.csv"
        self.pred_root_path = f"data_input/{pred_dataset}"
        self.pred_dataset = pred_dataset
        self.pred_dataset_list = self.create_dataset()

    def create_dataset(self):
        Mgraph_set = self.model.model_Mgraph.get_dataset("data_input", self.pred_dataset)

        _, target_emb, target_len, smiles_emb, smiles_len = self.model.model_TFusion.preparation(
            f"data_input/{self.pred_dataset}/raw/data.csv", self.drug_vocab, self.target_vocab,
            tar_len=2600, sm_len=536)
        TFusion_set = self.model.model_TFusion.get_dataset(
            f"data_input/{self.pred_dataset}/raw/data.csv", smiles_emb, target_emb, smiles_len, target_len)
        del _, target_emb, target_len, smiles_emb, smiles_len

        smiles_graph, target_graph, self.model.model_NHGNN.target_seq, self.model.model_NHGNN.smiles_emb, \
        self.model.model_NHGNN.target_emb, self.model.model_NHGNN.smiles_len, self.model.model_NHGNN.target_len, \
        self.model.model_NHGNN.smiles_idx = self.model.model_NHGNN.preparation(
            f"data_input/{self.pred_dataset}/raw/data.csv",
            self.drug_vocab, self.target_vocab, tar_len=2600, sm_len=536)
        NHGNN_set = self.model.model_NHGNN.get_dataset(
            f"data_input/{self.pred_dataset}/raw/data.csv",
            self.model.model_NHGNN.smiles_idx, smiles_graph, target_graph,
            self.model.model_NHGNN.smiles_len, self.model.model_NHGNN.target_len)

        return [Mgraph_set, TFusion_set, NHGNN_set]

    def predict(self):
        graph_data = [True, False, True]
        dataloaders_test = [gLoader(test_set, batch_size=128, shuffle=False) if is_graph
                            else DataLoader(test_set, batch_size=128, shuffle=False)
                            for (test_set, is_graph) in zip(self.pred_dataset_list, graph_data)]
        model = self.model.cuda()
        model.eval()
        total_pred = torch.Tensor()
        ii = 1
        with torch.no_grad():
            for datas in tqdm(zip(*dataloaders_test), ncols=80):
                for i, data in enumerate(datas):
                    try:
                        datas[i] = data.cuda()
                    except:
                        pass
                out = model(datas[0], datas[1], datas[2])
                total_pred = torch.cat((total_pred, out.cpu()), 0)
        np.savetxt(f"FDA_predict_{self.dataset_name}.csv", total_pred.numpy(), delimiter=",", fmt="%.3f")


def main():
    dr = DrugScreening("S1R", "AD")
    dr.predict()


if __name__ == "__main__":
    main()
