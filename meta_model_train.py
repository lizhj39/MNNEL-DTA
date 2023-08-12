import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from meta_model import TransformerRegressor
from base_models.NHGNN_DTA.build_vocab import WordVocab
import torch

dataset_name = "kiba"
pretrain_dataset = None

atom_addition = ["Gd", "Co", "Al", "Rb", "Mg", "He", "Sn", "Zr", "Li", "Ga", "Na", "In", "Ti", "Ag", "V", "Lu", "Hg", "Au", "Xe", "Tl", "La", "Ra", "Ga", "Y", "2H", "H+"]
drug_vocab = WordVocab.load_vocab('base_models/Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('base_models/Vocab/protein_vocab.pkl')
for atom in atom_addition:
    drug_vocab.stoi[atom] = 44

model = TransformerRegressor(save_path=f"saved_models/final_model_{dataset_name}_ab--11.pt",
                             drug_vocab=drug_vocab, target_vocab=target_vocab, dataset_name=dataset_name,
                             num_layers=2)

model.load_state_dict(torch.load(f"saved_models/final_model_{dataset_name}_ab--1.pt"))

model.fit(train_sets=[model.model_Mgraph.train_set, model.model_TFusion.train_set, model.model_NHGNN.train_set],
          test_sets=[model.model_Mgraph.test_set, model.model_TFusion.test_set, model.model_NHGNN.test_set],
          graph_data=[True, False, True], early_stop_epoch=200, lr=7e-4, save_model=True, weight_decay=1e-5)


# test
# model.load_state_dict(torch.load("saved_models/final_model_kiba_m2.pt"))
# model = model.cuda()
# mse, ci = model.val(
#     test_sets=[model.model_Mgraph.test_set, model.model_TFusion.test_set, model.model_NHGNN.test_set],
#     graph_data=[True, False, True]
# )
# print(f"mse={mse}, ci={ci}")
print("okk")

