import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from meta_model import TransformerRegressor
# from meta_model_pure import TransformerRegressor
from base_models.NHGNN_DTA.build_vocab import WordVocab
import torch

dataset_name = "davis"
pretrain_dataset = None

atom_addition = ["Gd", "Co", "Al", "Rb", "Mg", "He", "Sn", "Zr", "Li", "Ga", "Na", "In", "Ti", "Ag", "V", "Lu", "Hg", "Au", "Xe", "Tl", "La", "Ra", "Ga", "Y", "2H", "H+"]
drug_vocab = WordVocab.load_vocab('base_models/Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('base_models/Vocab/protein_vocab.pkl')
for atom in atom_addition:
    drug_vocab.stoi[atom] = 44

model = TransformerRegressor(save_path=f"saved_models/final_model_{dataset_name}_LSTMCNN2.pt",
                             drug_vocab=drug_vocab, target_vocab=target_vocab, dataset_name=dataset_name,
                             num_layers=3)


# model.fit(train_sets=[model.model_Mgraph.train_set, model.model_TFusion.train_set, model.model_NHGNN.train_set],
#           test_sets=[model.model_Mgraph.test_set, model.model_TFusion.test_set, model.model_NHGNN.test_set],
#           graph_data=[True, False, True], early_stop_epoch=200, lr=7e-4, save_model=True, weight_decay=1e-5)


# test
# model = model.cuda()
# model.load_state_dict(torch.load(f"saved_models/final_model_{dataset_name}_LSTMCNN2.pt"))
# mse, ci, attn_weights = model.val(
#     test_sets=[model.model_Mgraph.test_set, model.model_TFusion.test_set, model.model_NHGNN.test_set],
#     graph_data=[True, False, True], get_attn=True
# )
# print(f"mse={mse}, ci={ci}")
#
# rollout = torch.Tensor()
# attn_weights = [layer.detach().cpu() for layer in attn_weights]
# for ii, layer in enumerate(attn_weights):
#     if ii == 0:
#         rollout = torch.eye(4) + layer
#     else:
#         rollout = torch.matmul(rollout, (torch.eye(4) + layer))
# rollout = torch.softmax(rollout, dim=2)
# av_rollout = torch.mean(rollout, dim=0)
# print(av_rollout)

# attn_weights = []
# for layer in model.transformer_encoder.layers:
#     attn_weights.append(layer.self_attn.attn_output_weights)


print("okk")
#
