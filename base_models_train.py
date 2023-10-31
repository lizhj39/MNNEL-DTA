import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from base_models.model_Mgragh import MGraphDTA
from base_models.model_TFusion import TFusionDTA
from base_models.model_NHGNN import NHGNN_DTA
from base_models.NHGNN_DTA.build_vocab import WordVocab
from base_models.model_Mgragh_modify import MGraphDTA as MGM
from base_models.model_TFusion_CNN import TFusionDTA_CNN
from base_models.model_Mgragh_LSTM import MGraphDTA_LSTM

if __name__ == "__main__":
    dataset_name = "davis"
    pretrain_dataset = dataset_name
    # pretrain_dataset = None
    test_ratio = 0.15
    seed = 42
    drug_vocab = WordVocab.load_vocab('base_models/Vocab/smiles_vocab.pkl')
    target_vocab = WordVocab.load_vocab('base_models/Vocab/protein_vocab.pkl')
    atom_addition = ["Gd", "Co", "Al", "Rb", "Mg", "He", "Sn", "Zr", "Li", "Ga", "Na", "In", "Ti", "Ag", "V", "Lu",
                     "Hg", "Au", "Xe", "Tl", "La", "Ra", "Ga", "Y", "2H", "H+"]
    for atom in atom_addition:
        drug_vocab.stoi[atom] = 44

    # # Mgraph
    model_Mgraph = MGraphDTA(data_input_path="data_input", seed=seed, test_ratio=test_ratio, dataset_name=dataset_name,
                             model_dir="saved_models", model_name=f"Mgraph_{dataset_name}_test2").cuda()
    # if pretrain_dataset is not None:
    #     model_Mgraph.load_state_dict(torch.load(f"saved_models/Mgraph_{pretrain_dataset}_test1.pt"), strict=False)
    #     print(model_Mgraph.val(model_Mgraph.test_set))

    model_Mgraph.fit(model_Mgraph.train_set, model_Mgraph.test_set, early_stop_epoch=300, freeze_front=True)

    # model_Mgraph = MGraphDTA_LSTM(drug_vocab=drug_vocab, target_vocab=target_vocab, data_input_path="data_input",
    #                               seed=seed, test_ratio=test_ratio, dataset_name=dataset_name,
    #                          model_dir="saved_models", model_name=f"Mgraph_LSTM_{dataset_name}_test66666")
    #
    # model_Mgraph.fit(model_Mgraph.train_set, model_Mgraph.test_set, early_stop_epoch=30000, freeze_front=False)


    # Tokenize Fusion
    model_TFusion = TFusionDTA(data_path=f"data_input/{dataset_name}/raw/data.csv",
                               saved_path=f"saved_models/TFusion_{dataset_name}_test2.pt",
                               drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed, test_ratio=test_ratio).cuda()
    # if pretrain_dataset is not None:
    #     model_TFusion.load_state_dict(torch.load(f"saved_models/TFusion_{pretrain_dataset}_test1.pt"))
    #     print(model_TFusion.val(model_TFusion.test_set))
    model_TFusion.fit(model_TFusion.train_set, model_TFusion.test_set, early_stop_epochs=80)

    # model_TFusion = TFusionDTA_CNN(data_path=f"data_input/{dataset_name}/raw/data.csv",
    #                                saved_path=f"saved_models/TFusion_CNN_{dataset_name}.pt",
    #                                drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed,
    #                                test_ratio=test_ratio).cuda()
    # # if pretrain_dataset is not None:
    # #     model_TFusion.load_state_dict(torch.load(f"saved_models/TFusion_{pretrain_dataset}_test1.pt"))
    # #     print(model_TFusion.val(model_TFusion.test_set))
    # model_TFusion.fit(model_TFusion.train_set, model_TFusion.test_set, early_stop_epochs=80)

    # # NHGNN
    model_NHGNN = NHGNN_DTA(pretrain_path=f"saved_models/TFusion_{dataset_name}_test1.pt",
                            data_path=f"data_input/{dataset_name}/raw/data.csv",
                            saved_path=f"saved_models/NHGNN_{dataset_name}_test1.pkl",
                            drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed, test_ratio=test_ratio).cuda()
    # if pretrain_dataset is not None:
    #     model_NHGNN.load_state_dict(torch.load(f"saved_models/NHGNN_{pretrain_dataset}_test1.pkl")['model'], strict=False)
    #     print(model_NHGNN.val(model_NHGNN.test_set))
    model_NHGNN.fit(model_NHGNN.train_set, model_NHGNN.test_set, early_stop_epochs=60)
    #
    # Mgraph modify
    # model_Mgraph_modify = MGM(data_input_path="data_input", data_path=f"data_input/{dataset_name}/raw/data.csv",
    #                           drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed, test_ratio=test_ratio,
    #                           dataset_name=dataset_name,
    #                           model_dir="saved_models", model_name=f"Mgraph_modify_{dataset_name}")
    # #
    # model_Mgraph_modify.fit(model_Mgraph_modify.train_set, model_Mgraph_modify.test_set, early_stop_epoch=100)
