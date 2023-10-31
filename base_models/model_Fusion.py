import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from FusionDTA.src.models.layers import LinkAttention
import torch
import torch.nn as nn
import numpy as np
from FusionDTA.src.utils import pack_sequences, pack_pre_sequences, unpack_sequences, split_text, load_protvec, graph_pad, DrugTargetDataset, collate, AminoAcid
from FusionDTA.src.getdata import getdata_from_csv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from tqdm import tqdm
from lifelines.utils import concordance_index as ci
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from FusionDTA.src.build_vocab import WordVocab
import random

device_ids = [0, 1, 2, 3]


class FusionDTA(nn.Module):
    def __init__(self, embedding_dim=1280, rnn_dim=128, hidden_dim=256, graph_dim=256, dropout_rate=0.2, alpha=0.2,
                 n_heads=8, graph_input_dim=78, rnn_layers=2, n_attentions=1, attn_type='dotproduct',
                 vocab=26, smile_vocab=45, is_pretrain=True, is_drug_pretrain=False, n_extend=1,
                 dataset_name="kiba", seed=42, test_ratio=0.15,
                 saved_path=f'../saved_models/Fusion_all_train_kiba.pkl'):
        super(FusionDTA, self).__init__()
        # data
        self.dataset_name = dataset_name
        self.dataset = self.get_dataset(dataset_name)
        self.train_set, self.test_set = self.split_dataset(self.dataset, seed, test_ratio)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.save_path = saved_path

        # drugs
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.n_attentions = n_attentions
        self.n_heads = n_heads
        self.graph_head_fc1 = nn.Linear(graph_dim * n_heads, graph_dim)
        self.graph_head_fc2 = nn.Linear(graph_dim * n_heads, graph_dim)
        self.graph_out_fc = nn.Linear(graph_dim, hidden_dim)
        self.out_attentions1 = LinkAttention(hidden_dim, n_heads)

        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=smile_vocab)
        self.rnn_layers = 2
        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, rnn_dim)
        self.smiles_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                                  , bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.smiles_out_fc = nn.Linear(rnn_dim * 2, rnn_dim)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.is_pretrain = is_pretrain
        if not is_pretrain:
            self.vocab = vocab
            self.embed = nn.Embedding(vocab + 1, embedding_dim, padding_idx=vocab)
        self.rnn_layers = 2
        self.is_bidirectional = True
        self.sentence_input_fc = nn.Linear(embedding_dim, rnn_dim)
        self.encode_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                                  , bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.rnn_out_fc = nn.Linear(rnn_dim * 2, rnn_dim)
        self.sentence_head_fc = nn.Linear(rnn_dim * n_heads, rnn_dim)
        self.sentence_out_fc = nn.Linear(2 * rnn_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(rnn_dim * 2)

    def get_dataset(self, dataset_name):
        all_drug, all_protein, all_affinity, pid = getdata_from_csv(
            f"../data_input/{dataset_name}/raw/data.csv", dataset_name=dataset_name)
        all_affinity = torch.from_numpy(np.array(all_affinity)).float()
        drug_vocab = WordVocab.load_vocab('FusionDTA/src/smiles_vocab.pkl')
        dataset = DrugTargetDataset(all_drug, all_protein, all_affinity, pid, is_target_pretrain=True,
                                    self_link=False, npz_path="FusionDTA", dataset=dataset_name,
                                    drug_vocab=drug_vocab)
        return dataset

    def split_dataset(self, dataset, seed, test_ratio):
        test_size = round(len(dataset) * test_ratio)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset) - test_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return train_dataset, test_dataset

    def forward(self, protein, smiles, output_vector=False):
        # drugs
        batchsize = len(protein)

        smiles_lengths = np.array([len(x) for x in smiles])
        temp = (torch.zeros(batchsize, max(smiles_lengths)) * 63).long()
        for i in range(batchsize):
            temp[i, :len(smiles[i])] = smiles[i]
        smiles = temp.cuda()
        smiles = self.smiles_embed(smiles)
        smiles = self.smiles_input_fc(smiles)
        smiles_out, _ = self.smiles_rnn(smiles)
        smiles_mask = self.generate_masks(smiles_out, smiles_lengths, self.n_heads)
        smiles_cat, smile_attn = self.out_attentions3(smiles_out, smiles_mask)

        # proteins
        if self.is_pretrain:
            protein_lengths = np.array([x.shape[0] for x in protein])
            protein = graph_pad(protein, max(protein_lengths))

        h = self.sentence_input_fc(protein)
        sentence_out, _ = self.encode_rnn(h)
        sent_mask = self.generate_masks(sentence_out, protein_lengths, self.n_heads)
        sent_cat, sent_attn = self.out_attentions2(sentence_out, sent_mask)

        # drugs and proteins
        h = torch.cat((smiles_out, sentence_out), dim=1)
        out_masks = self.generate_out_masks(smiles_lengths, smiles_mask, sent_mask, protein_lengths, self.n_heads)
        out_cat, out_attn = self.out_attentions(h, out_masks)

        out = torch.cat([smiles_cat, sent_cat, out_cat], dim=1) # (batch_size, d_model)
        if output_vector:
            with torch.no_grad():
                sm_vec = smiles_cat
                pro_vec = sent_cat
                sp_vec = out_cat

        d_block = self.dropout(self.relu(self.out_fc1(out)))
        o2 = self.out_fc2(d_block)
        if output_vector:
            return sm_vec, pro_vec, sp_vec, o2

        out = self.dropout(self.relu(o2))
        out = self.out_fc3(out).squeeze()

        return d_block, out

    def forward_vector(self, test_set, use_cuda=True, batch_size=256):
        dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
        sm_v = torch.Tensor()
        pro_v = torch.Tensor()
        sp_v = torch.Tensor()
        o2_v = torch.Tensor()
        for protein, smiles, affinity in tqdm(dataloader_test, ncols=80):
            if use_cuda:
                protein = [p.cuda() for p in protein]
                smiles = [s.cuda() for s in smiles]

            with torch.no_grad():
                sm_vec, pro_vec, sp_vec, o2_vec = self.forward(protein, smiles, output_vector=True)
                sm_v = torch.cat((sm_v, sm_vec.cpu()), 0)
                pro_v = torch.cat((pro_v, pro_vec.cpu()), 0)
                sp_v = torch.cat((sp_v, sp_vec.cpu()), 0)
                o2_v = torch.cat((o2_v, o2_vec.cpu()), 0)
        return sm_v, pro_v, sp_v, o2_v
                # TODO ------------------

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        for e_id, drug_len in enumerate(adj_sizes):
            out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def generate_out_masks(self, drug_sizes, adj, masks, source_lengths, n_heads):
        adj_size = adj.shape[2]
        sen_size = masks.shape[2]
        maxlen = adj_size + sen_size
        out = torch.ones(adj.shape[0], maxlen)
        for e_id in range(len(source_lengths)):
            src_len = source_lengths[e_id]
            drug_len = drug_sizes[e_id]
            out[e_id, drug_len: adj_size] = 0
            out[e_id, adj_size + src_len:] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def fit(self, train_set, test_set, use_cuda=True, lr=1e-3, weight_decay=0, epochs=1000, early_stop_epochs=60, batch_size=128):
        model = self
        if use_cuda:
            model.cuda()

        # datalodaer
        dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
        # dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

        # optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        print('--- GAT model --- ')

        best_ci = 0
        best_mse = 100000
        best_epoch = -1
        for epoch in range(epochs):
            # train
            model.train()
            print(f"epoch [{epoch + 1} / {epochs}]")
            for protein, smiles, affinity in tqdm(dataloader_train, ncols=80):

                if use_cuda:
                    protein = [p.cuda() for p in protein]
                    smiles = [s.cuda() for s in smiles]
                    affinity = affinity.cuda()

                _, out = model(protein, smiles)
                loss = criterion(out, affinity)

                loss.backward()
                optim.step()
                optim.zero_grad()

            # val
            all_mse, all_ci = self.val(test_set)

            if all_mse < best_mse:
                best_ci = all_ci
                best_mse = all_mse
                best_epoch = epoch
                model.cpu()
                save_dict = {'model': model.state_dict(), 'optim': optim.state_dict(), 'ci': best_ci}
                torch.save(save_dict, self.save_path)
                if use_cuda:
                    model.cuda()
            else:
                if epoch - best_epoch > early_stop_epochs:
                    break
            print(
                f"total_mse={all_mse}, total_ci={all_ci}, best mse={best_mse}, ci={best_ci}, best_epoch={best_epoch + 1}")

    def val(self, test_set, batch_size=128, use_cuda=True):
        dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                      collate_fn=collate)
        model = self
        model.eval()
        total_pred = torch.Tensor()
        total_label = torch.Tensor()
        with torch.no_grad():
            for protein, smiles, affinity in tqdm(dataloader_test, ncols=80):
                if use_cuda:
                    protein = [p.cuda() for p in protein]
                    smiles = [s.cuda() for s in smiles]
                    affinity = affinity.cuda()

                _, out = model(protein, smiles)

                total_pred = torch.cat((total_pred, out.cpu()), 0)
                total_label = torch.cat((total_label, affinity.cpu()), 0)

        all_ci = ci(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        all_mse = mean_squared_error(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        return all_mse, all_ci


def main():
    dataset_name = "kiba"
    model_Fusion = FusionDTA(seed=42, test_ratio=0.15, dataset_name=dataset_name,
                             saved_path="../saved_models/Fusion_all_train_kiba___2.pkl")
    model_Fusion.load_state_dict(torch.load("../saved_models/Fusion_all_train_kiba___.pkl")['model'])
    model_Fusion.cuda()
    mse, ci = model_Fusion.val(model_Fusion.test_set)
    print(f"best mse={mse}, ci={ci}")
    model_Fusion.fit(model_Fusion.train_set, model_Fusion.test_set, early_stop_epochs=60)
    # sm_v, pro_v, sp_v, o2_v = model_Fusion.forward_vector(model_Fusion.test_set)


if __name__ == "__main__":
    pass
    # main()

    # # 3 folds
    # kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    # for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
    #     if fold >= 2: break
    #     model_Fusion = FusionDTA(model_name=f"Fusion_{fold + 1}_{dataset_name}")
    #     train_set = Subset(train_dataset, train_ids)
    #     test_set = Subset(train_dataset, valid_ids)
    #     model_Fusion.fit(train_set, test_set, early_stop_epochs=60)

# print("ok")

