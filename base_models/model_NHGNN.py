import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lifelines.utils import concordance_index as ci
from sklearn.metrics import mean_squared_error
from .NHGNN_DTA.model import GINConvNet, LinkAttention
from .NHGNN_DTA.build_vocab import WordVocab
from .NHGNN_DTA.NHGNN_dataset import DTADataset
import torch_geometric
from torch_geometric.loader import DataLoader
import pandas as pd
import rdkit.Chem as Chem
import networkx as nx

#############################
atom_dict = {5: 'C',
             6: 'C',
             9: 'O',
             12: 'N',
             15: 'N',
             21: 'F',
             23: 'S',
             25: 'Cl',
             26: 'S',
             28: 'O',
             34: 'Br',
             36: 'P',
             37: 'I',
             39: 'Na',
             40: 'B',
             41: 'Si',
             42: 'Se',
             44: 'K',
             }


class NHGNN_DTA(nn.Module):
    def __init__(self, pretrain_path, data_path, saved_path, drug_vocab, target_vocab, auto_dataset: bool=True,
                 tar_len=2600, sm_len=536, embedding_dim=128, lstm_dim=64, hidden_dim=128, dropout_rate=0.1,
                 alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26, seed=42, test_ratio=0.15,
                 smile_vocab=45, theta=0.5):
        super(NHGNN_DTA, self).__init__()
        self.saved_path = saved_path
        self.data_path = data_path

        # model
        self.is_bidirectional = True
        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        self.hgin = GINConvNet()

        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=dropout_rate)
        # self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)

        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=self.is_bidirectional, dropout=dropout_rate)
        # self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

        # pretrained model
        if pretrain_path is not None:
            print('load_model...', pretrain_path)
            save_model = torch.load(pretrain_path)
            model_dict = self.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.load_state_dict(model_dict)

        # data
        if auto_dataset:
            print("NHGNN dataset preparation...")
            smiles_graph, target_graph, self.target_seq, self.smiles_emb, self.target_emb, \
            self.smiles_len, self.target_len, self.smiles_idx = self.preparation(
                data_path, drug_vocab, target_vocab, tar_len, sm_len)

            self.dataset = self.get_dataset(data_path, self.smiles_idx, smiles_graph, target_graph,
                                            self.smiles_len, self.target_len)
            self.train_set, self.test_set = self.split_dataset(self.dataset, seed, test_ratio)

    def preparation(self, data_path, drug_vocab, target_vocab, tar_len, sm_len):
        def target_to_graph(target_key, target_sequence, contact_dir):
            target_edge_index = []
            target_size = len(target_sequence)
            contact_file = os.path.join(contact_dir, target_key + '.npy')
            contact_map = np.load(contact_file)
            contact_map += np.matrix(np.eye(contact_map.shape[0]))
            index_row, index_col = np.where(contact_map > 0.8)
            for i, j in zip(index_row, index_col):
                target_edge_index.append([i, j])
            target_edge_index = np.array(target_edge_index)
            return target_size, target_edge_index

        def smiles_to_graph(smile):
            mol = Chem.MolFromSmiles(smile)
            c_size = mol.GetNumAtoms()

            edges = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            g = nx.Graph(edges).to_directed()
            edge_index = []
            mol_adj = np.zeros((c_size, c_size))
            for e1, e2 in g.edges:
                mol_adj[e1, e2] = 1
                # edge_index.append([e1, e2])
            mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
            index_row, index_col = np.where(mol_adj >= 0.5)
            for i, j in zip(index_row, index_col):
                edge_index.append([i, j])
            edge_index = np.array(edge_index)
            return c_size, edge_index

        df = pd.read_csv(data_path)
        smiles = set(df['compound_iso_smiles'])
        target = set(df['target_key'])

        target_seq = {}
        for i in range(len(df)):
            target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']
        target_graph = {}
        pconsc4_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), "pconsc4")
        for k in target_seq:
            seq = target_seq[k]
            _, graph = target_to_graph(k, seq, pconsc4_path)
            target_graph[seq] = graph
        smiles_graph = {}
        for sm in smiles:
            _, graph = smiles_to_graph(sm)
            smiles_graph[sm] = graph

        target_emb = {}
        target_len = {}
        for k in target_seq:
            seq = target_seq[k]
            content = []
            flag = 0
            for i in range(len(seq)):
                if flag >= len(seq):
                    break
                if (flag + 1 < len(seq)):
                    if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                        content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                        flag = flag + 2
                        continue
                content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
                flag = flag + 1

            if len(content) > tar_len-2:
                content = content[:tar_len-2]

            X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
            target_len[k] = len(content)
            if tar_len > len(X):
                padding = [target_vocab.pad_index] * (tar_len - len(X))
                X.extend(padding)
            target_emb[seq] = torch.tensor(X)

        smiles_idx = {}
        smiles_emb = {}
        smiles_len = {}

        for sm in smiles:
            content = []
            cut_graph = False
            flag = 0
            for i in range(len(sm)):
                if flag >= len(sm):
                    break
                if (flag + 1 < len(sm)):
                    if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                        content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                        flag = flag + 2
                        continue
                content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
                flag = flag + 1

            if len(content) > sm_len-2:
                content = content[:sm_len-2]
                cut_graph = True

            X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
            smiles_len[sm] = len(content)
            if sm_len > len(X):
                padding = [drug_vocab.pad_index] * (sm_len - len(X))
                X.extend(padding)

            smiles_emb[sm] = torch.tensor(X)

            if not smiles_idx.__contains__(sm):
                tem = []
                for i, c in enumerate(X):
                    if atom_dict.__contains__(c):
                        tem.append(i)
                smiles_idx[sm] = tem

                if cut_graph:
                    mask = np.any(smiles_graph[sm] >= len(tem), axis=1)
                    smiles_graph[sm] = smiles_graph[sm][~mask]

                if len(smiles_idx[sm]) != smiles_graph[sm][-1][0]+1:
                    print(sm)


        return smiles_graph, target_graph, target_seq, smiles_emb, target_emb, smiles_len, target_len, smiles_idx

    def get_dataset(self, data_path, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        print("processing NHGNN dataset...")
        root_path = os.path.dirname(os.path.dirname(data_path))
        data_set = DTADataset(root=root_path, path=data_path, smiles_idx=smiles_idx, smiles_graph=smiles_graph,
                              target_graph=target_graph, smiles_len=smiles_len, target_len=target_len)
        return data_set

    def split_dataset(self, dataset, seed, test_ratio):
        test_size = round(len(dataset) * test_ratio)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset) - test_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return train_dataset, test_dataset

    def forward(self, data, output_vector=False, tar_len=2600, seq_len=536):

        target_seq, smiles_emb, target_emb, smiles_len, target_len, smiles_idx = \
            self.target_seq, self.smiles_emb, self.target_emb, self.smiles_len, self.target_len, self.smiles_idx

        batchsize = len(data.sm)
        smiles = torch.zeros(batchsize, seq_len).cuda().long()
        protein = torch.zeros(batchsize, tar_len).cuda().long()

        for i in range(batchsize):
            sm = data.sm[i]
            seq_id = data.target[i]
            seq = target_seq[seq_id]
            smiles[i] = smiles_emb[sm]
            protein[i] = target_emb[seq]
        # smiles_lengths = [len(sm) for sm in data.sm]
        tar_len = [self.target_len[k] for k in data.target]

        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim

        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        # smiles = self.ln1(smiles)
        # del smiles

        protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim
        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        # protein = self.ln2(protein)

        # smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        # smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # B * lstm_dim*2
        #
        # protein_mask = self.generate_masks(protein, tar_len, self.n_heads)  # B * head * tar_len
        # protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)
        #
        # # drugs and proteins
        # out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        # out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+seq_len * (lstm_dim *2)
        # out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        # out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)
        # out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        # out = self.dropout(self.relu(self.out_fc2(out)))  # B *  hidden_dim*2
        # out = self.out_fc3(out).squeeze()

        # del smiles_out, protein_out

        sm_emb, pro_emb = smiles, protein
        idx = [smiles_idx[s] for s in data.sm]

        data.x = torch.cat([torch.cat([pro_emb[i, 1:tar_len[i] + 1], sm_emb[i, idx[i]], sm_emb[i, 0].unsqueeze(0)]).cpu() for i in range(len(data))], 0)

        if output_vector:
            NHGvec = self.hgin(data, output_vector=True)
            return NHGvec

        gout = self.hgin(data)
        return gout
        # return gout * self.theta + out.view(-1, 1) * (1 - self.theta)

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def fit(self, train_set, test_set, use_cuda=True, lr=1e-4, GIN_lr=5e-4, weight_decay=0, epochs=1000,
            early_stop_epochs=60, batch_size=128, save_model=True):

        # model
        model = self
        if use_cuda:
            model.cuda()

        # optimizer
        GIN_params = list(map(id, model.hgin.parameters()))
        base_params = filter(lambda p: id(p) not in GIN_params,
                             model.parameters())
        optimizer = torch.optim.Adam(
            [{'params': base_params},
             {'params': model.hgin.parameters(), 'lr': GIN_lr}], lr=lr, weight_decay=weight_decay)

        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=2e-5, last_epoch=-1)
        criterion = nn.MSELoss()
        dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        best_ci = 0
        best_mse = 100000
        best_epoch = -1
        torch.cuda.empty_cache()
        for epoch in range(epochs):
            model.train()
            print(f"epoch [{epoch + 1} / {epochs}]")
            for data in tqdm(dataloader_train, ncols=80):
                if use_cuda:
                    data = data.cuda()
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out.float(), data.y.view(-1, 1).float().cuda()).float()
                loss.backward()
                optimizer.step()
            schedule.step()

            # val
            all_mse, all_ci = self.val(test_set)

            if all_mse < best_mse:
                best_ci = all_ci
                best_mse = all_mse
                best_epoch = epoch
                model.cpu()
                if save_model:
                    save_dict = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'ci': best_ci}
                    torch.save(save_dict, self.saved_path)
                if use_cuda:
                    model.cuda()
            else:
                if epoch - best_epoch > early_stop_epochs:
                    break
            print(
                f"total_mse={all_mse}, total_ci={all_ci}, best mse={best_mse}, ci={best_ci}, best_epoch={best_epoch + 1}")

    def val(self, test_set, batch_size=128, use_cuda=True):
        dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        # dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate)
        model = self
        model.eval()
        total_pred = torch.Tensor()
        total_label = torch.Tensor()
        with torch.no_grad():
            for data in tqdm(dataloader_test, ncols=80):
                if use_cuda:
                    data = data.cuda()
                out = model(data)
                total_pred = torch.cat((total_pred, out.cpu()), 0)
                total_label = torch.cat((total_label, data.y.cpu()), 0)
        all_ci = ci(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        all_mse = mean_squared_error(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        return all_mse, all_ci

    # def collate(self, args):
    #     data = [a[0] for a in args]
    #     sm_emb = [a[1] for a in args]
    #     prot_emb = [a[2] for a in args]
    #     data = torch_geometric.data.Batch.from_data_list(data)  # 对Data对象使用默认的拼接
    #     return data, sm_emb, prot_emb

# def reset_feature(dataset, model, target_seq, target_len, smiles_idx):
#     torch.cuda.empty_cache()
#     batch_size = 128
#     with torch.no_grad():
#         model = model.cuda()
#         model.eval()
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         start = 0
#         for data in tqdm(dataloader, ncols=80):
#             sm, pro = model(data, reset=True)
#             tar_len = []
#             idx = []
#             for i in range(min(batch_size, len(dataset) - start)):
#                 sm_id = dataset[start + i].sm
#                 pro_id = dataset[start + i].target
#                 pro_id = target_seq[pro_id]
#                 tar_len.append(target_len[pro_id])
#                 idx.append(smiles_idx[sm_id])
#             for i in range(start, min(len(dataset), start + batch_size)):
#                 dataset.data[i].x = torch.cat(
#                     [pro[i - start, 1:tar_len[i - start] + 1], sm[i - start, idx[i - start]],
#                      sm[i - start, 0].unsqueeze(0)]).cpu()
#             start = start + batch_size
