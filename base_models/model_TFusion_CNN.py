import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from lifelines.utils import concordance_index as ci
from sklearn.metrics import mean_squared_error
from .NHGNN_DTA.model import LinkAttention
from .NHGNN_DTA.dataset import Pre_DTADataset
from collections import OrderedDict


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class TFusionDTA_CNN(nn.Module):
    def __init__(self, data_path, drug_vocab, target_vocab, saved_path, auto_dataset: bool = True, tar_len=2600, sm_len=536, embedding_dim=128, lstm_dim=64,
                 hidden_dim=128, dropout_rate=0.1, alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5, seed=42, test_ratio=0.15):
        super(TFusionDTA_CNN, self).__init__()

        self.save_path = saved_path

        # data
        self.data_path = data_path  # data.csv file path
        if auto_dataset:
            target_seq, target_emb, target_len, smiles_emb, smiles_len = self.preparation(
                data_path, drug_vocab, target_vocab, tar_len, sm_len)
            self.dataset = self.get_dataset(data_path, smiles_emb, target_emb, smiles_len, target_len)
            self.train_set, self.test_set = self.split_dataset(self.dataset, seed, test_ratio)

        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads

        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_encoder = TargetRepresentation(3, smile_vocab, embedding_dim)
        # self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)
        #
        # self.is_bidirectional = True
        # self.smiles_input_fc = nn.Linear(256, lstm_dim)
        # self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
        #                            bidirectional=self.is_bidirectional, dropout=dropout_rate)
        # # self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        # self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_encoder = TargetRepresentation(3, protein_vocab, embedding_dim)
        # self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        # self.is_bidirectional = True
        # self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)
        #
        # self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
        #                             bidirectional=self.is_bidirectional, dropout=dropout_rate)
        # self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        # self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        # self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        # self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        # self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(96*2, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 256)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

    def preparation(self, data_path, drug_vocab, target_vocab, tar_len, sm_len):
        df = pd.read_csv(data_path)
        smiles = set(df['compound_iso_smiles'])

        target_seq = {}
        for i in range(len(df)):
            target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']

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

            if len(content) > tar_len:
                content = content[:tar_len]

            X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
            target_len[seq] = len(content)
            if tar_len > len(X):
                padding = [target_vocab.pad_index] * (tar_len - len(X))
                X.extend(padding)
            target_emb[seq] = torch.tensor(X)

        smiles_emb = {}
        smiles_len = {}

        for sm in smiles:
            content = []
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

            X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
            smiles_len[sm] = len(content)
            if sm_len > len(X):
                padding = [drug_vocab.pad_index] * (sm_len - len(X))
                X.extend(padding)

            smiles_emb[sm] = torch.tensor(X)

        return target_seq, target_emb, target_len, smiles_emb, smiles_len

    def get_dataset(self, data_path, smiles_emb, target_emb, smiles_len, target_len):
        print("processing TFusion datset...")
        dataset = Pre_DTADataset(path=data_path,
                                 smiles_emb=smiles_emb, target_emb=target_emb,
                                 smiles_len=smiles_len, target_len=target_len)
        return dataset

    def split_dataset(self, dataset, seed, test_ratio):
        test_size = round(len(dataset) * test_ratio)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset) - test_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return train_dataset, test_dataset

    def forward(self, data, output_vector=False):
        protein, smiles = data[1].cuda(), data[0].cuda()
        smiles_lengths = data[-2].cuda()
        protein_lengths = data[-1].cuda()

        # smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim
        # smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        # smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        # # smiles = self.ln1(smiles)
        #
        # protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        # protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim
        # protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        # # protein = self.ln2(protein)
        #
        # smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        # smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # B * lstm_dim*2
        #
        # protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len
        # protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)
        #
        # # drugs and proteins
        # out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        # out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+sm_len * (lstm_dim *2)
        # out_cat, out_attn = self.out_attentions(out_cat, out_masks)

        smiles_out = self.smiles_encoder(smiles)
        protein_out = self.protein_encoder(protein)

        out = torch.cat([smiles_out, protein_out], dim=-1)  # B * (rnn*2 *3)

        out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        o2 = self.out_fc2(out)

        # if output_vector:
        #     return smiles_out, protein_out, out_cat, o2

        out = self.dropout(self.relu(o2))  # B *  hidden_dim*2
        out = self.out_fc3(out)

        if output_vector:
            return out
        out = torch.norm(out, dim=-1).view(-1)

        # del smiles_out, protein_out
        # if output_vector:
        #     out = out.unsqueeze(-1)
        #     return torch.cat((smiles_out, out), dim=-1), torch.cat((protein_out, out), dim=-1), torch.cat((out_cat, out), dim=-1), torch.cat((out_copy, out), dim=-1)

        return out

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

    def fit(self, train_set, test_set, lr=1e-3, weight_decay=0, epochs=3000, early_stop_epochs=80, use_cuda=True,
            save_model=True, batch_size=128):
        model = self
        if use_cuda:
            model = model.cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-5, last_epoch=-1, verbose=True)
        dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        best_ci = 0
        best_mse = 100000
        best_epoch = -1
        torch.cuda.empty_cache()
        for epoch in range(epochs):
            model.train()
            print(f"epoch [{epoch + 1} / {epochs}]")
            for data in tqdm(dataloader_train, ncols=80):
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out.float(), data[2].float().cuda())
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
                    torch.save(model.state_dict(), self.save_path)
                if use_cuda:
                    model.cuda()
            else:
                if epoch - best_epoch > early_stop_epochs:
                    break
            print(
                f"total_mse={all_mse}, total_ci={all_ci}, best mse={best_mse}, ci={best_ci}, best_epoch={best_epoch + 1}")

    def val(self, test_set, batch_size=128, use_cuda=True):
        dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        model = self
        model.eval()
        total_pred = torch.Tensor()
        total_label = torch.Tensor()
        with torch.no_grad():
            for data in tqdm(dataloader_test, ncols=80):
                out = model(data)
                total_pred = torch.cat((total_pred, out.cpu()), 0)
                total_label = torch.cat((total_label, data[2].cpu()), 0)
        all_ci = ci(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        all_mse = mean_squared_error(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        return all_mse, all_ci
