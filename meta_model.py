import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as gLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index as ci
from base_models.model_Mgragh import MGraphDTA
from base_models.model_TFusion import TFusionDTA
from base_models.model_NHGNN import NHGNN_DTA
from base_models.model_Mgragh_modify import MGraphDTA as M2
from base_models.model_Mgragh_LSTM import MGraphDTA_LSTM
from base_models.model_TFusion_CNN import TFusionDTA_CNN


def discretize(data, num_bins=1000, dmax=20, dmin=0):
    """
    Discretize the input float data into integer bins.

    Parameters:
        data (np.array): The input data to be discretized, 1-D numpy array.
        num_bins (int): The number of bins to divide the data into.

    Returns:
        np.array: The discretized data, where each float is replaced with its bin label.
    """
    bins = np.linspace(dmax, dmin, num_bins + 1)
    bin_labels = np.digitize(data.detach().cpu(), bins, right=True)
    return bin_labels


class CustomMultiheadAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, use_key_value_cache=False,
                incremental_state=None, static_kv=False, before_softmax=False,
                need_head_weights=False):
        # 使用父类的 forward 方法获取输出和注意力权重
        output, attn_output_weights = super().forward(query, key, value, key_padding_mask,
                                                      need_weights=True, attn_mask=attn_mask)
        # 将注意力权重存储为类属性，以便外部访问
        self.attn_output_weights = attn_output_weights
        return output


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # 替换原本的 self.self_attn 层为我们自定义的注意力层
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)


class TransformerRegressor(nn.Module):
    def __init__(self, save_path, drug_vocab, target_vocab, dataset_name: str, auto_dataset: bool=True, d_model=256, nhead=8, num_layers=2, seed=42, test_ratio=0.15):
        super(TransformerRegressor, self).__init__()
        self.dataset_name = dataset_name
        self.save_path = save_path

        # Mgraph
        self.model_Mgraph = MGraphDTA(data_input_path="data_input", seed=seed, test_ratio=test_ratio, dataset_name=dataset_name,
                                      model_dir="saved_models", model_name=f"Mgraph_{dataset_name}", auto_dataset=auto_dataset)
        self.model_Mgraph.load_state_dict(torch.load(f"saved_models/Mgraph_{dataset_name}_test1.pt"))
        self.model_Mgraph = self.model_Mgraph.cuda()
        print(self.model_Mgraph.val(self.model_Mgraph.test_set))

        # self.model_Mgraph = M2(drug_vocab=drug_vocab, target_vocab=target_vocab, data_path=f"data_input/{dataset_name}/raw/data.csv",
        #                        data_input_path="data_input", seed=seed, test_ratio=test_ratio, dataset_name=dataset_name,
        #                         model_dir="saved_models", model_name=f"Mgraph_{dataset_name}", auto_dataset=auto_dataset)
        # self.model_Mgraph.load_state_dict(torch.load(f"saved_models/Mgraph_modify_davis.pt"))

        # self.model_Mgraph = MGraphDTA_LSTM(drug_vocab=drug_vocab, target_vocab=target_vocab,
        #                                    data_input_path="data_input", seed=seed, test_ratio=test_ratio, dataset_name=dataset_name,
        #                               model_dir="saved_models", model_name=f"Mgraph_{dataset_name}", auto_dataset=auto_dataset)
        # self.model_Mgraph.load_state_dict(torch.load(f"saved_models/Mgraph_LSTM_davis_test2.pt"))

        # Tokenize Fusion
        self.model_TFusion = TFusionDTA(data_path=f"data_input/{dataset_name}/raw/data.csv",
                                        saved_path=f"saved_models/TFusion_{dataset_name}.pt", auto_dataset=auto_dataset,
                                        drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed, test_ratio=test_ratio)
        self.model_TFusion.load_state_dict(torch.load(f"saved_models/TFusion_{dataset_name}_test1.pt"))
        self.model_TFusion = self.model_TFusion.cuda()
        print(self.model_TFusion.val(self.model_TFusion.test_set))

        # self.model_TFusion = TFusionDTA_CNN(data_path=f"data_input/{dataset_name}/raw/data.csv",
        #                                 saved_path=f"saved_models/TFusion_{dataset_name}.pt", auto_dataset=auto_dataset,
        #                                 drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed,
        #                                 test_ratio=test_ratio)
        # self.model_TFusion.load_state_dict(torch.load(f"saved_models/TFusion_CNN_davis.pt"))

        # NHGNN
        self.model_NHGNN = NHGNN_DTA(pretrain_path=f"saved_models/TFusion_{dataset_name}.pt",
                                     data_path=f"data_input/{dataset_name}/raw/data.csv",
                                     saved_path=f"saved_models/NHGNN_{dataset_name}.pkl", auto_dataset=auto_dataset,
                                     drug_vocab=drug_vocab, target_vocab=target_vocab, seed=seed, test_ratio=test_ratio)
        self.model_NHGNN.load_state_dict(torch.load(f"saved_models/NHGNN_{dataset_name}_test1.pkl")['model'])
        self.model_NHGNN = self.model_NHGNN.cuda()
        print(self.model_NHGNN.val(self.model_NHGNN.test_set))

        # meta
        feature_dims = [192, 256, 128, 128, 128, 128*3, 128]
        feature_dims = [f+1 for f in feature_dims]
        n_features = [192, 256, 128, 128, 128, 128]
        n_features = [f + 1 for f in n_features]
        self.embeddings = nn.ModuleList([nn.Linear(d, d_model) for d in feature_dims])
        self.d_model = d_model
        self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc0 = nn.Linear(sum(feature_dims)+7*self.d_model, sum(feature_dims)+7*self.d_model)
        self.fc0 = nn.Linear(7*self.d_model, 1)

        self.fcc1 = nn.Linear(sum(n_features)+7*self.d_model, 14*self.d_model)
        # self.fcc2 = nn.Linear(14*self.d_model, 7*self.d_model)
        self.fcc3 = nn.Linear(14*self.d_model, 1)
        # self.fcc = nn.Linear(sum(feature_dims)+7*self.d_model, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.fc_ab = nn.Linear(3,1)
        self.fc_sq1 = nn.Linear(256*3, 256*3)
        self.fc_sq = nn.Linear(256*3, 256)

        self.emb = nn.Embedding(num_embeddings=1000, embedding_dim=d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model))  # 随机初始化cls_token
        self.fc_trans = nn.Linear(d_model, 1)

    def forward(self, data_Mgraph, data_TFusion, data_NHGNN, get_attn=False):
        batch_size = len(data_Mgraph)
        # with torch.no_grad():
        #     # MG_x1, MG_x2 = self.model_Mgraph.forward(data_Mgraph, output_vector=True)
        #     # TFsm_vec, TFpro_vec, TFsp_vec, TFo2 = self.model_TFusion.forward(data_TFusion, output_vector=True)
        #     # NHGvec = self.model_NHGNN.forward(data_NHGNN, output_vector=True)
        #     o1 = self.model_Mgraph.forward(data_Mgraph, output_vector=True)
        #     o2 = self.model_TFusion.forward(data_TFusion, output_vector=True)
        #     o3 = self.model_NHGNN.forward(data_NHGNN, output_vector=True)


        # src = torch.stack([o1,o2,o3], dim=0)  # (seq_len, batch_size, d_model)
        # output1 = self.transformer_encoder(src)
        # out = output1.transpose(0, 1).reshape(batch_size, 3*256)
        # # out = self.fc_sq(out)
        # out = torch.norm(out, dim=-1).view(-1, 1)

        # src = torch.cat([o1,o2,o3], dim=-1)
        # out = self.fc_sq1(src)
        # out = self.fc_sq(out)
        # out = torch.norm(out, dim=-1).view(-1, 1)

        # features = [MG_x1, MG_x2, TFsm_vec, TFpro_vec, TFsp_vec, TFo2, NHGvec]
        # for iii, f in enumerate(features):
        #     if len(f.shape) != 2:
        #         features[iii] = f.unsqueeze(0)
        # print([f.shape[-1] for f in features])

        # # method 1: Embedding + Transformer + cat  + linear
        # # embeddings = [embed(x) for embed, x in zip(self.embeddings, features)]
        # # src = torch.stack(embeddings, dim=0)  # (seq_len, batch_size, d_model)
        # # output = self.transformer_encoder(src)
        # # output = output.transpose(0, 1).reshape(batch_size, 7*self.d_model)
        # # output = self.fc0(output)

        # # method 2: Embedding cat
        # # output = torch.cat(features, dim=-1)
        # # output = self.fc2(output)

        # method 3: 1 + 2
        # embeddings = [embed(x) for embed, x in zip(self.embeddings, features)]
        # src = torch.stack(embeddings, dim=0)  # (seq_len, batch_size, d_model)
        # output1 = self.transformer_encoder(src)
        # output1 = output1.transpose(0, 1).reshape(batch_size, 7*self.d_model)
        # output2 = torch.cat([features[i] for i in range(len(features)) if i != 5], dim=-1)
        # output = torch.cat([output1, output2], dim=-1)
        # # output = self.fcc(output)
        # output = self.dropout(self.relu(self.fcc1(output)))
        # # output = self.dropout(self.relu(self.fcc2(output)))
        # output = self.fcc3(output)

        with torch.no_grad():
            y1 = self.model_Mgraph.forward(data_Mgraph[0].cuda(), [d.cuda() for d in data_Mgraph[1]], output_vector=False).view(-1,1)
            y2 = self.model_TFusion.forward(data_TFusion, output_vector=False).view(-1,1)
            y3 = self.model_NHGNN.forward(data_NHGNN, output_vector=False).view(-1,1)

        yy = torch.cat([y1,y2,y3], dim=-1)

        disc = torch.Tensor(discretize(yy, num_bins=1000)).cuda().long()
        out = self.emb(disc)
        cls_tokens = self.cls_token.unsqueeze(0).repeat(out.size(0), 1, 1)  # 复制cls_token到每个输入序列
        out = torch.cat((cls_tokens, out), dim=1)  # 附加cls_token到输入序列的开始
        out = out.permute(1, 0, 2)  # Transformer需要(seq_len, batch_size, features)
        out = self.transformer_encoder(out)
        out = self.fc_trans(out[0])

        # out = self.fc_ab(yy)

        return out

    def fit(self, train_sets: list, test_sets: list, graph_data: list, device=torch.device("cuda:0"),
            epochs=3000, early_stop_epoch=300, lr=1e-3, weight_decay=0, batch_size=128, save_model=True):

        print(train_sets[0][0][0].y, train_sets[1][0][2], train_sets[2][0].y)
        print(test_sets[0][0][0].y, test_sets[1][0][2], test_sets[2][0].y)

        # def set_bn_eval(module):
        #     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #         module.eval()

        model = self
        model = model.cuda()
        dataloaders_train = [gLoader(train_set, batch_size=batch_size, shuffle=True) if is_graph
                             else DataLoader(train_set, batch_size=batch_size, shuffle=True)
                             for (train_set, is_graph) in zip(train_sets, graph_data)]
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-5,
                                                              last_epoch=-1, verbose=True)
        best_ci = 0
        best_mse = 100000
        best_epoch = -1
        torch.cuda.empty_cache()
        for epoch in range(epochs):
            total_loss = 0
            total_example = 0
            model.train()
            model.model_Mgraph.eval()
            model.model_TFusion.eval()
            model.model_NHGNN.eval()

            print(f"epoch [{epoch + 1} / {epochs}]")
            for datas in tqdm(zip(*dataloaders_train), ncols=80):
                for i, data in enumerate(datas):
                    try:
                        datas[i] = data.cuda()
                    except:
                        try:
                            datas[i] = [d.cuda() for d in data]
                        except:
                            pass
                optimizer.zero_grad()
                out = model(datas[0], datas[1], datas[2])
                loss = criterion(out.float(), datas[0][0].y.view(-1, 1).float().cuda())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(datas[0])
                total_example += len(datas[0])
            schedule.step()
            print(f"train mse = {total_loss / total_example}")

            # val
            all_mse, all_ci = self.val(test_sets, graph_data)

            if all_mse < best_mse:
                best_ci = all_ci
                best_mse = all_mse
                best_epoch = epoch
                if save_model:
                    model.cpu()
                    torch.save(model.state_dict(), self.save_path)
                    model.cuda()
            else:
                if epoch - best_epoch > early_stop_epoch:
                    break
            print(
                f"total_mse={all_mse}, total_ci={all_ci}, best mse={best_mse}, ci={best_ci}, best_epoch={best_epoch + 1}")


    def val(self, test_sets: list, graph_data: list, batch_size=128, use_cuda=True, output_pred=False, get_attn=False):
        dataloaders_test = [gLoader(test_set, batch_size=batch_size, shuffle=False) if is_graph
                             else DataLoader(test_set, batch_size=batch_size, shuffle=False)
                             for (test_set, is_graph) in zip(test_sets, graph_data)]
        # dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate)
        model = self
        model.eval()
        total_pred = torch.Tensor()
        total_label = torch.Tensor()
        attn_weights = []
        with torch.no_grad():
            for ii, datas in tqdm(enumerate(zip(*dataloaders_test)), ncols=80):
                for i, data in enumerate(datas):
                    try:
                        datas[i] = data.cuda()
                    except:
                        try:
                            datas[i] = [d.cuda() for d in data]
                        except:
                            pass
                out = model(datas[0], datas[1], datas[2])
                total_pred = torch.cat((total_pred, out.cpu()), 0)
                total_label = torch.cat((total_label, data.y.cpu()), 0)

                if get_attn:
                    if ii == 0:
                        for layer in model.transformer_encoder.layers:
                            attn_weights.append(layer.self_attn.attn_output_weights)
                    else:
                        for jj, layer in enumerate(model.transformer_encoder.layers):
                            attn_weights[jj] = torch.cat([attn_weights[jj], layer.self_attn.attn_output_weights], dim=0)

        if output_pred:
            return total_label, total_pred
        all_mse = mean_squared_error(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
        all_ci = ci(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())

        if get_attn:
            return all_mse, all_ci, attn_weights
        return all_mse, all_ci

