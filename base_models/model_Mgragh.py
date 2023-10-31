import os

import math
import numpy as np
import torch.optim as optim
import torch
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index as get_cindex
from torch.utils.data import Subset
from sklearn.model_selection import KFold

# from .MgraphDTA.dataset import *
from .MgraphDTA.preprocessing import GNNDataset
from .MgraphDTA.model import *
from .MgraphDTA.utils import *
from tqdm import tqdm
from lifelines.utils import concordance_index as ci
from sklearn.metrics import mean_squared_error
# from MgraphDTA.preprocessing import *


class MGraphDTA(nn.Module):
    def __init__(self, data_input_path, auto_dataset: bool = True, block_num=3, vocab_protein_size=25+1, embedding_size=128, filter_num=32, out_dim=1,
                 model_name="Mgraph_1", model_dir="saved_models", dataset_name="kiba", seed=42, test_ratio=0.15):
        super().__init__()
        # data
        self.data_input_path = data_input_path
        self.dataset_name = dataset_name
        if auto_dataset:
            self.dataset = self.get_dataset(data_input_path, dataset_name)
            self.train_set, self.test_set = self.split_dataset(self.dataset, seed, test_ratio)

        # model
        self.model_name = model_name
        self.model_dir = model_dir
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num * 3, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
        )
        self.classifier2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )

        # self.classifier3 = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 1)
        # )
        # self.dd1 = nn.Linear(256, 256)
        # self.dd2 = nn.Linear(256, 1)


    def get_dataset(self, data_input_path, dataset_name):
        print(f"processing Mgraph dataset of {dataset_name}...")
        fpath = os.path.join(data_input_path, dataset_name)
        data_set = GNNDataset(fpath)
        return data_set

    def split_dataset(self, dataset, seed, test_ratio):
        test_size = round(len(dataset) * test_ratio)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset) - test_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return train_dataset, test_dataset

    def forward(self, data, output_vector=False, freeze_front=False):
        target = data.target
        protein_x = self.protein_encoder(target)
        ligand_x = self.ligand_encoder(data)

        xx = torch.cat([protein_x, ligand_x], dim=-1)
        x = self.classifier(xx)
        out = self.classifier2(x)

        if output_vector:
            return out

        # if freeze_front:
        #     out = self.dd1(out) * out
        #     out = self.dd2(out)
        #     return out

        out = torch.norm(out, dim=-1)

        # if output_vector:
        #     return torch.cat((xx, out), dim=-1), torch.cat((x, out), dim=-1)
        return out

    def val(self, test_set, batch_size=128, use_cuda=True):
        dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        # dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate)
        model = self
        model.eval()
        total_pred = torch.Tensor()
        total_label = torch.Tensor()
        print("validating...")
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

    def fit(self, train_set, test_set, save_model=True, device=torch.device('cuda'), early_stop_epoch=400, lr=5e-4,
            freeze_front=False):
        print(f"len(train_set)={len(train_set)}")
        print(f"len(test_set)={len(test_set)}")

        train_loader = DataLoader(train_set, batch_size=512, shuffle=True)

        epochs = 3000
        steps_per_epoch = 50
        num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
        break_flag = False

        model = self.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        global_step = 0
        global_epoch = 0

        running_loss = AverageMeter()
        running_cindex = AverageMeter()
        running_best_mse = BestMeter("min")

        model.train()

        if freeze_front:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.dd1.parameters():
                param.requires_grad = True
            for param in model.dd2.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr)

        for i in range(num_iter):
            if break_flag:
                break

            for data in train_loader:

                global_step += 1
                data = data.to(device)
                pred = model(data, freeze_front=freeze_front)

                loss = criterion(pred.view(-1), data.y.view(-1))
                cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1),
                                    pred.detach().cpu().numpy().reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), data.y.size(0))
                running_cindex.update(cindex, data.y.size(0))

                if global_step % steps_per_epoch == 0:

                    global_epoch += 1

                    epoch_loss = running_loss.get_average()
                    epoch_cindex = running_cindex.get_average()
                    running_loss.reset()
                    running_cindex.reset()

                    mse, ci = self.val(test_set)

                    msg = "epoch-%d, loss-%.4f, cindex-%.4f, mse-%.4f ci-%.4f" % (
                    global_epoch, epoch_loss, epoch_cindex, mse, ci)
                    print(msg)

                    if mse < running_best_mse.get_best():
                        running_best_mse.update(mse)
                        if save_model:
                            save_model_dict(model, self.model_dir, self.model_name)
                    else:
                        count = running_best_mse.counter()
                        if count > early_stop_epoch:
                            break_flag = True
                            break


