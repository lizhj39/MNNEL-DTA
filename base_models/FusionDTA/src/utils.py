from __future__ import print_function,division

import os.path

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import re
import csv
import random
import copy
from math import sqrt
from scipy import stats
import torch.nn.functional as F
from tqdm import tqdm

device_ids = [0, 1, 2, 3]

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def kd_loss(guide, hint, outputs, labels, alpha=0.2):
        KD_loss = F.mse_loss(guide, hint) * \
            alpha + F.mse_loss(outputs, labels) * (1. - alpha)

        return KD_loss

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]

def pack_sequences(X, lengths, padding_idx, order=None):
    
    #X = [x.squeeze(0) for x in X]
    
    n = len(X)#2*batchsize
    #lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]#从后向前取反向的元素
    m = max(lengths)
    
    X_block = X[0].new(n,m).zero_() + padding_idx
    
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x)] = x

    return X_block.cuda(), order

def pack_pre_sequences(X, lengths, padding_idx=0, order=None):
    n = len(X)
    hidden_size = X[0].shape[1]
    if order is None:
        order = np.argsort(lengths)[::-1]#从后向前取反向的元素
    m = max(lengths)
    X_block = X[0].new(n,m,hidden_size).zero_() + padding_idx
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x),:] = x
    
    return X_block, order

def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = torch.zeros(size=X.size())
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i]
    return X_block.cuda()

def split_text(text, length):
    text_arr = re.findall(r'.{%d}' % int(length), text)
    return(text_arr)  # ['123', '456', '789', 'abc', 'def']


def load_protvec(filename):
    protvec = []
    key_aa = {}
    count = 0
    with open(filename, "r") as csvfile:
        protvec_reader = csv.reader(csvfile, delimiter='\t')
        for k, row in enumerate(protvec_reader):
            if k == 0:
                continue
            protvec.append([float(x) for x in row[1:]])
            key_aa[row[0]] = count
            count = count + 1

    protvec.append([0.0] * 100)
    key_aa["zero"] = count
    return protvec, key_aa


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

    
class DrugTargetDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, Y, pid, drug_vocab, is_target_pretrain=True, is_drug_pretrain=False, self_link=True,
                 npz_path="FusionDTA", dataset='davis', ):
        self.X0 = X0 #smiles
        self.X1 = X1 #protein
        self.Y = Y
        self.pid = pid
        # self.smilebet = Smiles()
        # smiles = copy.deepcopy(self.X0)
        # smiles = [x.encode('utf-8').upper() for x in smiles]
        # smiles = [torch.from_numpy(self.smilebet.encode(x)).long() for x in smiles]
        # self.X2 = smiles #smiles
        smiles_idx = {}
        smiles_emb = {}
        smiles_len = {}
        smiles_lengths = np.array([len(x) for x in self.X0])
        seq_len = max(smiles_lengths) + 2

        for sm in tqdm(self.X0):
            if sm in smiles_emb.keys():
                continue
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

            # if len(content) > seq_len:
            #     content = content[:seq_len]

            X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
            smiles_len[sm] = len(content)
            # if seq_len > len(X):
            #     padding = [drug_vocab.pad_index] * (seq_len - len(X))
            #     X.extend(padding)

            smiles_emb[sm] = torch.tensor(X)

            if not smiles_idx.__contains__(sm):
                tem = []
                for i, c in enumerate(X):
                    if atom_dict.__contains__(c):
                        tem.append(i)
                smiles_idx[sm] = tem
        self.smiles_idx = smiles_idx
        self.X2 = smiles_emb
        self.smiles_len = smiles_len

        self.is_target_pretrain = is_target_pretrain
        self.is_drug_pretrain = is_drug_pretrain
        z = np.load(os.path.join(npz_path, f"{dataset}.npz"), allow_pickle=True)
        self.z = z['dict'][()]

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        prot = (i,self.X1[i])
        prot = torch.from_numpy(self.z[self.pid[i]]).squeeze()
        sm = self.X2[self.X0[i]]
        #print(prot.shape)
        return [prot, sm, self.Y[i]]


    

def collate(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)



class Alphabets():
    def __init__(self, chars, encoding=None, missing=255):
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + missing
        if encoding == None:
            self.encoding[self.chars] = np.arange(self.size)
        else:
            self.encoding[self.chars] = encoding
            
    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]
    
class AminoAcid(Alphabets):
    def __init__(self):
        chars = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        super(AminoAcid, self).__init__(chars)
        
class Smiles(Alphabets):
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        super(Smiles, self).__init__(chars)


def adj_mask(adj, maxsize):
    #adj should be list   [torch(N,N)] *batch
    b = len(adj)
    out = torch.zeros(b, maxsize, maxsize) #(b, N, N)
    for i in range(b):
        a = adj[i]
        out[i,:a.shape[0],:a.shape[1]] = a
    return out.cuda()

def graph_pad(x, maxsize):
    #x should be list   [torch(N,features)] *batch
    b = len(x)
    features = x[0].shape[1]
    out = torch.zeros(b, maxsize, features)
    for i in range(b):
        a = x[i]
        out[i,:a.shape[0],:] = a
    return out.cuda(device=a.device)

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
               
    if pair != 0:
        #print(summ)
        #print(pair)
        return summ/pair
    else:
        return 0
    
def feature_mask(sizes, old_edges, rate=0.2):
    # size of feature: [batchsize, num_of_atom, hidden_dim]
    edges = copy.deepcopy(old_edges)
    batchsize = edges.shape[0]
    mask_list = []
    for i in range(batchsize):
        mask_list.append(random.sample(range(sizes[i]),int(sizes[i]*rate)))
        for mask in mask_list[i]:
                edges[i,:,mask] = 0
    #torch.set_printoptions(profile="full")
    return mask_list, edges
    













