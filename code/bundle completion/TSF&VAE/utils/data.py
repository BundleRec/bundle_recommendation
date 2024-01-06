import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import torch
import random
from collections import defaultdict

class TSFData(data.Dataset):
    def __init__(self, data_set):
        """
        Dataset formatter adapted AutoEncoder-like algorithms
        Parameters
        ----------
        neg_set : List,
        is_training : bool,
        """
        super(TSFData, self).__init__()
        # self.useridx = []
        self.itemset = []
        self.target = []

        for uid, group in data_set.groupby('user'):
            # self.useridx.append(int(uid))
            for index in range(len(group)):
                itemids = []
                idx = 0
                for _, row in group.iterrows():
                    if idx == index:
                        self.target.append(int(row['item']))
                    else:
                        itemids.append(int(row['item']))
                    idx += 1
                self.itemset.append(itemids)
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        items = self.itemset[idx]
        targets = self.target[idx]

        return torch.tensor(items), torch.tensor(targets)

class PreTrainData(data.Dataset):
    def __init__(self, user_num, item_num, train_set, embed):
        """
        user-level Dataset formatter adapted AutoEncoder-like algorithms
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        train_set : pd.DataFrame, training set
        test_set : pd.DataFrame, test set
        """
        super(PreTrainData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        # embed.embed_item()
        self.R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # true label
        self.mask_R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # only concern interaction known
        self.user_idx = np.array(range(user_num))
        self.ui_embed = sp.dok_matrix((user_num, 20), dtype=np.float32)
        for uid, group in train_set.groupby('user'):
            total = torch.zeros(20)
            # total = embed.embed_item(torch.tensor(0).cuda()).cpu()
            for index, row in group.iterrows():
                # ie = embed.embed_item(torch.tensor(int(row['item'])).cuda()).cpu()
                ie = embed[int(row['item'])]
                total += ie
            mean_embed = total / group.shape[0]
            for idex, i in enumerate(mean_embed):
                self.ui_embed[int(uid), idex] = i.detach().numpy()

        for _, row in train_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[user, item] = 1.
            self.mask_R[user, item] = 1.

        # for _, row in test_set.iterrows():
        #     user, item = int(row['user']), int(row['item'])
        #     self.R[user, item] = 1.

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        u = self.user_idx[idx]
        ur = self.R[idx].A.squeeze()
        mask_ur = self.mask_R[idx].A.squeeze()
        u_embed = self.ui_embed[idx].A.squeeze()

        return u, ur, mask_ur, u_embed

class PaddingTrainData(data.Dataset):
    def __init__(self, user_num, item_num, train_set, test_set, embed, maxdim):
        """
        user-level Dataset formatter adapted AutoEncoder-like algorithms
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        train_set : pd.DataFrame, training set
        test_set : pd.DataFrame, test set
        """
        super(PaddingTrainData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        # embed.embed_item()
        self.R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # true label
        self.mask_R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # only concern interaction known
        self.user_idx = np.array(range(user_num))
        self.ui_embed = sp.dok_matrix((user_num, 200), dtype=np.float32)
        for uid, group in train_set.groupby('user'):
            total = torch.tensor([])
            # total = embed.embed_item(torch.tensor(0).cuda()).cpu()
            for index, row in group.iterrows():
                ie = embed[int(row['item'])].detach()
                total = torch.cat((ie, total))
            num_padding = maxdim - len(group)
            padding_embed = torch.nn.functional.pad(total, (0, 20 * num_padding))
            for idex, i in enumerate(padding_embed):
                self.ui_embed[int(uid), idex] = i.detach().numpy()

        for _, row in train_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[user, item] = 1.
            self.mask_R[user, item] = 1.
        # print('123')
        # for _, row in test_set.iterrows():
        #     user, item = int(row['user']), int(row['item'])
        #     self.R[user, item] = 1.

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        u = self.user_idx[idx]
        ur = self.R[idx].A.squeeze()
        mask_ur = self.mask_R[idx].A.squeeze()
        u_embed = self.ui_embed[idx].A.squeeze()

        return u, ur, mask_ur, u_embed

class PreTrainIntentData(data.Dataset):
    def __init__(self, user_num, item_num, train_set, embed, intent_embed, combine_type='mean'):
        """
        user-level Dataset formatter adapted AutoEncoder-like algorithms
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        train_set : pd.DataFrame, training set
        test_set : pd.DataFrame, test set
        """
        super(PreTrainIntentData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.intent_embed = intent_embed.cpu().detach().numpy()
        factor_dim = embed.shape[1]
        # embed.embed_item()
        self.R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # true label
        self.mask_R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # only concern interaction known
        self.user_idx = np.array(range(user_num))
        self.ui_embed = sp.dok_matrix((user_num, factor_dim), dtype=np.float32)
        if combine_type == 'mean':
            for uid, group in train_set.groupby('user'):
                total = torch.zeros(20)
                # total = embed.embed_item(torch.tensor(0).cuda()).cpu()
                for index, row in group.iterrows():
                    # ie = embed.embed_item(torch.tensor(int(row['item'])).cuda()).cpu()
                    ie = embed[int(row['item'])]
                    total += ie
                mean_embed = total / group.shape[0]
                for idex, i in enumerate(mean_embed):
                    self.ui_embed[int(uid), idex] = i.detach().numpy()
        elif combine_type == 'concat':
            for uid, group in train_set.groupby('user'):
                total = torch.tensor([])
                # total = embed.embed_item(torch.tensor(0).cuda()).cpu()
                for index, row in group.iterrows():
                    ie = embed[int(row['item'])].detach()
                    total = torch.cat((ie, total))
                num_padding = 10 - len(group)
                padding_embed = torch.nn.functional.pad(total, (0, 20 * num_padding))
                for idex, i in enumerate(padding_embed):
                    self.ui_embed[int(uid), idex] = i.detach().numpy()

        for _, row in train_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[user, item] = 1.
            self.mask_R[user, item] = 1.


    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        u = self.user_idx[idx]
        ur = self.R[idx].A.squeeze()
        mask_ur = self.mask_R[idx].A.squeeze()
        u_embed = self.ui_embed[idx].A.squeeze()
        intent_embed = self.intent_embed[idx]

        return u, ur, mask_ur, u_embed, intent_embed    



def get_ur(df):
    """
    Method of getting user-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur

def pretrain_validate_npy_mat(user_num, item_num, df, embed):
    """
    method of convert dataframe to numoy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe

    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for user, items in df.groupby('user'):
        total = torch.zeros(20)
        for _, row in items.iterrows():
            # print(row['item'])
            # ie = embed.embed_item(torch.tensor(int(row['item'])).cuda()).cpu()
            ie = embed[int(row['item'])]
            total += ie
        mean_embed = total / items.shape[0]
        for idx, vec in enumerate(mean_embed):
            mat[int(user), idx] = vec.detach().numpy()
    # for _, row in df.iterrows():
    #     u, i, r = row['user'], row['item'], row['rating']
    #     mat[int(u), int(i)] = float(r)
    return mat


def paddingtrain_validate_npy_mat(user_num, item_num, df, embed, maxdim):
    """
    method of convert dataframe to numoy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe

    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for user, items in df.groupby('user'):
        total = torch.tensor([])
        for _, row in items.iterrows():
            # ie = embed.embed_item(torch.tensor(int(row['item'])).cuda()).cpu()
            ie = embed[int(row['item'])]
            total = torch.cat((ie, total))
        num_padding = maxdim - len(items)
        padding_embed = torch.nn.functional.pad(total, (0, 20 * num_padding))
        # mean_embed = total / items.shape[0]
        for idx, vec in enumerate(padding_embed):
            mat[int(user), idx] = vec.detach().numpy()
    # for _, row in df.iterrows():
    #     u, i, r = row['user'], row['item'], row['rating']
    #     mat[int(u), int(i)] = float(r)
    return mat