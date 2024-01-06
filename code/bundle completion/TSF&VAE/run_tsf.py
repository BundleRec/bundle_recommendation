import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import product

import torch
import torch.utils.data as data

from utils.data import TSFData, get_ur
from utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k
from model.TSFRecommender import TsfCompletion
from torch.nn.utils.rnn import pad_sequence

def parse_args():
    parser = argparse.ArgumentParser(description='test recommender')
    parser.add_argument('--dataset', type=str, default='clothing', help='dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--topk', type=int, default=50, help='top number')

    return parser



if __name__ == '__main__':

    args = parse_args()
    ''' all parameter part '''
    data_dir = ''
    ''' Test Process for Metrics Exporting '''
    if args.dataset == 'clothing':
        data_dir = '/home/user50101/fkd/DaisyRec-PreTrain/Mapping/Clothing/'
    elif args.dataset == 'food':
        data_dir = '/home/user50101/fkd/DaisyRec-PreTrain/Mapping/Food/'
    elif args.dataset == 'electronic':
        data_dir = '/home/user50101/fkd/DaisyRec-PreTrain/Mapping/Electronic/'
    train_set = pd.read_csv(f'{data_dir}Completion/vae_train_data.csv')
    val_tr_set = pd.read_csv(f'{data_dir}Completion/val_tr_set.csv')
    val_te_set = pd.read_csv(f'{data_dir}Completion/val_te_set.csv')
    test_tr_set = pd.read_csv(f'{data_dir}Completion/test_tr_set.csv')
    test_te_set = pd.read_csv(f'{data_dir}Completion/test_te_set.csv')
    item_trained = torch.load(f'{data_dir}item_embedding.pt')
    test_ucands = np.load(f'{data_dir}Completion/val_candidate_set.npy', allow_pickle=True).item()
    val_set = pd.concat([val_tr_set, val_te_set], ignore_index=True)
    ftest_set = pd.concat([test_tr_set, test_te_set], ignore_index=True)
    test_set = pd.concat([val_set, ftest_set], ignore_index=True)
    df = pd.concat([train_set, test_set], ignore_index=True)
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    train_set['rating'] = 1.0
    test_set['rating'] = 1.0

    # get ground truth
    test_ur = get_ur(val_te_set)
    total_train_ur = get_ur(train_set)
    # initial candidate item pool

    train_dataset = TSFData(train_set)

    result_save_path = f'./res/TSF/{args.dataset}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    para_list = ['lr', 'batch', 'layers', 'head']
    res_list = []

    lr_opt = [1e-4, 1e-3, 1e-2]
    batch_opt = [64, 128, 256]
    layers_opt = [1, 2, 3]
    head_opt = [1 ,2 ,4]
    for lr, batch, layers, head in product(lr_opt, batch_opt, layers_opt, head_opt):
        temp_res = {}
        for para in para_list:
            temp_res[para] = eval(para)
        
        model = TsfCompletion(
            item_num=item_num,
            factors=20,
            num_layers=layers,
            num_heads=head,
            item_embedding=item_trained,
            epochs=100,
            lr=lr,
            reg_1=0.0001,
            gpuid=args.gpu,
            vali_tra = val_tr_set,
            input_type='mean'
        )

        def collate_fn(batch):
            data, labels = zip(*batch)
            padded_data = pad_sequence(data, batch_first=True, padding_value=item_num)
            return padded_data, torch.tensor(labels)
        
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

        # build recommender model
        model.fit(train_loader)

        print('Start Calculating Metrics......')

        # get predict result
        print('')
        print('Generate recommend list...')
        print('')
        preds = {}
        topk = args.topk

        for u in tqdm(test_ucands.keys()):
            res = model.predict(u, test_ucands[u]).detach().cpu().numpy().squeeze()
            rec_idx = np.argsort(res)[::-1][:topk]
            top_n = np.array(test_ucands[u])[rec_idx]
            preds[u] = top_n

        # convert rank list to binary-interaction
        for u in preds.keys():
            preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
        # process topN list and store result for reporting KPI
        print('Save metric@k result to res folder...')
        
        for k in [1, 5, 10, 20, 30, 50]:
            if k > args.topk:
                continue
            tmp_preds = preds.copy()
            tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

            pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
            rec_k = recall_at_k(tmp_preds, test_ur, k)
            hr_k = hr_at_k(tmp_preds, test_ur)
            map_k = map_at_k(tmp_preds.values())
            mrr_k = mrr_at_k(tmp_preds, k)
            ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])
            
            hr_res = f'HR@{k}'
            ndcg_res = f'NDCG@{k}'
            temp_res[hr_res] = hr_k
            temp_res[ndcg_res] = ndcg_k

            if k == 5 or k == 10:
                # print(f'Precision@{k}: {pre_k:.4f}')
                # print(f'Recall@{k}: {rec_k:.4f}')
                print(f'HR@{k}: {hr_k:.4f}')
                # print(f'MAP@{k}: {map_k:.4f}')
                # print(f'MRR@{k}: {mrr_k:.4f}')
                print(f'NDCG@{k}: {ndcg_k:.4f}')

        res_list.append(temp_res)
    
    res_cols = list(res_list[0].keys())
    res_df = pd.DataFrame(res_list, columns=res_cols)
    res_df.to_csv(f'{result_save_path}res_mean.csv', index=False)