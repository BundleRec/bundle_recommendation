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

from utils.data import TSFData, get_ur, PreTrainIntentData, PaddingTrainData, pretrain_validate_npy_mat, paddingtrain_validate_npy_mat, PreTrainData
from utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k
from model.VAERecommender import VAE, CVAE
from torch.nn.utils.rnn import pad_sequence

def parse_args():
    parser = argparse.ArgumentParser(description='test recommender')
    parser.add_argument('--dataset', type=str, default='clothing', help='dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--topk', type=int, default=50, help='top number')
    parser.add_argument('--algo_name', type=str, default='vae', help='algo name')
    parser.add_argument('--input_type', type=str, default='mean', help='input type')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--layers', type=int, default=1, help='number of layers')
    parser.add_argument('--epochs', type=int, default=50, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--reg_1', type=float, default=0.001, help='reg_1')
    parser.add_argument('--reg_2', type=float, default=0.001, help='reg_2')
    parser.add_argument('--kl_reg', type=float, default=0.001, help='kl_reg')
    parser.add_argument('--loss_type', type=str, default='CL', help='loss type')

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
    intent_embed = torch.load(f'{data_dir}Completion/intent_embed_compress.npy')
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

    in_dims = 20
    if args.algo_name in ['vae']:
        if args.input_type == 'mean':
            in_dims = 20
            train_dataset = PreTrainData(user_num, item_num, train_set, item_trained)
            training_mat = pretrain_validate_npy_mat(user_num, in_dims, val_tr_set, item_trained)
        elif args.input_type == 'concat':
            in_dims = 200
            train_dataset = PaddingTrainData(user_num, item_num, train_set, test_set, item_trained, 10)
            training_mat = paddingtrain_validate_npy_mat(user_num, in_dims, val_tr_set, item_trained, 10)
    
    elif args.algo_name in ['cvae']:
        if args.input_type == 'mean':
            in_dims = 20
            train_dataset = PreTrainIntentData(user_num, item_num, train_set, item_trained, intent_embed)
            training_mat = pretrain_validate_npy_mat(user_num, in_dims, val_tr_set, item_trained)
        elif args.input_type == 'concat':
            in_dims = 200
            train_dataset = PreTrainIntentData(user_num, item_num, train_set, item_trained, intent_embed, combine_type='concat')
            training_mat = paddingtrain_validate_npy_mat(user_num, in_dims, val_tr_set, item_trained, 10)
    
    
    
    para_list = ['lr', 'batch', 'layers', 'head']
    res_list = []

    lr_opt = [1e-4, 1e-3, 1e-2]
    batch_opt = [64, 128, 256]
    kl_set = [1, 2, 3]
    reg_set = [1 ,2 ,4]
    for lr, batch, kl, reg in product(lr_opt, batch_opt, kl_set, reg_set):
        temp_res = {}
        for para in para_list:
            temp_res[para] = eval(para)
        if args.algo_name == 'vae':
            model = VAE(
                rating_mat=training_mat,
                item_num=item_num,
                in_dims=in_dims,
                q=args.dropout,
                epochs=args.epochs,
                lr=lr,
                reg_1=args.reg_1,
                reg_2=reg,
                beta=kl,
                loss_type=args.loss_type,
                gpuid=args.gpu
            )
        elif args.algo_name == 'cvae':
            model = CVAE(
                rating_mat=training_mat,
                intent_embed=intent_embed,
                item_num=item_num,
                in_dims=in_dims,
                q=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=reg,
                beta=kl,
                loss_type=args.loss_type,
                gpuid=args.gpu
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