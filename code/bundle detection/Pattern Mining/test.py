import numpy as np
from apriori_bundle import apriori, getsession, issubset
from collections import Counter
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='electronic')
    parser.add_argument('--bundle_size', type=int, default=10)
    parser.add_argument('--start_sup', type=float, default=0.01)
    parser.add_argument('--start_conf', type=float, default=0.01)
    args = parser.parse_args()
    bundle_size = args.bundle_size
    data_dir = ''
    ''' Test Process for Metrics Exporting '''
    if args.dataset == 'clothing':
        data_dir = 'data/Clothing/'
    elif args.dataset == 'food':
        data_dir = 'data/Food/'
    elif args.dataset == 'electronic':
        data_dir = 'data/Electronic/'

    # load data
    item_category = np.load(f'{data_dir}item_categories.npy', allow_pickle=True).item()
    train_data = np.load(f'{data_dir}train_data.npy', allow_pickle=True).item()
    test_truth_data = np.load(f'{data_dir}test_truth_data.npy', allow_pickle=True).item()
    test_data = np.load(f'{data_dir}test_data.npy', allow_pickle=True).item()

    train_data_list = []
    for i in train_data:
        train_data_list.extend(train_data.get(i))
    train_data = train_data_list
    print('getting association rules.')
    association_rules = []
    for pattern_size in [2, 3, 4, 5]:
        print(pattern_size)
        freqItemSet, rules = apriori(train_data, minSup=0.001, minConf=0.001, pattern_size=pattern_size)
        temp_association_rules = [rul[0] + rul[1] for rul in rules]
        association_rules += temp_association_rules

    # detection
    bid_prediction = {}
    for bid, session in test_data.items():
        bid_prediction[bid] = []
        for rule in association_rules:
            detection_items = list((Counter(rule) & Counter(session)).elements())
            if len(detection_items) > 1:
                bid_prediction[bid].append(detection_items)

    # evaluation
    print('evaluating...')
    # bundle_level_precision = 0
    bundle_level_recall = 0
    sequence_level_precision = 0
    sequence_level_recall = 0
    bundle_num = 0
    for bundles in test_truth_data.values():
        bundle_num += len(bundles)
    all_hit_bundle = {}
    for bid, itemset in bid_prediction.items():
        for truth in test_truth_data.get(bid):
            all_hit_bundle['@'.join(truth)] = []
            for pred in itemset:
                if issubset(pred, truth):
                    all_hit_bundle['@'.join(truth)].append(pred)
    final_hit_bundle = {}
    for truth, pred in all_hit_bundle.items():
        if len(pred) > 1:
            max_len = max(map(len, pred))
            max_cate = [i for i in pred if len(i) == max_len][0]
            final_hit_bundle[truth] = max_cate
        elif len(pred) == 1:
            final_hit_bundle[truth] = pred[0]
    hit_score = 0.0
    for truthb, pred in final_hit_bundle.items():
        if len(pred) > 0:
            len_truth = len(truthb.split('@'))
            score = len(pred) / len_truth
            hit_score += score
    print('bundle_level')
    print(hit_score/bundle_num)



    all_hit_itemset = []
    all_hit_items = []
    for bid, itemsets in bid_prediction.items():
        if len(itemsets) == 0:
            continue
        true_itemset = {}
        for pre_itemset in itemsets:
            true_itemset['@'.join(pre_itemset)] = []
            for itemset in test_truth_data.get(bid):
                if issubset(pre_itemset, itemset):
                    true_itemset['@'.join(pre_itemset)].append(itemset)
                    if itemsets not in all_hit_itemset:
                        all_hit_itemset.append(itemsets)
                    if pre_itemset not in all_hit_items:
                        all_hit_items.append(pre_itemset)
        # evaluate bundle-level
        temp_bundle_pre = 0
        temp_bundle_recall = 0
        for pred_items, truth_items in true_itemset.items():
            # only one hitted
            if len(truth_items) == 1:
                temp_bundle_recall += len(
                    list((Counter(pred_items.split('@')) & Counter(truth_items[0])).elements())) / len(
                    truth_items[0])
                temp_bundle_pre += 1
                # more than one hitted
            elif len(truth_items) > 1:
                # find min length set
                min_len = min(map(len, truth_items))
                min_itemset = [i for i in truth_items if len(i) == min_len][0]
                temp_bundle_recall += len(
                    list((Counter(pred_items.split('@')) & Counter(min_itemset)).elements())) / len(min_itemset)
                temp_bundle_pre += 1
        #  bundle_level_precision += temp_bundle_pre/len(true_itemset)
        bundle_level_recall += temp_bundle_recall / len(true_itemset)
        # evaluate sequence-level
        hit_num = 0
        hitted_bundle = set()
        for pred_items in true_itemset:
            if len(true_itemset.get(pred_items)) > 0:
                hit_num += 1
                # calculate the number of itemsets in gdtruth!!!
            for pre_bundle in true_itemset.get(pred_items):
                hitted_bundle.add('@'.join(pre_bundle))
        sequence_level_precision += hit_num / len(true_itemset)
        sequence_level_recall += len(hitted_bundle) / len(test_truth_data.get(bid))
        if len(hitted_bundle) > len(test_truth_data.get(bid)):
            print(true_itemset)
            print(test_truth_data.get(bid))

    # bundle_level_precision = bundle_level_precision / len(bid_prediction)
    bundle_level_recall = bundle_level_recall / len(bid_prediction)
    sequence_level_precision = sequence_level_precision / len(bid_prediction)
    sequence_level_recall = sequence_level_recall / len(bid_prediction)
    print(bundle_level_recall, sequence_level_precision, sequence_level_recall)

