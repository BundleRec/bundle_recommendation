import numpy as np
from apriori_bundle import apriori, issubset
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
    
    item_category = np.load(f'{data_dir}item_categories.npy', allow_pickle=True).item()
    train_data = np.load(f'{data_dir}train_data.npy', allow_pickle=True).item()
    validate_data = np.load(f'{data_dir}validate_truth_data.npy', allow_pickle=True).item()
    validate_session = np.load(f'{data_dir}validate_data.npy', allow_pickle=True).item()


    minSup_range = args.start_sup
    minCon_range = args.start_conf
    result = pd.DataFrame(columns=['minSup', 'minConf', 'accuracy', 'seq_pre', 'seq_recall'])
    
    # load data
    train_data = train_data.get(bundle_size)
    validate_data = validate_data.get(bundle_size)
    validate_gdtruth = {}
    validate_test = {}

    validate_bid = list(validate_data.keys())
    for bundleid, gdtruths in validate_data.items():
        if bundleid not in validate_gdtruth:
            validate_gdtruth[bundleid] = []
        for gdtruth in gdtruths:
            validate_gdtruth[bundleid].append(gdtruth.split('@'))

    validate_session = [session[0].split(',') for session in validate_session[validate_bid]]
    for idx, sess in enumerate(validate_bid):
        leaf_category = []
        for item in validate_session[idx]:
            #   for electronic
            # if len(item_category[item][0]) > 1:
            #     leaf_category.append(item_category[item][0][-1])
            # for clothing
            if len(item_category[item])>1:
                max_len = max(map(len,item_category[item]))
                max_cate = [i for i in item_category[item] if len(i) == max_len][0]
                if len(max_cate)>0:
                    leaf_category.append(max_cate[-1])
            else:
                if len(item_category[item][0])>1:
                    leaf_category.append(item_category[item][0][-1])
            validate_test[sess] = leaf_category
    for minSup in minSup_range:
        for minCon in minCon_range:
            freqItemSet, rules = apriori(train_data, minSup=minSup, minConf=minCon)
            # get association rules
            association_rules = [rul[0] + rul[1] for rul in rules]
            # detection
            bid_prediction = {}
            for bid, session in validate_test.items():
                bid_prediction[bid] = []
                for rule in association_rules:
                    detection_items = list((Counter(rule) & Counter(session)).elements())
                    if len(detection_items) > 1:
                        bid_prediction[bid].append(detection_items)

            # evaluation

            bundle_level_recall = 0  # bundle_level_accuracy
            sequence_level_precision = 0
            sequence_level_recall = 0

            for bid, itemsets in bid_prediction.items():
                if len(itemsets) == 0:
                    continue
                true_itemset = {}
                for pre_itemset in itemsets:
                    true_itemset['@'.join(pre_itemset)] = []
                    for itemset in validate_gdtruth.get(bid):
                        if issubset(pre_itemset, itemset):
                            true_itemset['@'.join(pre_itemset)].append(itemset)
                # evaluate bundle-level
                temp_bundle_pre = 0
                temp_bundle_recall = 0
                for pred_items, truth_items in true_itemset.items():
                    # only one hitted
                    if len(truth_items) == 1:
                        temp_bundle_recall += len(
                            list((Counter(pred_items.split('@')) & Counter(truth_items[0])).elements())) / len(truth_items[0])
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
                sequence_level_recall += len(hitted_bundle) / len(validate_gdtruth.get(bid))
                if len(hitted_bundle) > len(validate_gdtruth.get(bid)):
                    print(true_itemset)
                    print(validate_gdtruth.get(bid))

            # bundle_level_precision = bundle_level_precision / len(bid_prediction)
            bundle_level_recall = bundle_level_recall / len(bid_prediction)
            sequence_level_precision = sequence_level_precision / len(bid_prediction)
            sequence_level_recall = sequence_level_recall / len(bid_prediction)
            epoch_result = {'minSup':minSup, 'minConf':minCon, 'accuracy':bundle_level_recall, 'seq_pre':sequence_level_precision, 'seq_recall':sequence_level_recall}
            result = result.append(epoch_result, ignore_index=True)
            print(bundle_level_recall, sequence_level_precision, sequence_level_recall)
    # res_path = './result/cloth_len_'+str(bundle_size)+'.csv'
    # result.to_csv(res_path, index=False)
