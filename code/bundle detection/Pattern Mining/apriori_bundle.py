from collections import defaultdict, Counter
from itertools import chain, combinations, groupby
import pymysql as psql


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def getAboveMinSup(itemSet, itemSetList, minSup, globalItemSetWithSup):
    freqItemSet = set()
    localItemSetWithSup = defaultdict(int)

    for item in itemSet:
        for items in itemSetList:
            # if item.issubset(itemSet):
            if issubset(item, items):
                item.sort()
                globalItemSetWithSup["@".join(item)] += 1
                localItemSetWithSup["@".join(item)] += 1

    for item, supCount in localItemSetWithSup.items():
        support = float(supCount / len(itemSetList))
        if (support >= minSup):
            freqItemSet.add(item)
    freqItemList = list()

    for i in freqItemSet:
        freqItemList.append(i.split("@"))

    return freqItemList


def issubset(minSet, maxSet):
    max_set_copy = maxSet.copy()
    for item in minSet:
        if item in max_set_copy:
            max_set_copy.remove(item)
    return len(maxSet) == len(minSet) + len(max_set_copy)


def getUnion(itemSet):
    result = []
    for iidx, i in enumerate(itemSet):
        for jidx, j in enumerate(itemSet):
            if jidx < iidx:
                continue
            for k in j:
                temp = i.copy()
                temp.append(k)
                result.append(temp)
    # remove repeat items
    final_result = ListList2ListSet(result)
    return final_result


def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if list(subset) not in prevFreqSet:
                tempCandidateSet.remove(item)
                break
    return tempCandidateSet


def associationRule(freqItemSet, itemSetWithSup, minConf, pattern_size=2):
    rules = []
    if pattern_size in freqItemSet:
        for item in freqItemSet.get(pattern_size):
            item.sort()
            all_support = 0
            for i in item:
                all_support += itemSetWithSup[i]
                try:
                    confidence = float(
                        itemSetWithSup['@'.join(item)] / all_support)
                except:
                    print(item)
                else:
                    if confidence > minConf:
                        rules.append([item, [], confidence])
    else:
        rules.append([[], [], 0])
    # for k, itemSet in freqItemSet.items():
    #     for item in itemSet:
    #         # subsets = powerset(item)
    #         item.sort()
    #         all_support = 0
    #         for i in item:
    #             all_support += itemSetWithSup[i]
    #
    #         try:
    #             confidence = float(
    #                 itemSetWithSup['@'.join(item)] / all_support)
    #         except:
    #             print(item)
    #         else:
    #             if confidence > minConf:
    #                 # duplicate items in array
    #                 # difference = [it for it in item if it not in list(s)]
    #                 # difference = list((Counter(item) - Counter(s)).elements())
    #                 rules.append([item, [], confidence])
    return rules


# remove repeat lists
def ListList2ListSet(itemset):
    itemset.sort()
    return list(k for k, _ in groupby(itemset))


def getItemSetFromList(itemSetList):
    tempItemSet = list()

    for itemSet in itemSetList:
        for item in itemSet:
            tempItemSet.append([item])
            # tempItemSet.add(frozenset([item]))

    return ListList2ListSet(tempItemSet)
    # return tempItemSet


def apriori(itemSetList, minSup, minConf, pattern_size=2):
    C1ItemSet = getItemSetFromList(itemSetList)
    # Final result global frequent itemset
    globalFreqItemSet = dict()
    # Storing global itemset with support count
    globalItemSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(
        C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
    currentLSet = L1ItemSet
    # the length of the pattern
    k = 2
    # Calculating frequent item set
    while currentLSet:
        #
        if k-1 > pattern_size:
            break
        # Storing frequent itemset
        globalFreqItemSet[k - 1] = currentLSet
        # Self-joining Lk
        candidateSet = getUnion(currentLSet)
        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k - 1)
        # Scanning itemSet for counting support
        currentLSet = getAboveMinSup(
            candidateSet, itemSetList, minSup, globalItemSetWithSup)
        k += 1

    rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf, pattern_size)
    rules.sort(key=lambda x: x[2])

    return globalFreqItemSet, rules


def connectdb():
    db = psql.connect('192.168.2.104', 'root', 'XIAOkai3762', 'collectionA')
    return db


def getsession(bundleid):
    conn = connectdb()
    cursor = conn.cursor()
    # sql = "select item_id from metadata_item where item_category = \'electronic\'"
    # sql = "SELECT compairbundles.bundle_id,compairbundles.items, sampled_bundle.bundle_item FROM collectionA.compairbundles, sampled_bundle where compairbundles.bundle_id = sampled_bundle.bundle_id and sampled_bundle.bundle_domain = \"clothing\""
    sql = "select bundle_item from bundle_set where bundle_id = %s"
    # sql = "SELECT pairbundles.bundle_id,pairbundles.items, sampled_bundle.bundle_item FROM collectionA.pairbundles, sampled_bundle where pairbundles.bundle_id = sampled_bundle.bundle_id and sampled_bundle.bundle_domain = \"clothing\""
    result = []
    for bid in bundleid:
        try:
            # print('executing bundle_set...')
            cursor.execute(sql, bid)
            conn.commit()
            # print('finished...')
            res = cursor.fetchall()
            result.append(res[0])
        #   print(result)
        # print(list(result))
        except Exception as e:
            print(e)
            conn.rollback()

    cursor.close()
    conn.close()
    return result
