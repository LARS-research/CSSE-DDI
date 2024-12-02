from collections import defaultdict as ddict
import random


def process(dataset, num_rel, n_layer, add_reverse):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """

    so2r = ddict(set)
    so2randomhop = ddict()
    class2num = ddict()
    # print(len(dataset['train']))
    # index = 0
    # cnt = 0
    for subj, rel, obj in dataset['train']:
        class2num[rel] = class2num.setdefault(rel, 0) + 1
        so2r[(subj,obj)].add(rel)
        subj_hop, obj_hop = random.randint(1, n_layer), random.randint(1, n_layer)
        so2randomhop.setdefault((subj, obj), (subj_hop, obj_hop))
        if add_reverse:
            so2r[(obj, subj)].add(rel + num_rel)
            class2num[rel + num_rel] = class2num.setdefault(rel + num_rel, 0) + 1
            so2randomhop.setdefault((obj, subj), (obj_hop, subj_hop))
        # index+=1
        # print("______________________")
        # print(subj, rel, obj)
        # print(so2r[(subj, obj)])
        # print(so2r[(obj, subj)])
        # print(len(so2r))
        # print(2*index)
        # assert len(so2r) == 2*index
    # print(len(so2r))
    # print(cnt)
    so2r_train = {k: list(v) for k, v in so2r.items()}
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            so2r[(subj, obj)].add(rel)
            so2r[(obj, subj)].add(rel + num_rel)
            subj_hop, obj_hop = random.randint(1, n_layer), random.randint(1, n_layer)
            so2randomhop.setdefault((subj, obj), (subj_hop, obj_hop))
            so2randomhop.setdefault((obj, subj), (obj_hop, subj_hop))
    so2r_all = {k: list(v) for k, v in so2r.items()}
    triplets = ddict(list)

    # for (subj, obj), rel in so2r_train.items():
    #     triplets['train_rel'].append({'triple': (subj, rel, obj), 'label': so2r_train[(subj, obj)], 'random_hop':so2randomhop[(subj, obj)]})
    # FOR DDI
    for subj, rel, obj in dataset['train']:
        triplets['train_rel'].append(
            {'triple': (subj, rel, obj), 'label': so2r_train[(subj, obj)], 'random_hop': so2randomhop[(subj, obj)]})
        if add_reverse:
            triplets['train_rel'].append(
                {'triple': (obj, rel+num_rel, subj), 'label': so2r_train[(obj, subj)], 'random_hop': so2randomhop[(obj, subj)]})
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_rel"].append({'triple': (subj, rel, obj), 'label': so2r_all[(subj, obj)], 'random_hop':so2randomhop[(subj, obj)]})
            triplets[f"{split}_rel_inv"].append(
                {'triple': (obj, rel + num_rel, subj), 'label': so2r_all[(obj, subj)], 'random_hop':so2randomhop[(obj, subj)]})
    triplets = dict(triplets)
    return triplets, class2num


def process_multi_label(input, multi_label, pos_neg):
    triplets = ddict(list)
    for index, data in enumerate(input['train']):
        subj, _, obj = data
        triplets['train_rel'].append(
            {'triple': (subj, -1, obj), 'label': multi_label['train'][index], 'pos_neg': pos_neg['train'][index]})
    for split in ['valid', 'test']:
        for index, data in enumerate(input[split]):
            subj, _, obj = data
            triplets[f"{split}_rel"].append({'triple': (subj, -1, obj), 'label': multi_label[split][index],'pos_neg': pos_neg[split][index]})
    triplets = dict(triplets)
    return triplets