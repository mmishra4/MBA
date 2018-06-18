

from __future__ import division
import itertools
import os
import time
import sys
# os.chdir('/root/mb/market_basket_data')
#os.chdir('/home/prudhvi/Documents/market_basket_data')

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import anytree
from anytree import RenderTree
from copy import deepcopy
import math
from itertools import *
import numpy as np
from collections import Counter
import operator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
sys.setrecursionlimit(1500)
import multiprocessing
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import random
from sklearn import linear_model
#from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

order_products_train_df = pd.read_csv("order_products__train.csv")
order_products_prior_df = pd.read_csv("order_products__prior.csv")
orders_df = pd.read_csv("orders.csv")


class FPNode(object):
    """
    A node in the FP tree.
    """

    def __init__(self, value, count, parent):
        """
        Create the node.
        """
        self.name = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []
        self.transactions = []
        self.flag = 0

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.name == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.name == value:
                return node

        return None

    def add_child(self, value):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child


# frequent items with key value pairs required


def build_header_table(frequent):
    """
    Build the header table.
    """
    headers = {}
    for key in frequent.keys():
        headers[key] = None

    return headers


class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, frequent, root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = frequent
        self.headers = build_header_table(frequent)
        self.root = self.build_fptree(transactions, root_value,
                                      root_count, self.frequent, self.headers)

    def __repr__(self):
        return 'node(' + repr(self.root.value) + ', ' + repr(self.root.children) + ')'

    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None

        return headers

    def build_fptree(self, transactions, root_value, root_count, frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None)

        for ind, transaction in enumerate(transactions):
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: (frequent[x], x), reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, ind, root, headers)

        return root

    def insert_tree(self, items, ind, node, headers):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
            if len(items) == 1:
                child.transactions.append(ind)
        else:
            # Add new child.
            child = node.add_child(first)
            if len(items) == 1:
                child.transactions.append(ind)
            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, ind, child, headers)

    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])


def plist(orders):
    pf_list = {}
    for index, order in enumerate(orders):
        for product in order:
            if product in pf_list.keys():
                pf_list[product]['freq'] += 1
                new_per = (index + 1) - pf_list[product]['ts']
                if new_per > pf_list[product]['per']:
                    pf_list[product]['per'] = new_per
                pf_list[product]['ts'] = index + 1

            else:
                d = {}
                d['freq'] = 1
                d['per'] = index + 1
                d['ts'] = index + 1
                pf_list[product] = d
    for key in pf_list.keys():
        if len(orders) - pf_list[key]['ts'] > pf_list[key]['per']:
            pf_list[key]['per'] = len(orders) - pf_list[key]['ts']

    return pf_list


def prune_plist(pf_list):
    frqs = [pf_list[key]['freq'] for key in pf_list.keys()]
    min_freq = np.percentile(frqs, 20)
    pers = [pf_list[key]['per'] for key in pf_list.keys()]
    max_per = np.percentile(pers, 80)
    for key in pf_list.keys():
        if pf_list[key]['per'] > max_per or pf_list[key]['freq'] < min_freq:
            del pf_list[key]

    for key in pf_list.keys():
        pf_list[key] = pf_list[key]['freq']

    return pf_list


def prune_tree(temp_tree, node_value):
    tree = deepcopy(temp_tree)
    current = tree.headers[node_value]
    while current.link is not None:
        temp = current
        while temp.name is not 0:
            temp.flag = 1
            temp = temp.parent

        if current.parent.name != 0:
            current.parent.transactions.extend(current.transactions)
        current.parent.children.remove(current)
        current = current.link

    temp = current
    while temp.name is not 0:
        temp.flag = 1
        temp = temp.parent

    if current.parent.name != 0:
        current.parent.transactions.extend(current.transactions)
    current.parent.children.remove(current)

    for pre, fill, node in RenderTree(tree.root):
        if node.name != 0 and node.flag == 0:
            node.parent.children.remove(node)

    for pre, fill, node in RenderTree(tree.root):
        # print pre
        # node.flag = 0
        if len(node.transactions) != 0:
            temp = node
            while temp.parent is not None:
                if temp.parent.name != 0:
                    temp.parent.transactions.extend(temp.transactions)
                    temp.parent.transactions = list(set(temp.parent.transactions))

                temp = temp.parent
    return tree


def conditional_patterns(tree_pruned, pattern_node, prns):
    for pre, fill, node in RenderTree(tree_pruned.root):
        if node.name is not 0:
            try:
                trns = node.transactions
                if len(trns) > 0:
                    # print trns
                    trns.sort()
                    k = [(trns[ll + 1] - trns[ll]) for ll in range(len(trns)) if ll <= len(trns) - 2]
                    # print k
                    per = max(k)
                    f = len(trns)
                    pattern = str(pattern_node) + "," + str(node.name)
                    if per < 20 and f > 1:
                        prns[pattern] = [f, per]
            except Exception, e:
                pass
    return prns


def next_pftree(original_tree, node):
    tem = deepcopy(original_tree)
    n = tem.headers[node]
    while True:
        n.parent.transactions.extend(n.transactions)
        n.parent.children.remove(n)
        if n.link is None:
            break
        else:
            n = n.link
    return tem


def generate_patterns(transaction_list, trans):
    frq = prune_plist(trans)
    fptree = FPTree(transaction_list, frq, 0, 0)
    pf_table = frq.items()
    pf_table.sort(key=operator.itemgetter(1, 0))
    patterns = {}
    prns = {}
    for item in pf_table:
        fptree_pruned = prune_tree(fptree, item[0])
        pat = conditional_patterns(fptree_pruned, item[0], prns)
        patterns.update(pat)
        fptree = next_pftree(fptree, item[0])

    return patterns


def intra_inter_time(sorted_transactions_df, pattern, del_min, qmin):
    time_intra = 0
    time_inter = 0
    last = 0
    intra = []
    inter = []
    period = []
    periods_list = []
    x = int(pattern.split(',')[0])
    y = int(pattern.split(',')[1])
    i = 0
    for index, row in sorted_transactions_df.iterrows():
        if x in row[0]:
            if i != 0:
                time_inter = time_inter + row['days_since_prior_order']
            last = row['order_number']
            for index2, row2 in islice(sorted_transactions_df.iterrows(), i + 1, None):
                if x in row2[0] and y not in row2[0]:
                    break
                if y in row2[0]:
                    time_intra = time_intra + row2['days_since_prior_order']
                    intra.append(time_intra)
                    if len(intra) > 1:
                        inter.append(time_inter)
                        if time_inter < del_min and last != 0:
                            period.append(last)
                        elif len(period) >= qmin:
                            periods_list.append(period)
                            period = []
                        else:
                            period = []
                    last = row['order_number']
                    time_inter = 0
                    time_intra = 0
                    break
                else:
                    time_intra = time_intra + row2['days_since_prior_order']
        else:
            if i != 0:
                time_inter = time_inter + row['days_since_prior_order']
        i = i + 1
    if len(period) >= qmin:
        periods_list.append(period)

    return intra, inter, periods_list


def del_max(sorted_transactions_df, pat):
    pats = [item[0] for item in pat.items()]
    inter_all = []
    for pat in pats:
        intra, inter, periods = intra_inter_time(sorted_transactions_df, pat, 2, 0)
        if len(inter) == 0:
            inter_max = 0
        else:
            inter_max = np.median(inter)
        inter_all.append(inter_max)
    cluster_labels = np.digitize(inter_all, bins=np.histogram(inter_all, bins='auto')[1])

    df = pd.DataFrame()
    df['pats'] = pats
    df['del_max'] = inter_all
    df['del_cluster_labels'] = cluster_labels
    df2 = df.groupby(['del_cluster_labels']).apply(lambda x: np.median(x['del_max'])).reset_index()
    df3 = pd.merge(df, df2, on='del_cluster_labels', how='left')
    df3 = df3.rename(columns={0: 'assigned_inter_max'})
    return df3


def q_min(sorted_transactions_df, df):
    pats = df['pats'].tolist()
    del_assigned = df['assigned_inter_max']
    q_medians = []
    for y in range(len(pats)):
        intra, inter, periods = intra_inter_time(sorted_transactions_df, pats[y], del_assigned[y], 0)
        periods_lens = [len(p) for p in periods]
        if len(periods_lens) == 0:
            q_medians.append(0)
        else:
            median_occ = np.median(periods_lens)
            q_medians.append(median_occ)

    q_labels = np.digitize(q_medians, bins=np.histogram(q_medians, bins='auto')[1])
    df['q_medians'] = q_medians
    df['q_cluster_labels'] = q_labels
    df2 = df.groupby(['q_cluster_labels']).apply(lambda x: np.median(x['q_medians'])).reset_index()
    df3 = pd.merge(df, df2, on='q_cluster_labels', how='left')
    df3 = df3.rename(columns={0: 'assigned_q_min'})
    all_occ = []
    num_periods = []
    for index, row in df3.iterrows():
        intra, inter, periods = intra_inter_time(sorted_transactions_df, row['pats'], row['assigned_inter_max'],
                                                 row['assigned_q_min'])
        sum = 0
        for ps in periods:
            sum = sum + len(ps)
        exp_occ = int(sum / len(periods))
        all_occ.append(exp_occ)
        num_periods.append(len(periods))

    periods_labels = np.digitize(all_occ, bins=np.histogram(all_occ, bins='auto')[1])
    df3['num_periods'] = num_periods
    df3['labels_pmin'] = periods_labels
    df4 = df3.groupby(['labels_pmin']).apply(lambda x: np.median(x['num_periods'])).reset_index()
    df5 = pd.merge(df3, df4, on='labels_pmin', how='left')
    df5 = df5.rename(columns={0: 'assigned_p_min'})
    return df5


def tbp_predictor(df, patterns_df):
    Q = 0
    pats = patterns_df['pats'].tolist()
    tot_items = [m.split(',') for m in pats]
    tot_items = [item for sublist in tot_items for item in sublist]
    predictors = Counter(tot_items)

    for index, row in patterns_df.iterrows():
        try:
            intra, inter, periods = intra_inter_time(df, row['pats'], row['assigned_inter_max'], row['assigned_q_min'])
            if len(periods) >= row['assigned_p_min'] and len(periods) != 0:
                p = len(periods[len(periods) - 1])
                q = row['q_medians']
                if p - q > 0:
                    Q = p - q
                else:
                    Q = 0
            kp = row['pats'].split(',')
            predictors[kp[0]] = predictors[kp[0]] + Q
            predictors[kp[1]] = predictors[kp[1]] + Q
        except:
            pass
    # print predictors
    return dict(predictors)




def final_product_list(sorted_transactions_df, orders_df, items_dict):
    sorted_items = sorted(items_dict.items(), key=operator.itemgetter(1), reverse=True)
    order_lengths_Y = np.array([len(sorted_transactions_df[0][i]) for i in sorted_transactions_df[0]][1:]).reshape(-1,1)
    reg_df = pd.DataFrame()
    reg_df['days'] = sorted_transactions_df['days_since_prior_order'][1:]
    reg_df['last_order_len'] = [len(sorted_transactions_df[0][i]) for i in sorted_transactions_df[0]][:-1]
    model = linear_model.LinearRegression()
    model.fit(reg_df, order_lengths_Y)
    final_order_days = int(orders_df.loc[(orders_df['user_id'] == sorted_transactions_df.iloc[1]['user_id']) & (
    orders_df['eval_set'] == 'test')]['days_since_prior_order'])
    final_order_size = order_lengths_Y[-1]
    pred_size = abs(int(model.predict([final_order_days, final_order_size])))

    if pred_size < len(sorted_items):
        final_items = [int(sorted_items[i][0]) for i in range(pred_size)]
    else:
        final_items = [int(item[0]) for item in sorted_items]
    return final_items



def final_submission(total_df, orders_df, userids_list):
    i = 0
    submiss = {}
    for z in userids_list:
        i = i + 1
        try:
            final_df = total_df[total_df['user_id'] == z]
            transaction_list  = final_df[0].tolist()
            trans = plist(final_df[0])
            patrns = generate_patterns(transaction_list, trans)
            final_df = final_df.sort_values(by='order_number')
            df_with_del_max = del_max(final_df, patrns)
            df_with_q_del_p = q_min(final_df, df_with_del_max)
            rated_items = tbp_predictor(final_df, df_with_q_del_p)
            predicted_list = final_product_list(final_df, orders_df, rated_items)
            if len(predicted_list) == 0:
                #pl = prune_plist(trans)
                predicted_list = final_product_list(final_df, orders_df, trans)
                submiss[z] = " ".join(str(c) for c in predicted_list)
            else:
                submiss[z] = " ".join(str(c) for c in predicted_list)
        except Exception, e:
            print e
            submiss[z] = ' '
            pass
        print i, "users predicted"
    return submiss

if __name__ == "__main__":
    products_orders_df = order_products_prior_df.groupby(['order_id']).apply(
        lambda x: x['product_id'].tolist()).reset_index()
    total_df = pd.merge(orders_df, products_orders_df, on='order_id', how='left')
    total_df = total_df[total_df['eval_set'] == 'prior']
    orders_df_test = orders_df[orders_df['eval_set'] == 'test']
    userids_list = list(set(orders_df_test['user_id']))
    kk = final_submission(total_df, orders_df, userids_list)
    sub = pd.DataFrame(kk.items(), columns=['user_id', 'Products'])
    final = pd.merge(orders_df_test, sub, on='user_id', how='outer')
    final.drop(final.columns[[1,2,3,4,5,6]], inplace=True, axis=1)
    final.to_csv(path_or_buf="sub.csv", header=True)




