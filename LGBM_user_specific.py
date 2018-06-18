
from __future__ import division
import itertools
import os
import time
import sys
# os.chdir('/root/mb/market_basket_data')
os.chdir('/home/prudhvi/Documents/market_basket_data')

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

priors = pd.read_csv('order_products__prior.csv', dtype={
    'order_id': np.int32,
    'product_id': np.uint16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})

print('loading train')

train = pd.read_csv('order_products__train.csv', dtype={
    'order_id': np.int32,
    'product_id': np.uint16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})

print('loading orders')
orders = pd.read_csv('orders.csv', dtype={
    'order_id': np.int32,
    'user_id': np.int32,

    'eval_set': 'category',
    'order_number': np.int16,
    'order_dow': np.int8,
    'order_hour_of_day': np.int8,
    'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv('products.csv', dtype={
    'product_id': np.uint16,
    'product_name': 'category',
    'aisle_id': np.uint8,
    'department_id': np.uint8},
                       usecols=['product_id', 'product_name', 'aisle_id', 'department_id'])


""" orders with all details incluing product list """

products_orders_df = priors.groupby(['order_id']).apply(
    lambda x: x['product_id'].tolist()).reset_index()
products_orders_df = products_orders_df.rename(columns={0: 'Products'})
total_df = pd.merge(orders, products_orders_df, on='order_id', how='left')


""" product related features """

print('computing product f')
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id',rsuffix = '_')
products.set_index('product_id', drop=False, inplace=True)
del prods

""" User related features """


orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)

print('computing user f')

usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user f', users.shape)




""" distinct items data frame """

def dist_items(final_df) :

    dist_items_df = pd.DataFrame()
    dist_items_df['order_id'] = final_df['order_id']
    dist_items = [ ]
    dist_items_count = [ ]
    for index,row in final_df.iterrows():
        dist_items = list(set(dist_items + row['Products']))
        dist_items_count.append(len(dist_items))

    dist_items_df['num_of_distinct_items'] = dist_items_count

    dist_items_df['order_id'] = dist_items_df['order_id'].astype(np.int32)

    return dist_items_df,dist_items


"""Products order ratio """

def productwise_order_ratio(final_df,dist_items) :

    products_ratio_df = pd.DataFrame(index= final_df['order_id'] , columns = dist_items )
    for index,row in final_df.iterrows() :
        for item in row['Products'] :
            products_ratio_df.loc[row['order_id']][int(item)] = 1
    products_ratio_df = products_ratio_df.fillna(value = 0)
    products_ratio_df = products_ratio_df.cumsum(axis = 0)
    products_ratio_df = products_ratio_df.reset_index()
    products_ratio_df.iloc[:,1:] = products_ratio_df.iloc[:,1:].divide(products_ratio_df.index.values + 1 , axis = 0)
    #products_ratio_df = products_ratio_df.melt(id_vars = ['order_id'])
    #products_ratio_df.columns = ['order_id','product_id','productwise_order_ratio']
    #products_ratio_df['order_id'] = products_ratio_df['order_id'].astype(np.int32)

    return  products_ratio_df




""" Average days between orders """

def avg_days(final_df) :

    avg_days_orders_df = final_df[['days_since_prior_order','order_id']]
    overall_avg = sum(avg_days_orders_df['days_since_prior_order'][1:])/(len(avg_days_orders_df)-1)
    avg_days_orders_df = avg_days_orders_df.fillna(value = overall_avg)
    avg_days_orders_df['avg_order_days'] = avg_days_orders_df.days_since_prior_order.cumsum(axis = 0)
    avg_days_orders_df = avg_days_orders_df.reset_index()
    avg_days_orders_df['avg_order_days'] = avg_days_orders_df['avg_order_days'].divide(avg_days_orders_df.index.values+1 , axis = 0)
    avg_days_orders_df['days_since_ratio'] = avg_days_orders_df.days_since_prior_order/avg_days_orders_df.avg_order_days
    avg_days_orders_df['order_id'] = avg_days_orders_df['order_id'].astype(np.int32)

    return avg_days_orders_df

""" days since last """

def days_since_last_productwise(final_df,dist_items) :
    days_since_productwise_df = pd.DataFrame(0 , index= final_df['order_id'] , columns = dist_items )
    occ = {}
    for index, row in final_df.iterrows():
        for item in row['Products']:
            if item in occ.keys():
                occ[item].append(row['order_number'])
            else:
                occ[item] = []
                occ[item].append(row['order_number'])

    for item in occ.keys() :
        inter = [ ]
        k = 0
        for i in list(final_df['order_number']) :
            if i in occ[item] :
                if np.isnan(final_df.iloc[i-1]['days_since_prior_order'])  :
                    inter.append(k)
                else :
                    k = k + final_df.iloc[i-1]['days_since_prior_order']
                    inter.append(k)
                    k = 0
            else :
                if np.isnan(final_df.iloc[i-1]['days_since_prior_order'])  :
                    inter.append(k)
                else :
                    k = k + final_df.iloc[i-1]['days_since_prior_order']
                    inter.append(k)
        days_since_productwise_df[item] = inter

    days_since_productwise_df = days_since_productwise_df.reset_index()

    days_since_productwise_df = days_since_productwise_df.melt(id_vars = ['order_id'])
    days_since_productwise_df.columns = ['order_id','product_id','days_since_productwise']
    days_since_productwise_df['order_id'] = days_since_productwise_df.astype(np.int32)
    return days_since_productwise_df

""" features function """




def features(selected_orders,final_df,priors_single,dist_items_df,avg_days_orders_df,days_since_productwise_df,products_ratio_df,labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    print len(selected_orders)
    for row in selected_orders.itertuples():
        # print i
        i += 1
        if i % 1000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in priors_single.index for product in user_products]
    print Counter(labels)
    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    print('order related features')

    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)

    print('product related features')

    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)
    df = df.merge(days_since_productwise_df)
    df = df.merge(products_ratio_df)
    df = df.merge(dist_items_df, on ='order_id')
    df = df.merge(avg_days_orders_df[['order_id','avg_order_days','days_since_ratio','days_since_prior_order']], on = 'order_id')

    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)

""" final function """

def final_function(userids_list) :

    for id in userids_list :
        #fin_df = total_df[(total_df['user_id'] == id) & (total_df['eval_set']== 'prior')]
        fin_df = total_df[total_df['user_id'] == id]
        fin_df['order_size'] = fin_df.apply(lambda x : len(x['Products']),axis = 1)
        fin_df['Products'] = fin_df['Products'].fillna(value = [ ])
        priors_single_k = priors[priors['user_id']==id]
        priors_single_k.set_index(['order_id', 'product_id'], inplace=True, drop=False)
        orders_single = orders[(orders['user_id']== id) & (orders['eval_set']== 'prior')]
        dist_items_df_k,items = dist_items(fin_df)
        avg_days_orders_df_k = avg_days(fin_df)
        days_since_last_productwise_df_k = days_since_last_productwise(fin_df,items)
        products_orders_df_k = productwise_order_ratio(fin_df,items)
        df_train, labels = features(orders_single,fin_df,priors_single_k,dist_items_df_k,avg_days_orders_df_k,days_since_last_productwise_df_k,products_orders_df_k,labels_given=True)







