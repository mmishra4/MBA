# Author : Paul-Antoine Nguyen

# This script considers all the products a user has ordered
#
# We train a model computing the probability of reorder on the "train" data
#
# For the submission, we keep the orders that have a probability of
# reorder higher than a threshold


import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import io
from sklearn import tree


import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
from __future__ import division
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

os.chdir('/home/prudhvi/Documents/market_basket_data')


print('loading prior')
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
        usecols=['product_id','product_name','aisle_id', 'department_id'])

departments = pd.read_csv('departments.csv')
aisles = pd.read_csv('aisles.csv')

import re
products['product_name'] = products.apply(lambda row : re.sub('[^A-Za-z0-9]+', '', row['product_name']),axis = 1)

products.to_csv('products.csv')
priors = pd.concat([priors,train])
priors = priors.merge(products , on='product_id')
priors.to_csv('priors.csv')

""" Counts of different departments and clustering code """


priors = priors.merge(products, on = 'product_id')
priors = priors.merge(orders, on = 'order_id')

priors_produce = priors[priors.department_id == 4]
priors_alcohol = priors[priors.department_id == 5]
priors_pets = priors[priors.department_id == 8]
priors_babies = priors[priors.department_id == 18]

all_users = priors.user_id.unique()
produce_users = priors_produce.user_id.unique()
alcohol_users = priors_alcohol.user_id.unique()
pet_users = priors_pets.user_id.unique()
babies_users = priors_babies.user_id.unique()

col = ['alcohol','pet','baby']

df = pd.DataFrame(0,index = all_users , columns = col)
df = df.reset_index()
df = df.rename(columns = { 'index' : 'user_id'})

df['alcohol'] =df.apply(lambda row : 1 if row['user_id'] in  alcohol_users else 0 , axis = 1)

df['pet'] = df.apply(lambda row : 1 if row['user_id'] in pet_users else 0,axis = 1)

df['baby'] =df.apply(lambda row : 1 if row['user_id'] in babies_users else 0,axis = 1)

""" Days since criteria """

group = orders.groupby('user_id')['days_since_prior_order'].aggregate('median').reset_index()
group['binned_days_since'] = pd.cut(group.days_since_prior_order, [0,4,9,17,24,30],labels = [1,2,3,4,5])
del group['days_since_prior_order']

df = df.merge(group , on = 'user_id')
df= df.fillna(value = 1)

inertia = [ ]

""" Finding optimum clusters 
for i in range(2,14) :

    kmeans = KMeans(n_clusters=i, random_state=0).fit(df.ix[:,1:5])
    inertia.append(kmeans.inertia_)
    print "For" , i , "clusters, inertia is" ,kmeans.inertia_
    
"""

kmeans = KMeans(n_clusters=4, random_state=0).fit(df.ix[:,1:5])

df['labels'] = kmeans.labels_

clf = DecisionTreeClassifier(random_state=0)
clf.fit(df.ix[:,1:5],kmeans.labels_)





print "% of produce users is " , round((produce_users/all_users)*100,2)
print "% of alcohol users is " , round((alcohol_users/all_users)*100,2)
print "% of pets users is " , round((pet_users/all_users)*100,2)
print "% of babies users is " , round((babies_users/all_users)*100,2)



""" Ratio of departments share among fresh produce users """

df_prior_product = pd.merge(left=priors, right= products, how='left').drop('product_id', axis=1)

departments = pd.read_csv('departments.csv')
aisles = pd.read_csv('aisles.csv')

df_prior_product = df_prior_product.merge(aisles, on ='aisle_id')
df_prior_product = df_prior_product.merge(departments, on ='department_id')

plt.figure(figsize=(10,10))
temp_series = df_prior_product['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())))
print sum(sizes[:6])
plt.figure(figsize=(20,8))
sns.barplot(labels, sizes, alpha=0.8 , color = 'C0')
#plt.barplot(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.xticks(rotation='vertical' , fontsize = 12)
plt.show()

all_produce_users = list(priors_produce.user_id.unique())

df_produce_prior = df_prior_product[df_prior_product['user_id'].isin(all_produce_users)]



print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

# Binning departments

def bin_dept(row) :
    if row['department_id'] == 4 :
        return 1
    elif row['department_id'] == 16 :
        return 2
    elif row['department_id'] == 19 or row['department_id'] == 7:
        return 3
    elif row['department_id']== 1 or row['department_id'] == 13 :
        return 4
    else :
        return 5


products['depts_binned'] = products.apply(lambda row:bin_dept(row),axis=1)


# Binning aisles

def bin_aisle(row) :
    if row['aisle_id'] == 24 or row['aisle_id'] == 83:
        return 1
    elif row['aisle_id']==123 or row['aisle_id'] ==120:
        return 2
    elif row['aisle_id'] in [21,84,115,107,91] :
        return 3
    elif row['aisle_id'] in [112,31,116,37,78] :
        return 4
    else :
        return 5


products['aisles_binned'] = products.apply(lambda row:bin_aisle(row),axis=1)


print('add order info to priors')


orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)

###
""" 
df_complete_prod = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right= aisles, how='left').drop(['department_id', 'aisle_id'], axis=1)

"""
# Getting products data frame with 2500 ids only

priors_with_products = pd.merge(left= priors, right=products, how='left' ,on ='product_id')
del priors
del priors_with_products['product_name']

products_list = priors_with_products.product_id.value_counts()
top_products_list = list(products_list.index[:2500])
rest_products_list = list(products_list.index[2500:])
products = products[products.product_id.isin(top_products_list)]

del products_list

top_priors_with_products = priors_with_products[priors_with_products.product_id.isin(top_products_list)]

del priors_with_products

#merged = priors_with_products.merge(top_priors_with_products, indicator=True, how='outer')
# rest_priors_with_products = merged[merged['_merge'] == 'left_only']
# rest_priors_with_products = priors_with_products[priors_with_products.product_id.isin(rest_products_list)]

priors = top_priors_with_products
products_orders_df = priors.groupby(['order_id']).apply(
    lambda x: x['product_id'].tolist()).reset_index()
products_orders_df = products_orders_df.rename(columns={0: 'Products'})






products_orders_df['user_id'] = products_orders_df.order_id.map(orders.user_id)
products_orders_df['days_since'] = products_orders_df.order_id.map(orders.days_since_prior_order)
products_orders_df['order_number'] = products_orders_df.order_id.map(orders.order_number)
products_orders_df['eval_set'] = products_orders_df.order_id.map(orders.eval_set)

userids_list = products_orders_df.user_id.unique()

final = pd.DataFrame()
pk = 1
for p in userids_list :
    print pk , "users done"
    pk = pk + 1
    single_df = products_orders_df[products_orders_df.user_id== p]
    single_df = single_df.sort_values(by ='order_number')
    occ = {}
    for index, row in single_df.iterrows():
        for item in row['Products'] :
            if item in occ.keys() :
                occ[item].append(row['order_number'])
            else :
                occ[item]= []
                occ[item].append(row['order_number'])
    inter_times = { }
    days_last_occ = { }

    for key in occ.keys() :
        inter = [ ]
        for i in range(len(occ[key])) :
            if i+1 == len(occ[key]) :
                days_last_occ[key] = (sum(single_df['days_since'].iloc[occ[key][i]:]))
                break
            else :
                inter.append(sum(single_df['days_since'].iloc[occ[key][i]:occ[key][i+1]]))
        inter_times[key] = inter
    # medians for inter times calculated
    for key in inter_times.keys() :
        if len(inter_times[key]) ==0 :
            inter_times[key] = 0
        else :
            inter_times[key] = np.median(inter_times[key])

    df_with_inter_lastocc = pd.DataFrame([inter_times, days_last_occ]).T
    df_with_inter_lastocc.index.name = 'product_id'
    df_with_inter_lastocc.columns = ['inter_time_median','days_since_last_occ']
    df_with_inter_lastocc = df_with_inter_lastocc.reset_index()
    df_with_inter_lastocc['user_id'] = p
    #print len(df_with_inter_lastocc)
    final = pd.concat([final,df_with_inter_lastocc])
    #print len(final)
final['user_product'] = final.product_id + final.user_id * 100000

print('computing product f')
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id',rsuffix = '_')
products.set_index('product_id', drop=False, inplace=True)
del prods

# orders_one = orders[orders.user_id == 3]

### user features

#orders = orders[orders['user_id'].isin(userids_list)]

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


### userXproduct features

print('compute userXproduct f - this is long...')
priors['user_product'] = priors.product_id + priors.user_id * 100000

# This was to slow !!
#def last_order(order_group):
#    ix = order_group.order_number.idxmax
#    return order_group.shape[0], order_group.order_id[ix],  order_group.add_to_cart_order.mean()
#userXproduct = pd.DataFrame()
#userXproduct['tmp'] = df.groupby('user_product').apply(last_order)

d= dict()
i = 0
for row in priors.itertuples():
    if i % 10000 == 0: print('order row', i)
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)
    i = i + 1

print('to dataframe (less memory)')
userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d

userXproduct = userXproduct.reset_index()

userXproduct.columns = ['user_product','nb_orders', 'last_order_id', 'sum_pos_in_cart']

userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)

userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)

userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)


userXproduct = userXproduct.merge(final, on='user_product' , how ='left')

userXproduct['inter_feature'] = userXproduct.days_since_last_occ/ (userXproduct.inter_time_median + 0.05)

print('user X product f', len(userXproduct))

del priors

### train / test orders ###
print('split orders : train, test')

orders = orders[orders.user_id.isin(list(users.index))]
test_orders_2500 = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']


train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

### build list of candidate products to reorder, with features ###

def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i=0
    print len(selected_orders)
    for row in selected_orders.itertuples():
        #print i
        i+=1
        if i%1000 == 0: print('order row',i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list +=  user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]

    print Counter(labels)

    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(users.average_basket)
    
    print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    df['inter_feature'] = df.z.map(userXproduct.inter_feature)

    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)
    #df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)
    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)






df_train,labels = features(train_orders, labels_given=True)

df_train['aisles_binned'] = df_train.product_id.map(products.aisles_binned)
df_train['depts_binned'] = df_train.product_id.map(products.depts_binned)



f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio','product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio', 'aisles_binned','depts_binned',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last'] # 'dow', 'UP_same_dow_as_last_order'


print('formating for lgb')
d_train = lgb.Dataset(df_train[f_to_use],
                      label=labels,
                      categorical_feature=['aisles_binned', 'depts_binned'])  # , 'order_hour_of_day', 'dow'
del df_train

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100

print('light GBM train :-)')
bst = lgb.train(params, d_train, ROUNDS)
# lgb.plot_importance(bst, figsize=(9,20))
del d_train

### build candidates list for test ###

df_test, _ = features(test_orders)

df_test['aisles_binned'] = df_test.product_id.map(products.aisles_binned)
df_test['depts_binned'] = df_test.product_id.map(products.depts_binned)


print('light GBM predict')
preds = bst.predict(df_test[f_to_use])

df_test['pred'] = preds

TRESHOLD = 0.18  # guess, should be tuned with crossval on a subset of train data

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('sub.csv', index=False)

userXproduct.to_csv('userXproduct.csv', index=False)

""" JUNK Code 


df_train, labels = features(orders_one, labels_given=True)

order_list = []
product_list = []
labels = []
i = 0
for row in orders_one.itertuples():
    print i
    i += 1
    if i % 10000 == 0: print('order row', i)
    order_id = row.order_id
    user_id = row.user_id
    user_products = users.all_products[user_id]
    #print user_products
    product_list += user_products
    print len(product_list)
    #print product_list
    order_list += [order_id] * len(user_products)

    labels += [(order_id, product) in train.index for product in user_products]
    #print labels
    #if i==2 : break


orders_df = orders[(orders['eval_set'] == 'prior')]

userids_list = list(set(orders['user_id']))

userids_list = userids_list[:200]

priors = priors[priors['user_id'].isin(userids_list)]


"""
