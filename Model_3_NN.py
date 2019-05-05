# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:34:10 2019

@author: Rajath Nandan
"""
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:16:54 2019

@author: Rajath Nandan
"""

import numpy as np
import pandas as pd
import os
from collections import defaultdict
import time
import datetime
from pprint import pprint
from keras.utils import plot_model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
seed = 0
np.random.seed(seed)
import pydot
import pydotplus
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot


#%% Classified the columns as product related or customer related. Also decided which columns to be dropped

cat_col = ['fecha_dato', 'ncodpers','ind_empleado','pais_residencia','sexo','age','fecha_alta','ind_nuevo','antiguedad','indrel', 'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall', 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento']

notuse = ["ult_fec_cli_1t","nomprov"]

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']

product_col = product_col[2:]

train_cols = cat_col + product_col

#%%  loaded the dataframe and then parsed the dates for processing
df_train = pd.read_csv('train_ver2.csv',usecols=train_cols)
pd.set_option('display.max_columns', None)
df_june = pd.read_csv('test_ver2.csv',usecols = cat_col)
df_train['fecha_dato'] = pd.to_datetime(df_train['fecha_dato'], format='%Y-%m-%d', errors='ignore')

#%% created a new df for every product picked for 6 months including June & before. Later merged all the dfs created
month = 6

df_train_curr = df_train.loc[df_train['fecha_dato']=='2015-06-28',:]
df_train_5 = df_train.loc[df_train['fecha_dato']=='2015-05-28', product_col+['ncodpers']]
df_train_4 = df_train.loc[df_train['fecha_dato']=='2015-04-28', product_col+['ncodpers']]
df_train_3 = df_train.loc[df_train['fecha_dato']=='2015-03-28', product_col+['ncodpers']]
df_train_2 = df_train.loc[df_train['fecha_dato']=='2015-02-28', product_col+['ncodpers']]
df_train_1 = df_train.loc[df_train['fecha_dato']=='2015-01-28', product_col+['ncodpers']]

dfm = pd.merge(df_train_curr,df_train_5, how='left', on=['ncodpers'], suffixes=('', '_5'))
dfm = pd.merge(dfm,df_train_4, how='left', on=['ncodpers'], suffixes=('', '_4'))
dfm = pd.merge(dfm,df_train_3, how='left', on=['ncodpers'], suffixes=('', '_3'))
dfm = pd.merge(dfm,df_train_2, how='left', on=['ncodpers'], suffixes=('', '_2'))
dfm = pd.merge(dfm,df_train_1, how='left', on=['ncodpers'], suffixes=('', '_1'))

dfm.head()


dfm1 = pd.merge(df_june,df_train_5, how='inner', on=['ncodpers'], suffixes=('', '_5'))
dfm1 = pd.merge(dfm1,df_train_4, how='left', on=['ncodpers'], suffixes=('', '_4'))
dfm1 = pd.merge(dfm1,df_train_3, how='left', on=['ncodpers'], suffixes=('', '_3'))
dfm1 = pd.merge(dfm1,df_train_2, how='left', on=['ncodpers'], suffixes=('', '_2'))
dfm1 = pd.merge(dfm1,df_train_1, how='left', on=['ncodpers'], suffixes=('', '_1'))


#%% split the columns to previous 5 months & current cols

prevcols = [col for col in dfm.columns if '_ult1_'+str(month-1) in col]
currcols = [col for col in dfm.columns if '_ult1' == col[-5:]]


#%% replace all na values in all columns with 0

all_product_col = [col for col in dfm.columns if '_ult1' in col]

for col in all_product_col:
    dfm[col].fillna(0, inplace=True)

all_product_col1 = [col for col in dfm1.columns if '_ult1' in col]

for col in all_product_col1:
    dfm1[col].fillna(0, inplace=True)

#%% dropped May data columns
for col in product_col:
    dfm1[col+'_5'] =dfm1[col]

dfm1.drop(product_col, axis=1, inplace=True)

dfm1.drop('fecha_dato', axis=1, inplace=True)
#%% found the diff. between current month & month 5. Handled the negative values created

for col in currcols:
    dfm[col] = dfm[col] - dfm[col+'_'+str(month-1)]
    dfm[col] = dfm[col].apply(lambda x: max(x,0))

#%% dropped the rows which doesn't have any products in current month

dfm = dfm[dfm[currcols].sum(axis=1) >0]
print(dfm[currcols].sum(axis = 1).value_counts())
dfm = dfm.reset_index(drop=True)
print(dfm.shape)

#%% considered only those users who are picking new products in June & created a target col for assisting prediction. Created a df with the rows that satisfied the constraints

data = []

for index, row in dfm.iterrows():
    if index%1000 == 0:
        print(index)
        print('finish: ', time.strftime('%a %H:%M:%S'))
    if row[currcols].sum() > 0:
        for i,col in enumerate(currcols):
            if row[col] == 1:
                row['target'] = float(currcols.index(col))
                data.append(list(row))

df_new = pd.DataFrame(data, columns = list(dfm.columns.values) + ['target'])

#%% saved only those columns which are not part of the current month

df_new.drop(currcols+['fecha_dato'], axis=1, inplace=True)
print(df_new.shape)
print(df_new['target'].value_counts())


#%% loaded the data sets

df_train = df_new
df_test = dfm1
pd.set_option('display.max_columns', None)

#%%

start_time = datetime.datetime.now()

demographic_cols = ['ncodpers','fecha_alta','ind_empleado','pais_residencia','sexo','age','ind_nuevo','antiguedad','indrel',
 'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall',
 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento']

notuse = ["ult_fec_cli_1t","nomprov",'fecha_dato']

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']


#%% filtered them as the constraint of rows set below

def filter_data(df):
    df = df[df['ind_nuevo'] == 0]
    df = df[df['antiguedad'] != -999999]
    df = df[df['indrel'] == 1]
    df = df[df['indresi'] == 'S']
    df = df[df['indfall'] == 'N']
    df = df[df['tipodom'] == 1]
    df = df[df['ind_empleado'] == 'N']
    df = df[df['pais_residencia'] == 'ES']
    df = df[df['indrel_1mes'] == 1]
    df = df[df['tiprel_1mes'] == ('A' or 'I')]
    df = df[df['indext'] == 'N']

filter_data(df_train)

#%% dropped the following columns pertaining to the user

drop_column = ['ind_nuevo','indrel','indresi','indfall','tipodom','ind_empleado','pais_residencia','indrel_1mes','indext','conyuemp','fecha_alta','tiprel_1mes']

df_train.drop(drop_column, axis=1, inplace = True)
df_test.drop(drop_column, axis=1, inplace = True)

#%% Data imputing with median values as per province

df_test["renta"]   = pd.to_numeric(df_test["renta"], errors="coerce")
unique_prov = df_test[df_test.cod_prov.notnull()].cod_prov.unique()
grouped = df_test.groupby("cod_prov")["renta"].median()

def impute_renta(df):
    df["renta"]   = pd.to_numeric(df["renta"], errors="coerce")
    for cod in unique_prov:
        df.loc[df['cod_prov']==cod,['renta']] = df.loc[df['cod_prov']==cod,['renta']].fillna({'renta':grouped[cod]}).values
    df.renta.fillna(df_test["renta"].median(), inplace=True)

impute_renta(df_train)
impute_renta(df_test)

#%% Dropped all the inactive clientile

def drop_na(df):
    df.dropna(axis = 0, subset = ['ind_actividad_cliente'], inplace = True)

drop_na(df_train)


#%% transforming categorical features with get_dummy

dummy_col = ['sexo','canal_entrada','cod_prov','segmento']
dummy_col_select = ['canal_entrada','cod_prov']

#%% creating the dummy columns using which the categorical data has to be converted to numerical data

limit = int(0.01 * len(df_train.index))
use_dummy_col = {}

for col in dummy_col_select:
    trainlist = df_train[col].value_counts()
    use_dummy_col[col] = []
    for i,item in enumerate(trainlist):
        if item > limit:
            use_dummy_col[col].append(df_train[col].value_counts().index[i])

#%% added the dummy columns for transforming categorical variables to numerical variables

def get_dummy(df):
    for col in dummy_col_select:
        for item in df[col].unique():
            if item not in use_dummy_col[col]:
                row_index = df[col] == item
                df.loc[row_index,col] = np.nan
    return pd.get_dummies(df, prefix=dummy_col, columns = dummy_col)

df_train = get_dummy(df_train)
df_test = get_dummy(df_test)

#%% cleaning data

def clean_age(df):
    df["age"]   = pd.to_numeric(df["age"], errors="coerce")
    max_age = 80

    df["age"]   = df['age'].apply(lambda x: min(x ,max_age))
    df["age"]   = df['age'].apply(lambda x: round( x/max_age, 6))

def clean_renta(df):
    max_renta = 1.0e6

    df["renta"]   = df['renta'].apply(lambda x: min(x ,max_renta))
    df["renta"]   = df['renta'].apply(lambda x: round( x/max_renta, 6))

def clean_antigue(df):
    df["antiguedad"]   = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["antiguedad"] = df["antiguedad"].replace(-999999, df['antiguedad'].median())
    max_antigue = 256

    df["antiguedad"]   = df['antiguedad'].apply(lambda x: min(x ,max_antigue))
    df["antiguedad"]   = df['antiguedad'].apply(lambda x: round( x/max_antigue, 6))

clean_age(df_train)
clean_age(df_test)

clean_renta(df_train)
clean_renta(df_test)

clean_antigue(df_train)
clean_antigue(df_test)

#%% classifying columns as per lag  & created a column that captures whether a product in head at a given lag

product_col_5 = [col for col in df_train.columns if '_ult1_5' in col]
product_col_4 = [col for col in df_train.columns if '_ult1_4' in col]
product_col_3 = [col for col in df_train.columns if '_ult1_3' in col]
product_col_2 = [col for col in df_train.columns if '_ult1_2' in col]
product_col_1 = [col for col in df_train.columns if '_ult1_1' in col]

df_train['tot5'] = df_train[product_col_5].sum(axis=1)
df_test['tot5'] = df_test[product_col_5].sum(axis=1)
df_train['tot4'] = df_train[product_col_4].sum(axis=1)
df_test['tot4'] = df_test[product_col_4].sum(axis=1)
df_train['tot3'] = df_train[product_col_3].sum(axis=1)
df_test['tot3'] = df_test[product_col_3].sum(axis=1)
df_train['tot2'] = df_train[product_col_2].sum(axis=1)
df_test['tot2'] = df_test[product_col_2].sum(axis=1)
df_train['tot1'] = df_train[product_col_1].sum(axis=1)
df_test['tot1'] = df_test[product_col_1].sum(axis=1)

#%% compute the total number of products held by a customer

for col in product_col[2:]:
    df_train[col+'_past'] = (df_train[col+'_5']+df_train[col+'_4']+df_train[col+'_3']+df_train[col+'_2']+df_train[col+'_1'])/5
    df_test[col+'_past'] = (df_test[col+'_5']+df_test[col+'_4']+df_test[col+'_3']+df_test[col+'_2']+df_test[col+'_1'])/5

#%% created another column with opposite convention regarding the change in status & created diff columns to find out how many customers have took up new products (per products) every month

for pro in product_col[2:]:
    df_train[pro+'_past'] = df_train[pro+'_past']*(1-df_train[pro+'_5'])
    df_test[pro+'_past'] = df_test[pro+'_past']*(1-df_test[pro+'_5'])

for col in product_col[2:]:
    for month in range(2,6):
        df_train[col+'_'+str(month)+'_diff'] = df_train[col+'_'+str(month)] - df_train[col+'_'+str(month-1)]
        df_test[col+'_'+str(month)+'_diff'] = df_test[col+'_'+str(month)] - df_test[col+'_'+str(month-1)]
        df_train[col+'_'+str(month)+'_add'] = df_train[col+'_'+str(month)+'_diff'].apply(lambda x: max(x,0))
        df_test[col+'_'+str(month)+'_add'] = df_test[col+'_'+str(month)+'_diff'].apply(lambda x: max(x,0))


#%% categorized the columns & added columns which captures the total number of products added

product_col_5_diff = [col for col in df_train.columns if '5_diff' in col]
product_col_4_diff = [col for col in df_train.columns if '4_diff' in col]
product_col_3_diff = [col for col in df_train.columns if '3_diff' in col]
product_col_2_diff = [col for col in df_train.columns if '2_diff' in col]

product_col_5_add = [col for col in df_train.columns if '5_add' in col]
product_col_4_add = [col for col in df_train.columns if '4_add' in col]
product_col_3_add = [col for col in df_train.columns if '3_add' in col]
product_col_2_add = [col for col in df_train.columns if '2_add' in col]

product_col_all_diff = [col for col in df_train.columns if '_diff' in col]
product_col_all_add = [col for col in df_train.columns if '_add' in col]

df_train['tot5_add'] = df_train[product_col_5_add].sum(axis=1)
df_test['tot5_add'] = df_test[product_col_5_add].sum(axis=1)
df_train['tot4_add'] = df_train[product_col_4_add].sum(axis=1)
df_test['tot4_add'] = df_test[product_col_4_add].sum(axis=1)
df_train['tot3_add'] = df_train[product_col_3_add].sum(axis=1)
df_test['tot3_add'] = df_test[product_col_3_add].sum(axis=1)
df_train['tot2_add'] = df_train[product_col_2_add].sum(axis=1)
df_test['tot2_add'] = df_test[product_col_2_add].sum(axis=1)

cols = list(df_train.drop(['target','ncodpers']+product_col_all_diff+product_col_all_add, 1).columns.values)

id_preds = defaultdict(list)
ids = df_test['ncodpers'].values

# predict model
y_train = pd.get_dummies(df_train['target'].astype(int))
x_train = df_train[cols]

#%%

# create model
model = Sequential()
model.add(Dense(150, input_dim=len(cols), activation='relu'))
model.add(Dense(22, activation='softmax'))
# Compile model
stopping = EarlyStopping(monitor = 'categorical_accuracy', patience = 10,verbose = 1,mode = 'max',restore_best_weights = True)
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['categorical_accuracy'] )

#model.fit(x_train.as_matrix(), y_train.as_matrix(), validation_split=0.2, nb_epoch=150, batch_size=10)
model.fit(x_train.as_matrix(), y_train.as_matrix(), nb_epoch=45, batch_size=10, callbacks=[stopping])

#%%
x_test = df_test[cols]
x_test = x_test.fillna(0)

p_test = model.predict(x_test.as_matrix())
#p_test_df = pd.DataFrame(p_test,columns = product_col[2:])
#p_test_df.to_csv("nn_prob.csv",index=False)

for id, p in zip(ids, p_test):
    #id_preds[id] = list(p)
    id_preds[id] = [0,0] + list(p)

customer=list(id_preds.keys())

fraction = 1
id_preds_combined = {}

for uid, p in id_preds.items():
    id_preds_combined[uid] = fraction*np.asarray(id_preds[uid])

id_preds = id_preds_combined

#%%

usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
df_recent =  pd.read_csv('train_ver2.csv',usecols=usecols)
df_recent=df_recent[df_recent['ncodpers'].isin(customer)]
df_recent.fillna(0, inplace=True)

sample = pd.read_csv('sample_submission.csv')
# check if customer already have each product or not.
already_active = {}
for row in df_recent.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(tuple(product_col), row) if c[1] > 0]
    already_active[id] = active

# add 7 products(that user don't have yet), higher probability first -> train_pred
train_preds = {}
for id, p in id_preds.items():
    preds = [i[0] for i in sorted([i for i in zip(tuple(product_col), p) if i[0] not in already_active[id]], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds

test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))

#%%

plot_model(model,show_shapes=True, to_file='model3.png')

#%%
sample['added_products'] = test_preds
sample.to_csv('Keras.csv', index=False)
print(datetime.datetime.now()-start_time)