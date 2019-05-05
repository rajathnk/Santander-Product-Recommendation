import pickle
import gc
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

months = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28',
         '2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28',
         '2015-11-28', '2015-12-28', '2016-01-28', '2016-02-28', '2016-03-28',
         '2016-04-28', '2016-05-28']
       
prods = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1', 'ind_nom_pens_ult1',  'ind_recibo_ult1']
        
targetprods = ['ind_recibo_ult1', 'ind_cco_fin_ult1', 'ind_nom_pens_ult1',
    'ind_nomina_ult1', 'ind_tjcr_fin_ult1', 'ind_ecue_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctma_fin_ult1', 'ind_reca_fin_ult1',
    'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_valo_fin_ult1']


prodict = dict(zip(range(len(targetprods)),targetprods))
        
clientfeatures = ['fecha_dato', 'ncodpers', 'ind_empleado', 
       'pais_residencia', 'sexo', 'age', 'fecha_alta', 'ind_nuevo', 
       'antiguedad', 'indrel', 'ult_fec_cli_1t', 'indrel_1mes',
       'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada',
       'indfall', 'tipodom', 'cod_prov', 'nomprov',
       'ind_actividad_cliente', 'renta', 'segmento']

def getDataByMonth(dates, molist, clientfeat, prodlist):
    ids = dates.index[dates.fecha_dato.isin(molist)]
    usecols = clientfeat + prodlist
    modata = pd.read_csv('../input/train_ver2.csv', usecols=usecols,
                skiprows= range(1,ids[0]+1), nrows=len(ids), header=0)
    return modata
    
def getClientFeatures(mdata, prmdata):
    print('Employee index')
    mdata['employeeYN'] = mdata['ind_empleado']. \
                    map(lambda x: 0 if (x=='N' or x=='S') else 1)
    print(np.unique(mdata.employeeYN, return_counts=True))

    print('Seniority')
    if mdata.antiguedad.dtype != np.int64 and mdata.antiguedad.dtype != np.float64:
        mdata['antiguedad'] = mdata.antiguedad.str.strip()
        mdata['antiguedad'] = mdata.antiguedad.map(lambda x: None if x=='NA' 
                                                                else int(x))
    mdata.antiguedad[mdata.antiguedad<0] = mdata.antiguedad.max()
    mdata.antiguedad.fillna(mdata.antiguedad.median(), inplace=True)

    print('Age')
    if mdata.age.dtype != np.int64 and mdata.age.dtype != np.float64:
        mdata['age'] = mdata.age.str.strip()
        mdata['age'] = mdata.age.map(lambda x: None if x=='NA' else int(x))
    mdata.age.fillna(mdata.age.median(), inplace=True)
    
    print('Age categorical')
    mdata['agecateg'] = mdata.age.map(lambda x: '<18' if x <18 
                            else '18-25' if (x>=18 and x<25)
                            else '25-35' if (x>=25 and x<35)
                            else '35-45' if (x>=35 and x<45)
                            else '45-55' if (x>=45 and x<55)
                            else '55-65' if (x>=55 and x<65)
                            else '>65' if x>=65  else 'NA')
    print(np.unique(mdata.agecateg, return_counts=True))

    print('New customer index')
    print(np.unique(mdata.ind_nuevo, return_counts=True))
    
    print('Customer type')
    print(np.unique(mdata.indrel_1mes, return_counts=True))
    
    print('Customer relation type')
    mdata.tiprel_1mes.fillna('I', inplace=True)
    print(np.unique(mdata.tiprel_1mes, return_counts=True))

    print('Activity index')
    print(np.unique(mdata.ind_actividad_cliente, return_counts=True))
    
    print('Sex')
    mdata['sexo'] = mdata['sexo'].map({'H':0, 'V':1, None:1}).astype(int)
    print(np.unique(mdata.sexo, return_counts=True))

    print('Segmentation')
    mdata.segmento.fillna('02 - PARTICULARES', inplace=True)
    mdata.segmento = mdata.segmento.map(lambda x: x[:2])
    print(np.unique(mdata.segmento, return_counts=True))

    print('Deceased client')
    mdata.indfall.fillna('N', inplace=True)
    mdata['indfall'] = mdata['indfall'].map({'N':0, 'S':1}).astype(int)
    print(np.unique(mdata.indfall, return_counts=True)) 
    
    print('Province code')
    print(np.unique(mdata.cod_prov, return_counts=True))
    mdata.cod_prov.fillna(99, inplace=True) # Foreign clients

    print('Income') # 702435 non-null float64
    # Convert NA (string) to NaN
    mdata['renta'] = pd.to_numeric(mdata['renta'], errors='coerce')
    print('Fill missing incomes with medians')
    for ac in mdata.agecateg.unique(): # agecateg
        for seg in mdata.segmento.unique(): # segment
            med = mdata[(mdata.agecateg==ac) & (mdata.segmento==seg)]['renta'] \
                                .dropna().median()
            mdata.loc[(mdata.renta.isnull()) & (mdata.agecateg==ac) & \
                        (mdata.segmento==seg), 'renta'] = med
    plt.show()
    
    Xclient = pd.concat([mdata[['ncodpers', 'employeeYN', 'sexo', 'age',
                                'antiguedad', 'indfall',
                                'ind_actividad_cliente', 'renta']], 
                        pd.get_dummies(mdata['tiprel_1mes'].apply(str)),
                        pd.get_dummies(mdata['segmento'].apply(str))],
                        axis=1)
    print(Xclient.columns)
    del mdata
    gc.collect()
    
    print('\nMerge with prev months prods...')
    
    X = pd.merge(Xclient, prmdata, how='left', on='ncodpers')
    print(X.shape)
    print(X.head())
    # Fill products of new clients
    X.fillna(0, inplace=True)
    return X
    
def getAddedProducts(mdata, prevmdata):

    intsec = np.intersect1d(mdata.ncodpers, prevmdata.ncodpers)
    print(intsec.size)
    print(np.unique(intsec).size)
    
    print('\nMerge...')
    mgd = pd.merge(mdata, prevmdata, how='left', on='ncodpers')
    print(mgd.shape)
    mgd.fillna(0, inplace=True)
      
    added = pd.DataFrame(mgd.ncodpers)
    print(added.head())
    
    for i, pr in enumerate(targetprods):
        # Difference between this and previous month
        # 0: no change in product, 1: added product, -1: removed product
        #added[pr] = mgd.iloc[:, i+1] - mgd.iloc[:, i+25]
        added[pr] = mgd.loc[:, pr + '_x'] - mgd.loc[:, pr + '_y']
        # Consider only added products
        added.loc[added[pr] == -1, pr] = 0
    
    print(added.head())
    print('Total added products')
    print(added.sum(axis=0))
    
    return added.drop(['ncodpers'], axis=1)
    
def getLaggedFeatures(mgd, molist):
    for mo in molist:
        print(mo)
        lagmodata = getDataByMonth(dates, [mo], ['ncodpers'], prods)
        print(lagmodata.shape)

        print('\nMerging lagged month (' +str(mo) + ')...')
        i = molist.index(mo)
        mgd = pd.merge(mgd, lagmodata, how='left', on='ncodpers', 
                                                    suffixes=[i, i+1])
        print(mgd.shape)
        print(mgd.info())
        mgd.fillna(0, inplace=True)
        print(mgd.info())
              
    print(mgd.info())
    print(mgd.head())
        
    return mgd    

# Load data
print('\nLoading dates...')
dates = pd.read_csv('train_ver2.csv', usecols=['fecha_dato'], header=0)
print('Dates')
print(dates.head())

# Client features
thismonth = '2015-06-28' # '2016-05-28'
prevmonth = months[months.index(thismonth) - 1]
print('\nThis month: %s. Previous month: %s' % (thismonth, prevmonth))

mdata = getDataByMonth(dates, [thismonth], clientfeatures, prods)
prevmdata = getDataByMonth(dates, [prevmonth], ['ncodpers'], prods)
print(mdata.head())
print(prevmdata.head())

print('Get train data (this month client features + prev month prods')

X = getClientFeatures(mdata[clientfeatures], prevmdata)
print(X.head())

print('\nLagged features...')
lag1month = months[months.index(prevmonth) - 1]
lag2month = months[months.index(lag1month) - 1]
lag3month = months[months.index(lag2month) - 1]
lag4month = months[months.index(lag3month) - 1]
lag5month = months[months.index(lag4month) - 1]
lag6month = months[months.index(lag5month) - 1]
lag7month = months[months.index(lag6month) - 1]
# List of months used to get lagged product features
lagmonths = [lag1month, lag2month, lag3month, lag4month] #, lag5month]
                           #lag6month, lag7month]
print('Lagged months: ' + str(lagmonths))
X = getLaggedFeatures(X, lagmonths)
print('X shape {}'.format(X.shape))
print(X.head())

# Remove codpers
X.drop(['ncodpers'], axis=1, inplace=True)

print('\nAdded products (targets)')
y = getAddedProducts(mdata[['ncodpers']+prods], prevmdata)
print('y shape {}'.format(y.shape))
print(y.values.sum()/y.size)
print(y[:5])

del mdata, prevmdata
gc.collect()

print('Training and validation sets')
# Create test set out of 20% samples
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)                                                      
Xtrain = X
ytrain = y    
print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
del X, y
gc.collect()

print('Select only clients with added products')
addedprods= np.sum(ytrain, axis=1)
Xtrain = Xtrain[addedprods!=0]
ytrain = ytrain[addedprods!=0]
print(Xtrain.shape, ytrain.shape)

######################## Train model #########################
print('\nTraining...')
clfdict = {}
probs = []
freq = ytrain.sum(axis=0)
for pr in targetprods:
    print(pr)
    clf = xgboost.XGBClassifier(max_depth=6, 
                                learning_rate = 0.05, 
                                subsample = 0.9, 
                                colsample_bytree = 0.9, 
                                n_estimators=100,
                                base_score = freq[pr]/Xtrain.shape[0], nthread=2)
    clfdict[pr] = clf
    clf.fit(Xtrain, ytrain.loc[:, pr])
#    ypredv = clf.predict(Xval)
#    probs.append(clf.predict_proba(Xval)[:, 1])


###################### Test predictions ###########################

print('\nTest predictions...')
testmonth = '2016-06-28'
prtmonth = '2016-05-28'
print('\nTest month: %s. Previous test month: %s' % (testmonth, prtmonth))

del Xtrain, Xval, ytrain, yval
gc.collect()

tdata = pd.read_csv('test_ver2.csv', usecols=clientfeatures, header=0)
prtmdata = getDataByMonth(dates, [prtmonth], ['ncodpers'], prods)
print(tdata.head())
print(prtmdata.head())

print('Get test data (test month client features + prev month prods')

Xtest = getClientFeatures(tdata[clientfeatures], prtmdata)

print('Lagged test months')
lag1month = months[months.index(prtmonth) - 1]
lag2month = months[months.index(lag1month) - 1]
lag3month = months[months.index(lag2month) - 1]
lag4month = months[months.index(lag3month) - 1]
lag5month = months[months.index(lag4month) - 1]

lagtestmonths= [lag1month, lag2month, lag3month, lag4month] #, lag5month]
print(lagtestmonths)
Xtest = getLaggedFeatures(Xtest, lagtestmonths)
         
tids = Xtest['ncodpers']

Xtest.drop(['ncodpers'], axis=1, inplace=True)

print(Xtest.shape)  
print(Xtest.head())
del tdata
del prtmdata
gc.collect()

print('Prediction list...')
tclfdict = clfdict

testprobs = []
for pr in targetprods:
    print(pr)
    testprobs.append(tclfdict[pr].predict_proba(Xtest)[:, 1])

testprobs = np.array(testprobs).T
print(testprobs.shape)
test_probs_df = pd.DataFrame(testprobs,columns = targetprods)
test_probs_df.to_csv("model2_prob.csv",index=False)
        
        
print('Creating list of most probable products...')
idsort7 = np.argsort(testprobs, axis=1)[:, :-8:-1] # ids of seven greatest probs
predlist = [[prodict[j] for j in irow] for irow in idsort7]

print('Output Generation...')
subname = 'Santander_predictions.csv'
sub = tids.to_frame()
sub['added_products'] = np.array([' '.join(p) for p in predlist])
sub.to_csv(subname, index=False)