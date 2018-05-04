#Pandas and Numpy
import pandas as pd
import numpy as np
 
#RDkit for fingerprinting and cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
 
#MolVS for standardization and normalization of molecules
import molvs as mv

#Keras for deep learning
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import WeightRegularizer
from keras.optimizers import SGD
 
#SKlearn for metrics and datasplits
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score, roc_curve
 
#Matplotlib for plotting
from matplotlib import pyplot as plt

#!/usr/bin/python
from rdkit import Chem
import pandas as pd


def dataprep():
    filename = "tox21_10k_data_all.sdf"
    basename = filename.split(".")[0]
    collector = []
    sdprovider = Chem.SDMolSupplier(filename)
    for i,mol in enumerate(sdprovider):
        try:
            moldict = {}
            moldict['smiles'] = Chem.MolToSmiles(mol)
            #Parse Data
            for propname in mol.GetPropNames():
                moldict[propname] = mol.GetProp(propname)
            collector.append(moldict)
        except:
            print "Molecule %s failed"%i

    data = pd.DataFrame(collector)
    data.to_csv(basename + '_pandas.csv')

    filename = "tox21_10k_challenge_test.sdf"
    basename = filename.split(".")[0]
    collector = []
    sdprovider = Chem.SDMolSupplier(filename)
    for i,mol in enumerate(sdprovider):
        try:
            moldict = {}
            moldict['smiles'] = Chem.MolToSmiles(mol)
            #Parse Data
            for propname in mol.GetPropNames():
                moldict[propname] = mol.GetProp(propname)
            collector.append(moldict)
        except:
            print "Molecule %s failed"%i

    data = pd.DataFrame(collector)
    data.to_csv(basename + '_pandas.csv')

    filename = "tox21_10k_challenge_score.sdf"
    basename = filename.split(".")[0]
    collector = []
    sdprovider = Chem.SDMolSupplier(filename)
    for i,mol in enumerate(sdprovider):
        try:
            moldict = {}
            moldict['smiles'] = Chem.MolToSmiles(mol)
            #Parse Data
            for propname in mol.GetPropNames():
                moldict[propname] = mol.GetProp(propname)
            collector.append(moldict)
        except:
            print "Molecule %s failed"%i

    data = pd.DataFrame(collector)
    data.to_csv(basename + '_pandas.csv')




#Read the data
data = pd.DataFrame.from_csv('tox21_10k_data_all_pandas.csv')
valdata = pd.DataFrame.from_csv('tox21_10k_challenge_test_pandas.csv')
testdata = pd.DataFrame.from_csv('tox21_10k_challenge_score_pandas.csv')

#Function to get parent of a smiles
def parent(smiles):
 st = mv.Standardizer() #MolVS standardizer
 try:
  mols = st.charge_parent(Chem.MolFromSmiles(smiles))
  return Chem.MolToSmiles(mols)
 except:
  print "%s failed conversion"%smiles
  return "NaN"
#Clean and standardize the data
def clean_data(data):
 #remove missing smiles
 data = data[~(data['smiles'].isnull())]
 #Standardize and get parent with molvs
 data["smiles_parent"] = data.smiles.apply(parent)
 data = data[~(data['smiles_parent'] == "NaN")]
 #Filter small fragents away
 def NumAtoms(smile):
  return Chem.MolFromSmiles(smile).GetNumAtoms()
 data["NumAtoms"] = data["smiles_parent"].apply(NumAtoms)
 data = data[data["NumAtoms"] > 3]
 return data

data = clean_data(data)
valdata = clean_data(valdata)
testdata = clean_data(testdata)

#Calculate Fingerprints
def morgan_fp(smiles):
 mol = Chem.MolFromSmiles(smiles)
 fp = AllChem.GetMorganFingerprintAsBitVect(mol,3, nBits=8192)
 npfp = np.array(list(fp.ToBitString())).astype('int8')
 return npfp

fp = "morgan"
data[fp] = data["smiles_parent"].apply(morgan_fp)
valdata[fp] = valdata["smiles_parent"].apply(morgan_fp)
testdata[fp] = testdata["smiles_parent"].apply(morgan_fp)


prop = 'SR-MMP'
#Choose property to model
print testdata
print testdata[prop]

#Convert to Numpy arrays
X_train = np.array(list(data[~(data[prop].isnull())][fp]))
X_val = np.array(list(valdata[~(valdata[prop].isnull())][fp]))
X_test = np.array(list(testdata[~(testdata[prop].isnull())][fp]))
 
#Select the property values from data where the value of the property is not null and reshape
y_train = data[~(data[prop].isnull())][prop].values.reshape(-1,1)
y_val = valdata[~(valdata[prop].isnull())][prop].values.reshape(-1,1)
y_test = testdata[~(testdata[prop].isnull())][prop].values.reshape(-1,1)

#Set network hyper parameters
l1 = 0.000
l2 = 0.016
dropout = 0.5
hidden_dim = 80
 
#Build neural network
model = Sequential()
model.add(Dropout(0.2, input_shape=(X_train.shape[1],)))
for i in range(3):
 wr = WeightRegularizer(l2 = l2, l1 = l1) 
 model.add(Dense(output_dim=hidden_dim, activation="relu", W_regularizer=wr))
 model.add(Dropout(dropout))
wr = WeightRegularizer(l2 = l2, l1 = l1) 
model.add(Dense(y_train.shape[1], activation='sigmoid',W_regularizer=wr))
 
##Compile model and make it ready for optimization
model.compile(loss='binary_crossentropy', optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True), metrics=['binary_crossentropy'])
#Reduce lr callback
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
 
#Training
history = model.fit(X_train, y_train, nb_epoch=1000, batch_size=1000, validation_data=(X_val,y_val), callbacks=[reduce_lr])


#Plot Train History
def plot_history(history):
    lw = 2
    fig, ax1 = plt.subplots()
    ax1.plot(history.epoch, history.history['binary_crossentropy'],c='b', label="Train", lw=lw)
    ax1.plot(history.epoch, history.history['val_loss'],c='g', label="Val", lw=lw)
    plt.ylim([0.0, max(history.history['binary_crossentropy'])])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(history.epoch, history.history['lr'],c='r', label="Learning Rate", lw=lw)
    ax2.set_ylabel('Learning rate')
    plt.legend()
    plt.show()
 
plot_history(history)

def show_auc(model):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
 
    auc_train = roc_auc_score(y_train, pred_train)
    auc_val = roc_auc_score(y_val, pred_val)
    auc_test = roc_auc_score(y_test, pred_test)
    print "AUC, Train:%0.3F Test:%0.3F Val:%0.3F"%(auc_train, auc_test, auc_val)
 
    fpr_train, tpr_train, _ =roc_curve(y_train, pred_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, pred_val)
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_test)
 
    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='b',lw=lw, label='Train ROC (area = %0.2f)'%auc_train)
    plt.plot(fpr_val, tpr_val, color='g',lw=lw, label='Val ROC (area = %0.2f)'%auc_val)
    plt.plot(fpr_test, tpr_test, color='r',lw=lw, label='Test ROC (area = %0.2f)'%auc_test)
 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of %s'%prop)
    plt.legend(loc="lower right")
    plt.interactive(True)
    plt.show()
 
show_auc(model)


#Compare with a Linear model
from sklearn import linear_model
#prepare scoring lists
fitscores = []
predictscores = []
##prepare a log spaced list of alpha values to test
alphas = np.logspace(-2, 4, num=10)
##Iterate through alphas and fit with Ridge Regression
for alpha in alphas:
  estimator = linear_model.LogisticRegression(C = 1/alpha)
  estimator.fit(X_train,y_train)
  fitscores.append(estimator.score(X_train,y_train))
  predictscores.append(estimator.score(X_val,y_val))
 
#show a plot
import matplotlib.pyplot as plt
ax = plt.gca()
ax.set_xscale('log')
ax.plot(alphas, fitscores,'g')
ax.plot(alphas, predictscores,'b')
plt.xlabel('alpha')
plt.ylabel('Correlation Coefficient')
plt.show()
 
estimator= linear_model.LogisticRegression(C = 1)
estimator.fit(X_train,y_train)
#Predict the test set
y_pred = estimator.predict(X_test)
print roc_auc_score(y_test, y_pred)

def deeptoxnn(l1 = 0.0, l2= 0.0, dropout = 0.0, dropout_in=0.0, hiddendim = 4, hiddenlayers = 3, lr = 0.001, nb_epoch=50, returnmodel=False):
    #Build neural network
    model = Sequential()
    model.add(Dropout(dropout_in, input_shape=(X_train.shape[1],)))
    for i in range(hiddenlayers):
        wr = WeightRegularizer(l2 = l2, l1 = l1)
        model.add(Dense(output_dim=hiddendim, activation="relu", W_regularizer=wr))
        model.add(Dropout(dropout))
    wr = WeightRegularizer(l2 = l2, l1 = l1)
    model.add(Dense(y_train.shape[1], activation='sigmoid',W_regularizer=wr))
    ##Compile model and make it ready for optimization
    model.compile(loss='binary_crossentropy', optimizer = SGD(lr=lr, momentum=0.9, nesterov=True), metrics=['binary_crossentropy'])
    #Reduce lr callback
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=10, min_lr=0.000001, verbose=returnmodel)
    #Save best model (Early stopping)
    modelcp = ModelCheckpoint("tempmodel.h5", monitor='val_loss', verbose=returnmodel, save_best_only=True)
    #Training
    history = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=1000, validation_data=(X_val,y_val), callbacks=[reduce_lr, modelcp], verbose=returnmodel)
    loss = history.history["val_loss"][-1]
    if returnmodel:
        return loss,model, history
    else:
        return loss

def fit_nn_val(x):
    x = np.atleast_2d(x)# Must take a 2 D array with parms
    fs = np.zeros((x.shape[0],1)) #prepare return array with similar return dimension
    for i in range(x.shape[0]):
        val_loss = deeptoxnn(l2=float(x[i,0]),dropout=float(x[i,1]), dropout_in=float(x[i,2]),lr=float(x[i,3]), hiddendim=int(x[i,4]),hiddenlayers=int(x[i,5]))
        fs[i] = np.log(val_loss)
    print val_loss, fs
    return fs

#Discrete Variable must be at the end
mixed_domain =[ {'name': 'l2', 'type': 'continuous', 'domain': (0.0,0.07)},
        {'name': 'dropout', 'type': 'continuous', 'domain': (0.0,0.7)},
        {'name': 'dropout_in', 'type': 'continuous', 'domain': (0.0,0.5)},
        {'name': 'lr', 'type': 'continuous', 'domain': (0.0001,0.1)},
               {'name': 'hiddendim', 'type': 'discrete', 'domain': [2**x for x in range(2,7)]},
               {'name': 'hiddenlayers', 'type': 'discrete', 'domain': range(1,3)}]

myBopt = BayesianOptimization(f=fit_nn_val, # function to optimize
                              domain=mixed_domain, # box-constrains of the problem
                              initial_design_numdata = 50,# number data initial design
                  			  model_type="GP_MCMC",         
                              acquisition_type='EI_MCMC', #EI
                  			  evaluator_type="predictive",  # Expected Improvement
                  			  batch_size = 1,
                              num_cores = 4,
                              exact_feval = False)    # May not always give exact results

myBopt.run_optimization(max_iter = 100)
x_best = myBopt.x_opt #myBopt.X[np.argmin(myBopt.Y)]
print x_best
myBopt.plot_convergence()

val_loss, model, history = deeptoxnn(l2=6.81e-3, dropout=1.76e-2, dropout_in=9.99e-2,lr=8.24e-02, nb_epoch = 200, hiddendim=8,hiddenlayers=1, returnmodel=True)
#Reporting
plot_history(history)
show_auc(model)