#!/usr/bin/python
import pandas as pd
import numpy as np

#RDkit for fingerprinting and cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

#MolVS for standardization and normalization of molecules
import molvs as mv

#SKlearn for metrics and datasplits
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score, roc_curve


#Read the data
data = pd.DataFrame.from_csv('tox21_10k_data_all_pandas.csv')
valdata = pd.DataFrame.from_csv('tox21_10k_challenge_test_pandas.csv')
testdata = pd.DataFrame.from_csv('tox21_10k_challenge_score_pandas.csv')

testdata.to_csv('testdata_before.csv')
print "saved pandas"

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

data.to_csv('data.csv')
valdata.to_csv('valdata.csv')
testdata.to_csv('testdata.csv')


prop = 'SR-MMP'
#Choose property to model
print len(testdata)


#Convert to Numpy arrays
X_train = np.array(list(data[~(data[prop].isnull())][fp]))
np.save('Xtrains', X_train)
X_val = np.array(list(valdata[~(valdata[prop].isnull())][fp]))
np.save('Xvals', X_val)
X_test = np.array(list(testdata[~(testdata[prop].isnull())][fp]))
np.save('Xtests', X_test)

#Select the property values from data where the value of the property is not null and reshape
#y_train = data[~(data[prop].isnull())][prop].values.reshape(-1,1)
#y_val = valdata[~(valdata[prop].isnull())][prop].values.reshape(-1,1)
#y_test = testdata[~(testdata[prop].isnull())][prop].values.reshape(-1,1)


