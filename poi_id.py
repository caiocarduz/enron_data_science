
# coding: utf-8

# In[65]:


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import Imputer



# In[67]:


#define as principais features antes da remoção de outliers e null values.

import matplotlib.pyplot as plt
feature_list_1 =['poi',
                'bonus',
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person', 'long_term_incentive', 'salary']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict_1 = pickle.load(data_file)


#CreateExercisedStockRatio(data_dict_1, feature_list_1)

# funcao que classifica as features quanto sua importância############################
def bestfeatures(data_dict_1, feature_list_1):
    selector = SelectKBest(f_classif, k = len(feature_list_1)-1)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict_1
### Extract features and labels from dataset for local testing
    data_1 = featureFormat(my_dataset, feature_list_1, sort_keys = True)
    labels_1, features_1 = targetFeatureSplit(data_1)
# Example starting point. Try investigating other evaluation techniques!
    from sklearn.cross_validation import train_test_split
    features_train_1, features_test_1, labels_train_1, labels_test_1 = train_test_split(features_1, labels_1, test_size=0.01, random_state=42)
        
    selector.fit(features_train_1, labels_train_1)
    scores = -np.log10(selector.pvalues_)
    return selector 

######################################################################################
# descomentar para plot das Kbesst Selector###########################################
#plt.bar(range(len(feature_list_1[1:])), bestfeatures(data_dict_1, feature_list_1))
#plt.xticks(range(len(feature_list_1[1:])), feature_list_1[1:], rotation = 'vertical')
#plt.show()
#######################################################################################

scores_1 = bestfeatures(data_dict_1, feature_list_1).scores_
tuples = zip(feature_list_1[1:], scores_1)
k_best_features = sorted(tuples, key = lambda x: x[1], reverse = True)
kbest_inicial = k_best_features


# In[68]:


# estratégia de manipular e editar os dados por meio do pandas obtida aqui:
#https://olegleyz.github.io/enron_classifier.html
df = pd.DataFrame.from_dict(data_dict_1, orient = 'index')
df = df[feature_list_1]
df = df.replace('NaN', np.nan)
df.info()


# In[69]:


# Substitui 'NaN' por 0
# =============================================================================
df.ix[:,:13] = df.ix[:,:13].fillna(0)
df.ix[:,19:20] = df.ix[:,19:20].fillna(0)
df.replace('inf',0)
df.info()
# =============================================================================


# In[70]:
#condiciona as medianas por poi.
#df['to_messages'].fillna(df.groupby(['poi'])['to_messages'].transform(np.median), inplace =True)
#df['from_messages'].fillna(df.groupby(['poi'])['from_messages'].transform(np.median), inplace =True)
#df['from_this_person_to_poi'].fillna(df.groupby(['poi'])['from_this_person_to_poi'].transform(np.median), inplace =True)
#df['from_poi_to_this_person'].fillna(df.groupby(['poi'])['from_poi_to_this_person'].transform(np.median), inplace =True)




df['to_messages'] = df['to_messages'].fillna(df['to_messages'].mean())
df['from_messages'] = df['from_messages'].fillna(df['from_messages'].mean())
df['from_this_person_to_poi'] = df['from_this_person_to_poi'].fillna(df['from_this_person_to_poi'].mean())
#df['long_term_incentive'] = df['long_term_incentive'].fillna((df['long_term_incentive'].median()))
df['long_term_incentive'] = df['long_term_incentive'].fillna(0)
#df['from_poi_to_this_person'] = df['from_poi_to_this_person'].fillna((df['from_poi_to_this_person'].mean()))
##df['salary'] = df['salary'].fillna((df['salary'].median()))
df['salary'] = df['salary'].fillna(0)
df.replace(np.inf,0, inplace= True)
df.replace(np.nan, 0, inplace = True)
#
#
###cria nova feature
df['poi_to_email'] = df['from_this_person_to_poi']/df['to_messages']
df['poi_from_email'] = df['from_poi_to_this_person']/df['from_messages']
#df['medio_poi_ratio'] = (df['from_this_person_to_poi']/df['to_messages'] * df['from_poi_to_this_person']/df['from_messages'])/2
df.info()


# In[71]:

# lista as features mais relevantes no metodo Kselector.
feature_list_2 = ['poi',
                'bonus',
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person', 'long_term_incentive', 'salary', 'poi_to_email','poi_from_email']


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
 # You will need to use more features
    
my_dataset = df[feature_list_2].to_dict(orient = 'index')

### Task 2: Remove outliers

my_dataset.pop("TOTAL", 0)


### define as melhores variaveis pelo metodo Kselector. 

#bestfeatures(my_dataset, feature_list_1)

# Para plotar os Scores do Kbest Selector descomente abaixo ###########################
#plt.bar(range(len(feature_list_1[1:])), bestfeatures(my_dataset, feature_list_1))
#plt.xticks(range(len(feature_list_1[1:])), feature_list_1[1:], rotation = 'vertical')
#plt.show()

#######################################################################################



scores_2 = bestfeatures(my_dataset, feature_list_2).scores_
tuples = zip(feature_list_2[1:], scores_2)
k_best_features = sorted(tuples, key = lambda x: x[1], reverse = True)
kbest_tratado = k_best_features


# In[72]:


# inspirado no código oferecido em https://www.kaggle.com/grfiv4/plotting-feature-importances

#Decision tree using features with non-null importance
clf = DecisionTreeClassifier(random_state = 75)
clf.fit(df.ix[:,1:], df.ix[:,:1])
dftrain = df.ix[:,1:]
top_n = 20

# show the features with non null importance, sorted and create features_list of features for the model
feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
feat_imp['feature'] = dftrain.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.iloc[:top_n]
    
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title="features importance", figsize=(8,8))
plt.xlabel('Feature Importance Score')
plt.show()


# In[73]:



## descomentar caso for utilizar o metodo pipeline.#####################################

#features_list = feature_list_2

########################################################################################
features_list = ['poi',
                'other',
                'expenses', 
                'total_payments',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person', 'long_term_incentive','poi_to_email','bonus',
                'restricted_stock', 'exercised_stock_options','salary']

# =============================================================================
# features para classificação com data wrangling condicionado 
# #features_list = ['poi',
# #                 'other','medio_poi_ratio', 'to_messages'
# #                ,
# #                'poi_to_email', 'bonus','total_payments',
# #           
# #                'salary']
# 
# =============================================================================


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


##### Provided to give you a starting point. Try a variety of classifiers.#############

clf = DecisionTreeClassifier(class_weight = {1:40,0:6},min_samples_split = 2,random_state = 29,max_depth = 3)

#### descomentar caso for utilizar o metodo pipeline. Descomentar somente um classificador por vez.#####
#clff = GaussianNB()
#clff = RandomForestClassifier(n_estimators = 3, random_state = 29,\
#                             min_samples_split = 2, max_depth = 7, class_weight = {1:40,0:6})  # {1:6.5,0:6})
#clff = neighbors.kNeighborsClassifier(n_neighbors = 6)
#clff = linear_model.LogisticRegression( C=1e5)
#clff = KMeans(n_clusters =2)
#clff = SVC()
########################################################################################################

####### Para pipeline descomentar as variaveis abaixo #######################################
#pca1 = PCA(n_components = 6)
#selector = SelectKBest(f_classif, k = 3)
#scaler = MinMaxScaler()
#clf = Pipeline([("PCA", pca1), ('clf', clff)])
##############################################################################################

### Pra GridSearch descomentar estimator, parameters e clf ###################################
#estimators = [('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]

#parameters = {'reduce_dim__n_components': [1, 2], 'clf__min_samples_split': [2, 4, 6, 8]}

#clf = GridSearchCV(Pipeline(estimators), parameters, scoring ="average_precision")

##############################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

####  GridSearchCV metricas ###################################################

#clf.fit(features_train, labels_train)
#print clf.cv_results_
#print clf.best_estimator_
#print clf.best_score_
#print clf.best_params_
#y_pred = clf.predict(features_test)
#print (len(features_test))
#print (accuracy_score(labels_test, y_pred))
#print (recall_score(labels_test, y_pred))
#print (confusion_matrix(labels_test, y_pred))
#print(metrics.classification_report(labels_test, y_pred)) 

#################################################################################


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

