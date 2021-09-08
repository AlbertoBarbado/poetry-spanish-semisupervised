# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:49:20 2021

@author: alber
"""

import re
import os
import pandas as pd
import numpy as np
import spacy
import pickle
import lightgbm as lgb
import imblearn

from sklearn import preprocessing
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import minmax_scale
# from sentence_transformers import SentenceTransformer, util
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from statsmodels.stats.inter_rater import cohens_kappa
from common.tools import get_files, file_presistance
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier
from common.config import (
    PATH_POEMS, PATH_RESULTS, PATH_AFF_LEXICON, PATH_GROUND_TRUTH
    )
from itertools import product
nlp = spacy.load("es_core_news_md")
stemmer = SnowballStemmer("spanish")


def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

# MAPE metric
def mape(y_real, y_pred):
    """
    TODO

    Parameters
    ----------
    y_real : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y_real, y_pred = np.array(y_real), np.array(y_pred)
    return np.mean(np.abs((y_real - y_pred) / y_real))

def _getReport(
        y_test, y_pred, y_pred_proba, target_names, using_affective = "yes",
        semantic_model = "", used_model = ""
        ):
    """
    TODO

    Parameters
    ----------
    y_test : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    target_names : TYPE
        DESCRIPTION.
    using_affective : TYPE, optional
        DESCRIPTION. The default is "yes".
    semantic_model : TYPE, optional
        DESCRIPTION. The default is "".
    used_model : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    df_metrics_iter : TYPE
        DESCRIPTION.

    """
   
    ### 1. Base Dataframe
    df_metrics_iter = pd.DataFrame(
        {
            'category': [category],
            'using_affective': [using_affective],
            'semantic_model': [semantic_model],
            'regression_model': [used_model]
           
            }
        )
   
    ### 2. Cohen's Kappa
    # Make Dataframe
    y_pred_cat = np.round(y_pred).astype(int)
    y_pred_cat = np.array([x if x>0 else 1 for x in y_pred_cat])
    df = pd.DataFrame({"A": y_test.astype(int), "B": y_pred_cat})
   
    # Switch it to three columns A's answer, B's answer and count of that combination
    df = df.value_counts().reset_index()
   
    l1 = [1,2,3,4]
    l2 = [1,2,3,4]
    df_aux = pd.DataFrame(list(product(l1, l2)), columns=['A', 'B'])
   
    # Check compliance
    if len(df) < len(df_aux):
        df = df.merge(df_aux, how="outer").fillna(0)
   
    # Make square
    square = df.pivot(columns="A",index="B").fillna(0).values
   
    # Get Kappa
    dct_kappa = cohens_kappa(square)
    kappa_max = dct_kappa['kappa_max']
    kappa = dct_kappa['kappa']
    df_metrics_iter['kappa'] = [kappa]
    df_metrics_iter['kappa_max'] = [kappa_max]
   
    ### Get Correlations
    df_corr = pd.DataFrame({
        'y_pred': y_pred,
        'y_test': y_test
        })
    df_corr = df_corr.corr().reset_index()
    df_corr = df_corr.round(2)
    corr = df_corr.values[0][2]
    df_metrics_iter['corr'] = [corr]
   
    ### 3. R2 & MAPE
    r2_result = r2_score(y_test, y_pred)
    mape_result = mape(y_test, y_pred)
    mae_result = mean_absolute_error(y_test, y_pred)
   
    df_metrics_iter['r2_result'] = [r2_result]
    df_metrics_iter['mape_result'] = [mape_result]  
    df_metrics_iter['mae'] = [mae_result]
   
    ### 3. AUC
    if len(y_pred_proba)>0:
       
        ## P, R, F1 metrics
        report = classification_report(
            y_test, y_pred,  output_dict = True
        )
        df_metrics_iter['precision_weighted'] = [report['weighted avg']['precision']]
        df_metrics_iter['recall_weighted'] = [report['weighted avg']['recall']]
        df_metrics_iter['f1_weighted'] = [report['weighted avg']['f1-score']]
       
        ## Multiclass AUC
        # Encoding
        y_test_c = pd.get_dummies(y_test)
        y_test_c = y_test_c.astype(int)
        y_test_c.columns = [int(x)-1 for x in list(y_test_c.columns)]
        y_pred_proba_c = y_pred_proba[y_test_c.columns].copy()
       
        # y_test_c = pd.get_dummies(y_test)
        # y_pred_c = pd.get_dummies(y_pred)
        # cols_ref = list(set(list(y_test_c.columns)+list(y_pred_c.columns)))
        # cols_ref = [1.0, 2.0, 3.0, 4.0]
        # for column in cols_ref:
        #     if column not in list(y_test_c.columns):
        #         y_test_c[column] = 0
        #     if column not in list(y_pred_c.columns):
        #         y_pred_c[column] = 0
        # y_test_c = y_test_c[cols_ref].astype(int)
        # y_pred_c = y_pred_c[cols_ref].astype(int)
       
        # Get Multiclass AUC
        auc = roc_auc_score(y_test_c, y_pred_proba_c, average='weighted')
        df_metrics_iter['auc'] = auc
   
        # y_pred_proba = np.asarray([x if str(x) != 'nan'  else 0.0 for x in y_pred_proba])
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_test, y_pred_proba, pos_label=1
        #     )
        # auc = metrics.auc(fpr, tpr)
        # df_metrics_iter['auc'] = [auc]
   
    return df_metrics_iter

# =============================================================================
# 1. Prepare Data
# =============================================================================
### Load Sonnets Features
# Load Data
file_to_read = open(f"{PATH_RESULTS}/dct_sonnets_input_v5", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

# Only DISCO
if False:
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x <= 4085}

# Sonnet Matrix
list_original_sentence = [
    'enc_text_model1',
    'enc_text_model2',
    'enc_text_model3',
    'enc_text_model4',
    'enc_text_model5'
    ]
list_semantic_models = [
    'enc_text_model1',
    'enc_text_model2',
    'enc_text_model3',
    'enc_text_model4',
    'enc_text_model5',
    # 'enc_text_model_hg_bert_max',
    # 'enc_text_model_hg_bert_span',
    # 'enc_text_model_hg_bert_median',
    'enc_text_model_hg_bert_avg_w',
    # 'enc_text_model_hg_bert_sp_max',
    # 'enc_text_model_hg_bert_sp_span',
    # 'enc_text_model_hg_bert_sp_median',
    'enc_text_model_hg_bert_sp_avg_w',
    # 'enc_text_model_hg_ro_max',
    # 'enc_text_model_hg_ro_span',
    # 'enc_text_model_hg_ro_median',
    # 'enc_text_model_hg_ro_avg_w'
    ]

# General Variables
dct_metrics_all_models = {}
df_meta = pd.concat(
    [
        pd.DataFrame({"index": [item["index"]], "text": [item["text"]]})
        for key, item in dct_sonnets.items()
    ]
)
df_affective = pd.concat([
    item["aff_features"] for key, item in dct_sonnets.items()]).fillna(0)

# Load psycho names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv",
    encoding="latin-1"
    )
list_names = list(df_names["es_name"].values)
list_aff = [
    "concreteness",
    "context availability",
    "anger",
    "arousal",
    "disgust",
    "fear",
    "happinness",
    "imageability",
    "sadness",
    "valence",
]

### Load Ground Truth
if False:
    df_gt = pd.read_csv(f"{PATH_GROUND_TRUTH}/poems_corpus_all.csv")
    df_gt = df_gt.rename(columns={"text": "text_original"})
    df_gt.columns = [str(x).rstrip().lstrip() for x in list(df_gt.columns)]
   
    ### Get Subsample from GT
    df_add = pd.DataFrame()
    for category in list_names:
        if category in list_aff:
            continue
        try:
            df_iter = df_gt.groupby(category).apply(lambda s: s.sample(2))
        except:
            continue
        df_add = df_add.append(df_iter)
    df_add = df_add.drop_duplicates()
   
    # New GT (without data used in training)
    df_gt = df_gt[~df_gt["index"].isin(df_add["index"])].copy()
   
    df_add.to_csv("train_dataset.csv", index=False)
    df_gt.to_csv("test_dataset.csv", index=False)
else:
    df_add = pd.read_csv("train_dataset.csv")
    df_gt = pd.read_csv("test_dataset.csv")
   
list_use_aff = [True, False]
   
# =============================================================================
# 2. Iter and get models
# =============================================================================
# semantic_model = "enc_text_model_hg_ro_avg_w"
dct_eq = {
     'concreteness': 'concreteness_mean',
     'context availability': 'cont_ava_mean',
     'anger': 'anger_mean',
     'arousal': 'arousal_mean',
     'disgust': 'disgust_mean',
     'fear': 'fear_mean',
     'happinness': 'happiness_mean',
     'imageability': 'imageability_mean',
     'sadness': 'sadness_mean',
     'valence': 'valence_mean'
    }

def _change_scale(x):
    if x[f'{feature_iter}'] <= 2:
        return 1
    elif x[f'{feature_iter}'] <= 3.5:
        return 2
    elif x[f'{feature_iter}'] <= 5.5:
        return 3
    else:
        return 4
   
### 1. Baseline Method
### Base model - use XXX_mean rescaling it    
df_metrics = pd.DataFrame()
for category in list_aff:      
    # Rescale data
    feature_iter = dct_eq[category]
    df_scaled = df_affective[['index', f'{feature_iter}']].copy()
   
    if True:
        df_scaled[f'{feature_iter}'] = list(
            minmax_scale(df_scaled[f'{feature_iter}'],
                          feature_range=(1,4))
            .round()
            )
    else:
        df_scaled['new'] = df_scaled.apply(lambda x: _change_scale(x), axis=1)
        df_scaled[f'{feature_iter}'] = df_scaled['new']
   
    df_scaled = df_scaled.merge(df_gt[['index', category]], how="inner")
    df_corr = df_scaled.drop(columns=['index']).corr().reset_index()
   
    # Get "prediction"
    y_pred = pd.Series(list(
        df_scaled[df_scaled['index'].isin(df_gt['index'])][f'{feature_iter}']
        .values
        ))
   
    # Get test metric
    y_test = pd.Series(list(df_scaled[category].values))
   
    # Evaluate results
    df_metrics_iter = _getReport(
        y_test, y_pred, "", "", using_affective = "yes",
        semantic_model = "None", used_model = "baseline_affective"
        )
    df_metrics = df_metrics.append(df_metrics_iter)
   
# Save Results
df_metrics.to_csv(
    f"{PATH_RESULTS}/emotion_aff/df_metrics_baseline_model.csv",
    index=False
    )

"""
df_previous = pd.read_csv("datasets/affective-features-disco/df_features_sonnet.csv")
df_previous = df_previous[df_previous['index'].isin(df_gt['index'])]

x_labels = ['happiness_mean', 'sadness_mean', 'anger_mean',
            'fear_mean', 'disgust_mean', 'concreteness_mean',
            'imageability_mean', 'cont_ava_mean']
y_labels = ['happinness', 'sadness', 'anger', 'fear', 'disgust', 'concreteness',
            'imageability', 'context availability']
df_corr = df_previous[x_labels + y_labels].corr().reset_index()
df_corr = df_corr.drop(columns = y_labels)[~df_corr['index'].isin(x_labels)]
df_corr.index = df_corr['index']
df_corr = df_corr.drop(columns=['index']).round(2)
"""

### Iters
for use_aff in list_use_aff:
    for semantic_model in list_semantic_models:
        print("*"*50)
        print()
        print("*"*50)
        print(f"Semantic Model: {semantic_model}")
       
        if semantic_model in list_original_sentence:
            df_semantic = pd.concat(
                [pd.DataFrame(item[semantic_model]).T for key, item in dct_sonnets.items()]
            )
        else:
            df_semantic = pd.concat(
                [item[semantic_model] for key, item in dct_sonnets.items()]
            )
           
        df_X = df_meta.copy().reset_index(drop=True)
        df_X[df_semantic.columns] = df_semantic.reset_index(drop=True).astype(float)
        if use_aff:
            df_X[df_affective.columns] = df_affective.reset_index(drop=True)
       
        # Remove weirds spaces
        df_X.columns = [str(x).rstrip().lstrip() for x in list(df_X.columns)]
       
        # List Train/Test Columns
        list_train_cols = list(df_X.drop(columns=["index", "text"]).columns)
       
        # Test Data
        df_test = df_X.merge(df_gt, how="inner")  

        # Train Data
        df_add_train = df_X.merge(df_add, how="inner")
        df_train = (
            df_X[
                (~df_X["index"].isin(df_add_train["index"])) &
                (~df_X["index"].isin(df_test["index"]))
                ]
            .append(df_add_train)
            )
   
        # Scale data
        df_train_scaled = df_train.copy()
        df_test_scaled = df_test.copy()
           
        if use_aff:
            df_train_scaled = df_train_scaled.reset_index(drop=True)
            df_test_scaled = df_test_scaled.reset_index(drop=True)
           
            scaler = preprocessing.StandardScaler()
            list_aff_cols = list(df_affective.drop(columns=['index']).columns)
            x = scaler.fit_transform(df_train_scaled[list_aff_cols].values)
            df_train_scaled[list_aff_cols] = x
           
            x = scaler.transform(df_test_scaled[list_aff_cols].values)
            df_test_scaled[list_aff_cols] = x
           
        ### Train/Test Matrices
        X_train = df_train_scaled[list_train_cols].copy()
        X_test = df_test_scaled[list_train_cols].copy()
       
        print(f"Length annotated: {len(df_gt)+len(df_add)} | Length check: {len(df_test) + len(df_add_train)}")
   
        print(
            f"Length all: {len(dct_sonnets)} | Length check: {len(df_test) + len(df_train)}"
        )
        print(
            f"Train: {df_train.shape} | Test: {df_test.shape}"
        )
         
        # ===================================================================
        # 2. ML Models
        # ===================================================================
        # category = list_names[1]
       
        ### Iter
        target_names = [1,2,3,4]
        df_metrics = pd.DataFrame()
        j = 0
        list_iter = [x for x in list_names if x in list_aff]
        for category in list_iter:
            print("*"*50)
            j += 1
            print(f"Iter {j}/{len(list_iter)} - {category}")
           
            ### Y Test
            y_test = df_test_scaled[category].copy()    
            target_names = []
           
            ### Baseline model - No semisupervision
            # Dataset
            df_train_scaled_2 = (
                df_train[
                    df_train['index'].isin(list(df_add_train['index'].values))
                    ]
                [list_train_cols + [category]].copy()
                )
            X_train2 = df_train_scaled_2[list_train_cols].copy()
            y_train2 = df_train_scaled_2[category].copy()
           
            ## LightGBM Classification
            # Train Model
            clf = lgb.LGBMClassifier(is_unbalance=True)
            clf.fit(X_train2, y_train2)
           
            # Evaluate Model
            y_pred = clf.predict(X_test)
            y_pred_proba = pd.DataFrame(clf.predict_proba(X_test))
            print(classification_report(y_test, y_pred))
           
            used_model = "class_baseline_lightgbm"
            df_metrics_iter = _getReport(
                y_test,
                y_pred,
                y_pred_proba,
                [],
                using_affective = f"{use_aff}",
                semantic_model = semantic_model,
                used_model = used_model
                )
            df_metrics = df_metrics.append(df_metrics_iter)
           
            ## LightGBM Clasification SMOTE
            # Oversampling
            try:
                oversample = SMOTE()
                X_train_over, y_train_over = oversample.fit_resample(
                    X_train2, y_train2
                    )
               
                # Train Model
                clf = lgb.LGBMClassifier()
                clf.fit(X_train_over, y_train_over)
               
                # Evaluate Model
                y_pred = clf.predict(X_test)
                y_pred_proba = pd.DataFrame(clf.predict_proba(X_test))        
                print(classification_report(y_test, y_pred))
               
                used_model = "class_baseline_smote_lightgbm"
                df_metrics_iter = _getReport(
                    y_test,
                    y_pred,
                    y_pred_proba,
                    [],
                    using_affective = f"{use_aff}",
                    semantic_model = semantic_model,
                    used_model = used_model
                    )
                df_metrics = df_metrics.append(df_metrics_iter)
            except:
                pass
           
            ## LightGBM Regression
            # Train Model
            clf = lgb.LGBMRegressor()
            clf.fit(X_train2, y_train2)
           
            # Evaluate Model
            y_pred = clf.predict(X_test)
           
            used_model = "reg_baseline_lightgbm"
            df_metrics_iter = _getReport(
                y_test,
                y_pred, [],
                target_names,
                using_affective = f"{use_aff}",
                semantic_model = semantic_model,
                used_model = used_model
                )
            df_metrics = df_metrics.append(df_metrics_iter)  
           
            # ===================================================================
            # Multiclass Classification Models
            # ===================================================================
            ### Semisupervised Model [Label Spreading]
            # category = list_names[1]
           
            # Mask Missing Labels
            # kernel = "rbf"
            for kernel in ['knn', 'rbf']:
                label_prop_model = LabelSpreading(kernel=kernel)
                labels = np.copy(df_train[[category]].fillna(-1)[category].values)
   
                # Train Model
                label_prop_model.fit(X_train, labels)
                labels_infer = label_prop_model.predict(X_train)
               
                df_labels = pd.DataFrame(
                    {
                    'labels': labels,
                    'labels_infer': labels_infer
                    }
                    )
                df_labels['labels_infer2'] = df_labels.apply(
                    lambda x: x['labels_infer'] if x['labels']==-1 else x['labels'],
                    axis=1
                    )
                labels_infer = list(df_labels['labels_infer2'].values)
               
                # Evaluate with label model
                y_pred = label_prop_model.predict(X_test)
                y_pred_proba = pd.DataFrame(
                    label_prop_model.predict_proba(X_test)).fillna(0)
               
                used_model = f"class_label_spreading_base_{kernel}"
                df_metrics_iter = _getReport(
                    y_test,
                    y_pred,
                    y_pred_proba,
                    target_names,
                    using_affective = f"{use_aff}",
                    semantic_model = semantic_model,
                    used_model = used_model
                    )
                df_metrics = df_metrics.append(df_metrics_iter)
   
                ### Classification Model
                # Dataset
                y_train = labels_infer
               
                # Train Model
                clf = lgb.LGBMClassifier(
                    is_unbalance=True,
                    objective="multiclassova"
                    )
                clf.fit(X_train, y_train)
               
                # Evaluate Model
                y_pred = clf.predict(X_test)
                y_pred_proba = pd.DataFrame(clf.predict_proba(X_test)).fillna(0)
               
                used_model = f"class_label_spreading_lightgbm_{kernel}"
                df_metrics_iter = _getReport(
                    y_test,
                    y_pred,
                    y_pred_proba,
                    target_names,
                    using_affective = f"{use_aff}",
                    semantic_model = semantic_model,
                    used_model = used_model
                    )
                df_metrics = df_metrics.append(df_metrics_iter)    
               
                # Oversampling
                try:
                    oversample = SMOTE()
                    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
                   
                    # Train Model
                    clf = lgb.LGBMClassifier(objective="multiclassova")
                    clf.fit(X_train_over, y_train_over)
                   
                    # Evaluate Model
                    y_pred = clf.predict(X_test)
                    y_pred_proba = pd.DataFrame(clf.predict_proba(X_test)).fillna(0)
                   
                    used_model = f"class_label_spreading_lightgbm_smote_{kernel}"
                    df_metrics_iter = _getReport(
                        y_test,
                        y_pred,
                        y_pred_proba,
                        target_names,
                        using_affective = f"{use_aff}",
                        semantic_model = semantic_model,
                        used_model = used_model
                        )
                    df_metrics = df_metrics.append(df_metrics_iter)
                except:
                    pass
           
            ### Self-Training Classification Model
            y_train = labels
            clf = lgb.LGBMClassifier(
                is_unbalance=True,
                objective="multiclassova"
                )
            self_training_model = SelfTrainingClassifier(clf)
            self_training_model.fit(X_train, y_train)
           
            # Evaluate Model
            y_pred = self_training_model.predict(X_test)
            y_pred_proba = pd.DataFrame(
                self_training_model.predict_proba(X_test)).fillna(0)
           
            used_model = "class_self_training_lightgbm"
            df_metrics_iter = _getReport(
                y_test,
                y_pred,
                y_pred_proba,
                target_names,
                using_affective = f"{use_aff}",
                semantic_model = semantic_model,
                used_model = used_model
                )
            df_metrics = df_metrics.append(df_metrics_iter)
           
            ### Random Model
            # Set Dummy Classifier
            dummy_clf = DummyClassifier(strategy="uniform")
            dummy_clf.fit(X_train, labels_infer)
           
            # Predictions
            y_pred = dummy_clf.predict(X_test)
            y_pred_proba = pd.DataFrame(dummy_clf.predict_proba(X_test)).fillna(0)
           
            used_model = "class_dummy_classifier"
            df_metrics_iter = _getReport(
                y_test,
                y_pred,
                y_pred_proba,
                target_names,
                using_affective = f"{use_aff}",
                semantic_model = semantic_model,
                used_model = used_model
                )
            df_metrics = df_metrics.append(df_metrics_iter)
           
            # ===================================================================
            # Regression Models
            # ===================================================================
            ### Semisupervised Model [Label Spreading]
            # category = list_names[1]
           
            # Mask Missing Labels
            for kernel in ['knn', 'rbf']:
               
                label_prop_model = LabelSpreading(kernel=kernel)
                labels = np.copy(df_train[[category]].fillna(-1)[category].values)
   
                # Train Model
                label_prop_model.fit(X_train, labels)
                labels_infer = label_prop_model.predict(X_train)
               
                ### Regression Model
                # Dataset
                y_train = labels_infer
               
                # Train Model
                clf = lgb.LGBMRegressor()
                clf.fit(X_train, y_train)
               
                # Evaluate Model
                y_pred = clf.predict(X_test)
               
                used_model = f"reg_label_spreading_lightgbm_{kernel}"
                df_metrics_iter = _getReport(
                    y_test,
                    y_pred, [],
                    target_names,
                    using_affective = f"{use_aff}",
                    semantic_model = semantic_model,
                    used_model = used_model
                    )
                df_metrics = df_metrics.append(df_metrics_iter)  

            print("*"*50)
       
        # Save Results
        if use_aff:
            df_metrics.to_csv(
                f"{PATH_RESULTS}/emotion_aff/df_metrics_{semantic_model}.csv",
                index=False)
        else:
            df_metrics.to_csv(
                f"{PATH_RESULTS}/emotion_no_aff/df_metrics_{semantic_model}.csv",
                index=False)
        print("*"*50)