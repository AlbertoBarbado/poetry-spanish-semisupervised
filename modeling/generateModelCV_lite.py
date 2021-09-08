# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:23:59 2021

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
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
# from sentence_transformers import SentenceTransformer, util
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from statsmodels.stats.inter_rater import cohens_kappa
from common.tools import get_files, file_presistance
from common.config import (
    PATH_POEMS, PATH_RESULTS, PATH_AFF_LEXICON, PATH_GROUND_TRUTH
    )

nlp = spacy.load("es_core_news_md")
stemmer = SnowballStemmer("spanish")


def _getReport(
        y_test, y_pred, y_pred_proba, target_names, using_affective = "yes",
        semantic_model = "", classification_model = ""
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
    classification_model : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    df_metrics_iter : TYPE
        DESCRIPTION.

    """
    
    ### 1. Standard Metrics
    report = classification_report(
        y_test, y_pred, target_names = target_names, output_dict = True
        )
    df_metrics_iter = pd.DataFrame(
        {
            'category': [category],
            'using_affective': [using_affective],
            'semantic_model': [semantic_model],
            'classification_model': [classification_model],
            'n_class_0': [report[f'{category}_0']['support']],
            'n_class_1': [report[f'{category}_1']['support']],
            'precision_class_0': [report[f'{category}_0']['precision']],
            'precision_class_1': [report[f'{category}_1']['precision']],
            'recall_class_0': [report[f'{category}_0']['recall']],
            'recall_class_1': [report[f'{category}_1']['recall']],
            'f1_class_0': [report[f'{category}_0']['f1-score']],
            'f1_class_1': [report[f'{category}_1']['f1-score']],
            'precision_weighted': [report['weighted avg']['precision']],
            'recall_weighted': [report['weighted avg']['recall']],
            'f1_weighted': [report['weighted avg']['f1-score']]
            
            }
        )
    
    ### 2. Cohen's Kappa
    # Make Dataframe
    df = pd.DataFrame({"A": y_test, "B": y_pred})
    
    # Switch it to three columns A's answer, B's answer and count of that combination
    df = df.value_counts().reset_index()
    
    # Check compliance
    if len(df) < 4:
        df_aux = pd.DataFrame({'A': [0.0, 1.0, 0.0, 1.0],
                               'B': [0.0, 0.0, 1.0, 1.0] 
                               })
        df = df.merge(df_aux, how="outer").fillna(0)
    
    # Make square
    square = df.pivot(columns="A",index="B").values
    
    # Get Kappa
    dct_kappa = cohens_kappa(square)
    kappa_max = dct_kappa['kappa_max']
    kappa = dct_kappa['kappa']
    df_metrics_iter['kappa'] = [kappa]
    df_metrics_iter['kappa_max'] = [kappa_max]
    
    ### 3. AUC
    y_pred_proba = np.asarray([x if str(x) != 'nan'  else 0.0 for x in y_pred_proba])
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_pred_proba, pos_label=1
        )
    auc = metrics.auc(fpr, tpr)
    df_metrics_iter['auc'] = [auc]
    
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
use_disco = False
if use_disco:
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x <= 4085}
    disco_add_name = "_DISCO"
else:
    disco_add_name = ""

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
    'enc_text_model_hg_ro_avg_w'
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
    item["aff_features"] for key, item in dct_sonnets.items()])

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
if True:
      
    ### Get Subsample from GT
    list_kfolds = []
    n_folds = 50
    for i in range(n_folds):
        df_gt = pd.read_csv(f"{PATH_GROUND_TRUTH}/poems_corpus_all.csv")
        df_gt = df_gt.rename(columns={"text": "text_original"})
        df_gt.columns = [str(x).rstrip().lstrip() for x in list(df_gt.columns)]
    
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
        
        ## Check no affective feature categories are missing
        for category in list_aff:
            l1 = list(df_add[category].unique())
            l2 = list(df_gt[category].unique())
            
            if len(l1)<len(l2):
                l3 = [x for x in l2 if x not in l1]
                df_add_new = df_gt[df_gt[category].isin(l3)]
                df_add_new = df_add_new.drop_duplicates(subset=category)
                df_add = df_add.append(df_add_new)
                df_gt = df_gt[~df_gt["index"].isin(df_add_new["index"])].copy()
        
        list_kfolds.append([{i: {'df_gt': df_gt, 'df_add': df_add}}])
    
    # df_add.to_csv("train_dataset.csv", index=False)
    # df_gt.to_csv("test_dataset.csv", index=False)
else:
    pass
    # df_add = pd.read_csv("train_dataset.csv")
    # df_gt = pd.read_csv("test_dataset.csv")
    
list_use_aff = [True, False]

### Load Reference
df_reference = pd.read_csv("tables_paper/df_results_psycho_reference.csv")
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']
list_categories = list(df_names['category'].values)
aff_labels = ['valence',
            'arousal',
            'happiness',
            'happinness',
            'sadness',
            'anger', 
            'fear',
            'disgust', 
            'concreteness', 
            'imageability', 
            'context availability'
            ]
list_categories = [x for x in list_categories if x not in aff_labels]

# =============================================================================
# 2. Iter and get models
# =============================================================================
# semantic_model = "enc_text_model_hg_ro_avg_w"
for kfold in list_kfolds:
    i_kfold = list(kfold[0].keys())[0]
    df_gt = list(kfold[0].values())[0]['df_gt']
    df_add = list(kfold[0].values())[0]['df_add']
    
    if i_kfold < 5:
        continue

    for use_aff in list_use_aff:
        print("*"*50)
        print()
        print("*"*50)
        df_metrics = pd.DataFrame()
        
        for category in list_categories:
            print("*"*50)
            print(f"Category: {category}")
            
            target_names = [f"{category}_0", f"{category}_1"]
            category_en_name = df_names[
                df_names['category']==category]['en_name'].values[0]
            
            # category_original = df_names[
            #     df_names['en_name']==category]['category'].values[0]
            
            ### Get Corresponing Semantic Model
            semantic_model = df_reference[
                df_reference['en_name']==category_en_name
                ]['semantic_model'].values[0]
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
                df_train_scaled = df_train_scaled.reset_index(drop=True).fillna(0)
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
            labels = np.copy(df_train[[category]].fillna(-1)[category].values)
            
            ### Get Predictive Model
            prediction_best = df_reference[
                df_reference['en_name']==category_en_name
                ]['classification_model'].values[0]
            y_test = df_test_scaled[category].copy() 
            
            
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
            
            # Train Model
            clf = lgb.LGBMClassifier(is_unbalance=True)
            clf.fit(X_train2, y_train2)
            
            # Evaluate Model
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:,1]
            print(classification_report(y_test, y_pred, target_names=target_names))
            
            classification_model = "baseline_lightgbm"
            df_metrics_iter = _getReport(
                y_test,
                y_pred,
                y_pred_proba,
                target_names,
                using_affective = f"{use_aff}",
                semantic_model = semantic_model,
                classification_model = classification_model
                )
            df_metrics = df_metrics.append(df_metrics_iter)
            
            # Oversampling
            try:
                oversample = SMOTE()
                X_train_over, y_train_over = oversample.fit_resample(X_train2, y_train2)
                
                # Train Model
                clf = lgb.LGBMClassifier(is_unbalance=True)
                clf.fit(X_train2, y_train2)
                
                # Evaluate Model
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)[:,1]            
                print(classification_report(y_test, y_pred, target_names=target_names))
                
                classification_model = "baseline_smote_lightgbm"
                df_metrics_iter = _getReport(
                    y_test,
                    y_pred, 
                    y_pred_proba, 
                    target_names, 
                    using_affective = f"{use_aff}",
                    semantic_model = semantic_model, 
                    classification_model = classification_model
                    )
                df_metrics = df_metrics.append(df_metrics_iter)
            except:
                pass
            
            if 'self_training_lightgbm' == prediction_best:
                
                ### Self-Training Classification Model
                y_train = labels
                clf = lgb.LGBMClassifier(is_unbalance=True)
                self_training_model = SelfTrainingClassifier(clf)
                self_training_model.fit(X_train, y_train)
                  
                # Evaluate Model
                y_pred = self_training_model.predict(X_test)
                y_pred_proba = pd.DataFrame(
                    self_training_model.predict_proba(X_test)).fillna(0).values
                y_pred_train = self_training_model.predict(X_train)


            elif 'label_spreading_lightgbm' in prediction_best:
                if 'knn' in prediction_best:
                    kernel = "knn"
                else:
                    kernel = "rbf"
                    
                label_prop_model = LabelSpreading(kernel=kernel)
                labels = np.copy(df_train[[category]]
                                 .fillna(-1)[category]
                                 .values
                                 )
        
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
                
                y_train = labels_infer
                
                if "smote" in prediction_best:
                    # Oversampling
                    oversample = SMOTE()
                    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
                    # Train Model
                    clf = lgb.LGBMClassifier()
                    clf.fit(X_train_over, y_train_over)
                else:
                    # Train Model
                    clf = lgb.LGBMClassifier(is_unbalance=True)
                    clf.fit(X_train, y_train)
                    
                # Evaluate Model
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)
                y_pred_train = clf.predict(X_train)
            else:
                raise ValueError("ERROR")
            
            classification_model = prediction_best
            df_metrics_iter = _getReport(
                        y_test,
                        y_pred, 
                        y_pred_proba[:,1], 
                        target_names, 
                        using_affective = f"{use_aff}",
                        semantic_model = semantic_model, 
                        classification_model = classification_model
                        )
            df_metrics = df_metrics.append(df_metrics_iter)
            
            ### Random Model
            # Set Dummy Classifier
            label_prop_model = LabelSpreading(kernel="rbf")
            labels = np.copy(df_train[[category]]
                             .fillna(-1)[category]
                             .values
                             )
            # Train Model
            label_prop_model.fit(X_train, labels)
            labels_infer = label_prop_model.predict(X_train)
            
            dummy_clf = DummyClassifier(strategy="uniform")
            dummy_clf.fit(X_train, labels_infer)
            
            # Predictions
            target_names = [f"{category}_0", f"{category}_1"]
            y_pred = dummy_clf.predict(X_test)
            y_pred_proba = dummy_clf.predict_proba(X_test)[:,1]
            print(classification_report(y_test, y_pred, target_names=target_names))
            
            classification_model = "dummy_classifier"
            df_metrics_iter = _getReport(
                y_test,
                y_pred, 
                y_pred_proba,
                target_names, 
                using_affective = f"{use_aff}",
                semantic_model = semantic_model, 
                classification_model = classification_model
                )
            df_metrics = df_metrics.append(df_metrics_iter)

        # Save Results
        if use_aff:
            f_path = f"{PATH_RESULTS}/results_cv/psycho_aff{disco_add_name}/{i_kfold}"
            if not os.path.exists(f_path):
                os.makedirs(f_path)
    
            df_metrics.to_csv(
                f_path + f'/df_metrics_{i_kfold}.csv',
                index=False
            )
        else:
            f_path = f"{PATH_RESULTS}/results_cv/psycho_no_aff{disco_add_name}/{i_kfold}"
            if not os.path.exists(f_path):
                os.makedirs(f_path)
                
            df_metrics.to_csv(
                f_path + f'/df_metrics_{i_kfold}.csv',
                index=False
            )
        print("*"*50)