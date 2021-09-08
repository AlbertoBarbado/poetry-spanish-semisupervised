# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:17:07 2021

@author: alber
"""

import os
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import lightgbm as lgb

from os import walk
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from sklearn import preprocessing
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)
from common.config import (
    PATH_POEMS, PATH_RESULTS, PATH_AFF_LEXICON, PATH_GROUND_TRUTH
    )

df_metrics_h_test = pd.DataFrame()

### Sample Size
# parameters for power analysis
effect = 0.8
alpha = 0.1 # Ojo al alpha, que no es 0.5
power = 0.8
# perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)

df_kappa_limits = pd.DataFrame(
    {
     'limit_k': [0, 0.2, 0.4],
     'category': ['poor', 'slight', 'fair']
     }
    )

# =============================================================================
# Best Models based on CV
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

### Load CV - Psychological
f_path = f"{PATH_RESULTS}/results_cv/emotion_aff_full"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_aff_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x, encoding="latin-1") for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_aff_cv = df_results_aff_cv.append(
        df_iter
        )
df_raw = df_results_aff_cv
df_raw = (
    df_raw
    .replace("AversiÃ³n", "Aversión")
    .replace("DepresiÃ³n", "Depresión")
    .replace('DramatizaciÃ³n', "Dramatización")
    .replace('IlusiÃ³n', "Ilusión")
    .replace("DesilusiÃ³n", "Desilusión")
    .replace("ObsesiÃ³n", "Obsesión")
    .replace("CompulsiÃ³n", "Compulsión")
    .replace("EnsoÃ±aciÃ³n", "Ensoñación")
    .replace("IdealizaciÃ³n", "Idealización")
    .dropna(subset=['category'])
    .drop(columns=['ï»¿category', 'en_name'], errors="ignore")
    )
df_raw = df_raw.merge(df_names, how="left").round(2)
    
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']
df_raw = (
    df_raw
    .merge(df_names, how="left")
    .drop(columns=['es_name'])
    )

### Get the metrics per emotion tag
df_results_aff = (
    df_raw
    .groupby(by=['category', 'regression_model', 'semantic_model'])
    .mean()
    .reset_index()
    )

df_results_aff['mean_metric'] = (
    (df_results_aff['kappa']+
    df_results_aff['auc'])
    /
    2
    ) 
    
df_median_ref = (
    df_results_aff
    .groupby(by=['regression_model', 'semantic_model'])
    .median()
    .reset_index()
    .copy()
    [['regression_model', 'semantic_model', 'f1_weighted', 'kappa', 'auc', 'corr']]
    .rename(columns={
        'f1_weighted': 'f1_weighted_median',
        'kappa': 'kappa_median',
        'auc': 'auc_median',
        'corr': 'corr_median'
        })
    )

df_results_aff = df_results_aff[df_results_aff['auc']>0.5]
df_results_aff = df_results_aff[df_results_aff.fillna(0)['corr']>=0]

# Remove baselines
df_results_aff = df_results_aff[
    (df_results_aff['regression_model'] != 'class_baseline_lightgbm') &
    (df_results_aff['regression_model'] != 'class_baseline_smote_lightgbm') &
    (df_results_aff['regression_model'] != 'class_label_spreading_base_knn') &
    (df_results_aff['regression_model'] != 'class_label_spreading_base_rbf') &
    (df_results_aff['regression_model'] != 'class_dummy_classifier') &
    (df_results_aff['regression_model'] != 'reg_baseline_lightgbm') &
    (df_results_aff['regression_model'] != 'reg_baseline_smote_lightgbm') &
    (df_results_aff['regression_model'] != 'reg_label_spreading_base') &
    (df_results_aff['regression_model'] != 'reg_dummy_classifier') 
    ].copy()

# Remove unused semantic models
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
df_results_aff = df_results_aff[
    df_results_aff['semantic_model'].isin(list_semantic_models)]


df_results_aff = (
    df_results_aff
    .sort_values(by=['category', 'mean_metric'], ascending=False)
    .groupby(by=['category'])
    .first()
    .reset_index()
    )
df_results_aff = (
    df_results_aff.merge(df_names, how="left").drop(columns=['es_name'])
    )

df_results = df_results_aff[[
    'en_name', 'semantic_model', 'regression_model',
    'f1_weighted', 'kappa', 'auc', 'corr'
    ]].copy().round(2)
df_reference = df_results
# df_reference = df_results[[
#     'en_name', 'semantic_model', 'classification_model', 
#     'f1_weighted', 'kappa', 'auc'
#     ]].copy().round(2)

df_reference = df_reference.merge(df_median_ref, how="left")


### Add data distribution
# Load psycho names
df_names = pd.read_csv(f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
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

list_kfolds = []
n_folds = 21
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
    
df_distribution = pd.DataFrame()
for iter_item in list_kfolds:
    iter_item = [x for x in iter_item[0].values()][0]['df_gt']
    for category in list_aff:
        data_cat = (
            pd.DataFrame(iter_item[category].copy().value_counts())
            .T
            .reset_index()
            .rename(columns={'index':'en_name'})
            )
        df_distribution = df_distribution.append(data_cat)
df_distribution = df_distribution.groupby(by=['en_name']).mean().reset_index().round(1)
df_distribution = df_distribution.replace("fear", "Fear (ordinal)")
df_distribution = df_distribution.replace("happinness", "happiness")
df_reference = df_distribution.merge(df_reference)

df_reference.round(2).to_csv(
    "tables_paper/df_results_emotions_reference.csv", index=False)

# =============================================================================
# Differences vs. Baselines
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

# Load best combinations
df_reference = pd.read_csv("tables_paper/df_results_emotions_reference.csv")
list_semantic_models = list(set(df_reference['semantic_model'].values))
list_prediction_models = list(set(df_reference['regression_model'].values))
list_categories = list(set(df_reference['en_name'].values))

### Load CV - Emotions
f_path = f"{PATH_RESULTS}/results_cv/emotion_aff_full"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_aff_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x) for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_aff_cv = df_results_aff_cv.append(
        df_iter
        )
df_raw = df_results_aff_cv
df_raw = df_raw.merge(df_names, how="left").drop(columns=['es_name'])

# Set missing SMOTE models as non-SMOTE results
df_aux = df_raw[(df_raw['regression_model']=='class_baseline_lightgbm') &
                (df_raw['category']=='happinness')
                ].copy()
df_aux['regression_model'] = 'class_baseline_smote_lightgbm'
df_raw = df_raw.append(df_aux)

df_aux = df_raw[(df_raw['regression_model']=='class_baseline_lightgbm') &
                (df_raw['category']=='fear')
                ].copy()
df_aux['regression_model'] = 'class_baseline_smote_lightgbm'
df_raw = df_raw.append(df_aux)

list_baselines = [
    'class_baseline_lightgbm',
    'class_baseline_smote_lightgbm',
    'class_dummy_classifier',
    'reg_baseline_lightgbm'
    ]
# Iter and get results
df_metrics = pd.DataFrame()
for i, row in df_reference.iterrows():
    for baseline in list_baselines:
        df_1 = df_raw[
            (df_raw['semantic_model']==row['semantic_model']) &
            (df_raw['regression_model']==row['regression_model']) &
            (df_raw['en_name']==row['en_name'])
            ]
        
        df_2 = df_raw[
            (df_raw['semantic_model']==row['semantic_model']) &
            (df_raw['regression_model']==baseline) &
            (df_raw['en_name']==row['en_name'])
            ]
        
        list_f1_df1 = list(df_1['f1_weighted'].values)
        list_f1_df2 = list(df_2['f1_weighted'].values)
        
        list_kappa_df1 = list(df_1['kappa'].values)
        list_kappa_df2 = list(df_2['kappa'].values)
        
        list_auc_df1 = list(df_1['auc'].values)
        list_auc_df2 = list(df_2['auc'].values)
        
        list_corr_df1 = list(df_1['corr'].values)
        list_corr_df2 = list(df_2['corr'].values)
        
        try:
            _, pVal_f1 = stats.kruskal(list_f1_df1, list_f1_df2)
            _, pVal_kappa = stats.kruskal(list_kappa_df1, list_kappa_df2)
            _, pVal_auc = stats.kruskal(list_auc_df1, list_auc_df2)
            _, pVal_corr = stats.kruskal(list_corr_df1, list_corr_df2)
        except:
            pVal_f1 = 1
            pVal_kappa = 1
            pVal_auc = 1
            pVal_corr = 1
        
        df_metrics_iter = pd.DataFrame(
            {'category': [row['en_name']],
             'semantic_model': [row['semantic_model']],
             'prediction_model_1': [row['regression_model']],
             'prediction_model_2': [baseline],
             'mean_1_f1': [np.mean(list_f1_df1)],
             'mean_2_f1': [np.mean(list_f1_df2)],
             'median_1_f1': [np.median(list_f1_df1)],
             'median_2_f1': [np.median(list_f1_df2)],
             'p-value_f1': [pVal_f1],
             'mean_1_kappa': [np.mean(list_kappa_df1)],
             'mean_2_kappa': [np.mean(list_kappa_df2)],
             'median_1_kappa': [np.median(list_kappa_df1)],
             'median_2_kappa': [np.median(list_kappa_df2)],
             'p-value_kappa': [pVal_kappa],
             'mean_1_auc': [np.mean(list_auc_df1)],
             'mean_2_auc': [np.mean(list_auc_df2)],
             'median_1_auc': [np.median(list_auc_df1)],
             'median_2_auc': [np.median(list_auc_df2)],
             'p-value_auc': [pVal_auc],
             'mean_1_corr': [np.mean(list_corr_df1)],
             'mean_2_corr': [np.mean(list_corr_df2)],
             'median_1_corr': [np.median(list_corr_df1)],
             'median_2_corr': [np.median(list_corr_df2)],
             'p-value_corr': [pVal_corr],
                }
            )
        df_metrics = df_metrics.append(df_metrics_iter)
        
df_metrics.round(2).to_csv(
    "tables_paper/df_results_emotions_cv_vs_baseline.csv", index=False)

# Plot Data
df_aux = (df_metrics[['category', 'prediction_model_1', 'mean_1_auc', 'p-value_auc']]
            .rename(columns={
                'prediction_model_1':'prediction_model',
                'mean_1_auc':'mean_auc'})
            )
df_aux['prediction_model'] = 'best reference'
df_plot = (
    df_metrics[['category', 'prediction_model_2', 'mean_2_auc', 'p-value_auc']]
    .rename(columns={'prediction_model_2':'prediction_model',
                     'mean_2_auc':'mean_auc'})
    .append(df_aux)
    )


df_plot = df_plot[df_plot['prediction_model']!='class_dummy_classifier']
df_plot = df_plot[df_plot['prediction_model']!='reg_dummy_classifier']
df_plot = df_plot[df_plot['prediction_model']!='reg_baseline_lightgbm']
df_plot = df_plot[df_plot['prediction_model']!='baseline_affective']
df_plot = df_plot.replace("Fear (ordinal)", "fear (ordinal)")
df_plot = df_plot.replace("anger", "anger (ordinal)")

plt.figure(figsize=(16, 10), dpi=250)
sns.set_theme(style="darkgrid")
sns.set(font_scale=1.2)
plot_fig = sns.barplot(data = df_plot,
                       x = 'category',
                       y = 'mean_auc',
                       hue = 'prediction_model'
                      )
plot_fig.set(
    ylabel = "AUC Value",
    xlabel = 'Psychological Category'
    )
plot_fig.set_title(
    "AUC metrics versus baseline models",
    fontdict = {'fontsize':16},
    pad = 12
    )
plt.ylim(0, 0.9)
plot_fig.set_xticklabels(
    plot_fig.get_xticklabels(), rotation=45, horizontalalignment='right')
# plot_fig.set_yticklabels(
#     plot_fig.get_yticklabels(), rotation=360, horizontalalignment='right')
plt.legend(loc="upper left")
plt.savefig('results/df_plot_emotions_metrics_vs_baseline.png', dpi=250)
plt.show()

### Analysis (manual)
df_analysis = df_plot[(df_plot['p-value_auc']>0.1) & (df_plot['prediction_model']!='best reference')].copy()


# =============================================================================
# Differences vs. original DISCO (Emotions)
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

# Load best combinations
df_reference = pd.read_csv("tables_paper/df_results_emotions_reference.csv")
list_semantic_models = list(set(df_reference['semantic_model'].values))
list_prediction_models = list(set(df_reference['regression_model'].values))
list_categories = list(set(df_reference['en_name'].values))

### Load CV - Emotions - All
f_path = f"{PATH_RESULTS}/results_cv/emotion_aff_full"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_aff_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x) for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_aff_cv = df_results_aff_cv.append(
        df_iter
        )
df_results_aff_cv = (
    df_results_aff_cv
    .replace("AversiÃ³n", "Aversión")
    .replace("DepresiÃ³n", "Depresión")
    .replace('DramatizaciÃ³n', "Dramatización")
    .replace('IlusiÃ³n', "Ilusión")
    .replace("DesilusiÃ³n", "Desilusión")
    .replace("ObsesiÃ³n", "Obsesión")
    .replace("CompulsiÃ³n", "Compulsión")
    .replace("EnsoÃ±aciÃ³n", "Ensoñación")
    .replace("IdealizaciÃ³n", "Idealización")
    .dropna(subset=['category'])
    .drop(columns=['ï»¿category', 'en_name'], errors="ignore")
    )
df_results_aff_cv = df_results_aff_cv.merge(df_names, how="left").round(2).drop(columns=['es_name'])

### Load CV - Psychological - DISCO
f_path = f"{PATH_RESULTS}/results_cv/emotion_aff_DISCO"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_disco_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x) for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_disco_cv = df_results_disco_cv.append(
        df_iter
        )
df_results_disco_cv = (
    df_results_disco_cv
    .replace("AversiÃ³n", "Aversión")
    .replace("DepresiÃ³n", "Depresión")
    .replace('DramatizaciÃ³n', "Dramatización")
    .replace('IlusiÃ³n', "Ilusión")
    .replace("DesilusiÃ³n", "Desilusión")
    .replace("ObsesiÃ³n", "Obsesión")
    .replace("CompulsiÃ³n", "Compulsión")
    .replace("EnsoÃ±aciÃ³n", "Ensoñación")
    .replace("IdealizaciÃ³n", "Idealización")
    .dropna(subset=['category'])
    .drop(columns=['ï»¿category', 'en_name'], errors="ignore")
    )
df_results_disco_cv = df_results_disco_cv.merge(df_names, how="left").round(2).drop(columns=['es_name'])

### Iter and get results
df_metrics = pd.DataFrame()
for i, row in df_reference.iterrows():
    df_1 = df_results_aff_cv[
        (df_results_aff_cv['semantic_model']==row['semantic_model']) &
        (df_results_aff_cv['regression_model']==row['regression_model']) &
        (df_results_aff_cv['en_name']==row['en_name'])
        ]
    
    df_2 = df_results_disco_cv[
        (df_results_disco_cv['semantic_model']==row['semantic_model']) &
        (df_results_disco_cv['regression_model']==row['regression_model']) &
        (df_results_disco_cv['en_name']==row['en_name'])
        ]
    
    list_f1_df1 = list(df_1['f1_weighted'].values)
    list_f1_df2 = list(df_2['f1_weighted'].values)
    
    list_kappa_df1 = list(df_1['kappa'].values)
    list_kappa_df2 = list(df_2['kappa'].values)
    
    list_auc_df1 = list(df_1['auc'].values)
    list_auc_df2 = list(df_2['auc'].values)
    
    try:
        _, pVal_f1 = stats.kruskal(list_f1_df1, list_f1_df2)
        _, pVal_kappa = stats.kruskal(list_kappa_df1, list_kappa_df2)
        _, pVal_auc = stats.kruskal(list_auc_df1, list_auc_df2)
    except:
        pVal_f1 = 1
        pVal_kappa = 1
        pVal_auc = 1
    
    df_metrics_iter = pd.DataFrame(
        {'category': [row['en_name']],
         'semantic_model': [row['semantic_model']],
         'prediction_model': [row['regression_model']],
         'comb_1': ['All'],
         'comb_2': ['DISCO'],
        'mean_1_f1': [np.mean(list_f1_df1)],
        'mean_2_f1': [np.mean(list_f1_df2)],
        'median_1_f1': [np.median(list_f1_df1)],
        'median_2_f1': [np.median(list_f1_df2)],
        'p-value_f1': [pVal_f1],
        'mean_1_kappa': [np.mean(list_kappa_df1)],
        'mean_2_kappa': [np.mean(list_kappa_df2)],
        'median_1_kappa': [np.median(list_kappa_df1)],
        'median_2_kappa': [np.median(list_kappa_df2)],
        'p-value_kappa': [pVal_kappa],
        'mean_1_auc': [np.mean(list_auc_df1)],
        'mean_2_auc': [np.mean(list_auc_df2)],
        'median_1_auc': [np.median(list_auc_df1)],
        'median_2_auc': [np.median(list_auc_df2)],
        'p-value_auc': [pVal_auc],
            }
        )
    df_metrics = df_metrics.append(df_metrics_iter)
df_metrics.round(2).to_csv(
    "tables_paper/df_results_emotion_all_vs_disco_cv.csv", index=False) 

# Plot Data
df_aux = (df_metrics[['category', 'comb_1', 'mean_1_auc', 'p-value_auc']]
            .rename(columns={
                'comb_1':'configuration',
                'mean_1_auc':'mean_auc'})
            )
df_aux['configuration'] = 'best reference'

df_aux_2 = (df_metrics[['category', 'comb_2', 'mean_2_auc', 'p-value_auc']]
    .rename(columns={'comb_2':'configuration',
                     'mean_2_auc':'mean_auc'})
    )
df_aux_2['configuration'] = 'only DISCO'  

df_plot = (
    df_aux_2
    .append(df_aux)
    )

df_plot = df_plot.replace("Fear (ordinal)", "fear (ordinal)")
df_plot = df_plot.replace("anger", "anger (ordinal)")

df_plot.round(2).to_csv(
    "tables_paper/df_results_emotions_q3.csv", index=False)   


# =============================================================================
# Differences vs not using affective
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

# Load best combinations
df_reference = pd.read_csv("tables_paper/df_results_emotions_reference.csv")
list_semantic_models = list(set(df_reference['semantic_model'].values))
list_prediction_models = list(set(df_reference['regression_model'].values))
list_categories = list(set(df_reference['en_name'].values))

### Load CV - Emotions - All
f_path = f"{PATH_RESULTS}/results_cv/emotion_aff_full"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_aff_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x) for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_aff_cv = df_results_aff_cv.append(
        df_iter
        )
df_results_aff_cv = (
    df_results_aff_cv
    .replace("AversiÃ³n", "Aversión")
    .replace("DepresiÃ³n", "Depresión")
    .replace('DramatizaciÃ³n', "Dramatización")
    .replace('IlusiÃ³n', "Ilusión")
    .replace("DesilusiÃ³n", "Desilusión")
    .replace("ObsesiÃ³n", "Obsesión")
    .replace("CompulsiÃ³n", "Compulsión")
    .replace("EnsoÃ±aciÃ³n", "Ensoñación")
    .replace("IdealizaciÃ³n", "Idealización")
    .dropna(subset=['category'])
    .drop(columns=['ï»¿category', 'en_name'], errors="ignore")
    )
df_results_aff_cv = df_results_aff_cv.merge(df_names, how="left").round(2).drop(columns=['es_name'])

### Load CV - Psychological - DISCO
f_path = f"{PATH_RESULTS}/results_cv/emotion_no_aff"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_disco_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x) for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_disco_cv = df_results_disco_cv.append(
        df_iter
        )
df_results_disco_cv = (
    df_results_disco_cv
    .replace("AversiÃ³n", "Aversión")
    .replace("DepresiÃ³n", "Depresión")
    .replace('DramatizaciÃ³n', "Dramatización")
    .replace('IlusiÃ³n', "Ilusión")
    .replace("DesilusiÃ³n", "Desilusión")
    .replace("ObsesiÃ³n", "Obsesión")
    .replace("CompulsiÃ³n", "Compulsión")
    .replace("EnsoÃ±aciÃ³n", "Ensoñación")
    .replace("IdealizaciÃ³n", "Idealización")
    .dropna(subset=['category'])
    .drop(columns=['ï»¿category', 'en_name'], errors="ignore")
    )
df_results_disco_cv = df_results_disco_cv.merge(df_names, how="left").round(2).drop(columns=['es_name'])

### Iter and get results
df_metrics = pd.DataFrame()
for i, row in df_reference.iterrows():
    df_1 = df_results_aff_cv[
        (df_results_aff_cv['semantic_model']==row['semantic_model']) &
        (df_results_aff_cv['regression_model']==row['regression_model']) &
        (df_results_aff_cv['en_name']==row['en_name'])
        ]
    
    df_2 = df_results_disco_cv[
        (df_results_disco_cv['semantic_model']==row['semantic_model']) &
        (df_results_disco_cv['regression_model']==row['regression_model']) &
        (df_results_disco_cv['en_name']==row['en_name'])
        ]
    
    list_f1_df1 = list(df_1['f1_weighted'].values)
    list_f1_df2 = list(df_2['f1_weighted'].values)
    
    list_kappa_df1 = list(df_1['kappa'].values)
    list_kappa_df2 = list(df_2['kappa'].values)
    
    list_auc_df1 = list(df_1['auc'].values)
    list_auc_df2 = list(df_2['auc'].values)
    
    try:
        _, pVal_f1 = stats.kruskal(list_f1_df1, list_f1_df2)
        _, pVal_kappa = stats.kruskal(list_kappa_df1, list_kappa_df2)
        _, pVal_auc = stats.kruskal(list_auc_df1, list_auc_df2)
    except:
        pVal_f1 = 1
        pVal_kappa = 1
        pVal_auc = 1
    
    df_metrics_iter = pd.DataFrame(
        {'category': [row['en_name']],
         'semantic_model': [row['semantic_model']],
         'prediction_model': [row['regression_model']],
         'comb_1': ['All'],
         'comb_2': ['DISCO'],
        'mean_1_f1': [np.mean(list_f1_df1)],
        'mean_2_f1': [np.mean(list_f1_df2)],
        'median_1_f1': [np.median(list_f1_df1)],
        'median_2_f1': [np.median(list_f1_df2)],
        'p-value_f1': [pVal_f1],
        'mean_1_kappa': [np.mean(list_kappa_df1)],
        'mean_2_kappa': [np.mean(list_kappa_df2)],
        'median_1_kappa': [np.median(list_kappa_df1)],
        'median_2_kappa': [np.median(list_kappa_df2)],
        'p-value_kappa': [pVal_kappa],
        'mean_1_auc': [np.mean(list_auc_df1)],
        'mean_2_auc': [np.mean(list_auc_df2)],
        'median_1_auc': [np.median(list_auc_df1)],
        'median_2_auc': [np.median(list_auc_df2)],
        'p-value_auc': [pVal_auc],
            }
        )
    df_metrics = df_metrics.append(df_metrics_iter)
df_metrics.round(2).to_csv(
    "tables_paper/df_results_emotion_all_vs_no_aff_cv.csv", index=False) 

# Plot Data
df_aux = (df_metrics[['category', 'comb_1', 'mean_1_auc', 'p-value_auc']]
            .rename(columns={
                'comb_1':'configuration',
                'mean_1_auc':'mean_auc'})
            )
df_aux['configuration'] = 'best reference'

df_aux_2 = (df_metrics[['category', 'comb_2', 'mean_2_auc', 'p-value_auc']]
    .rename(columns={'comb_2':'configuration',
                     'mean_2_auc':'mean_auc'})
    )
df_aux_2['configuration'] = 'without additional features'  

df_plot = (
    df_aux_2
    .append(df_aux)
    )

df_plot = df_plot.replace("Fear (ordinal)", "fear (ordinal)")
df_plot = df_plot.replace("anger", "anger (ordinal)")
df_plot = df_plot[df_plot['configuration']!='best reference']

df_plot_previous = pd.read_csv("tables_paper/df_results_emotions_q3.csv")
df_plot_previous = df_plot_previous[df_plot_previous['configuration']!='without additional features']
df_plot = df_plot.append(df_plot_previous)

df_plot.round(2).to_csv(
    "tables_paper/df_results_psycho_q2_q3.csv", index=False)  


plt.figure(figsize=(16, 10), dpi=250)
sns.set_theme(style="darkgrid")
sns.set(font_scale=1.2)
plot_fig = sns.barplot(data = df_plot,
                       x = 'category',
                       y = 'mean_auc',
                       hue = 'configuration'
                      )
plot_fig.set(
    ylabel = "AUC Value",
    xlabel = 'Psychological Category'
    )
plot_fig.set_title(
    "AUC metrics depending on the features and the corpus used",
    fontdict = {'fontsize':16},
    pad = 12
    )
plt.ylim(0, 0.9)
plot_fig.set_xticklabels(
    plot_fig.get_xticklabels(), rotation=45, horizontalalignment='right')
# plot_fig.set_yticklabels(
#     plot_fig.get_yticklabels(), rotation=360, horizontalalignment='right')
plt.legend(loc="upper left")
plt.savefig('results/df_plot_emotions_metrics_q2_q3.png', dpi=250)
plt.show()

### Analysis (manual)
df_analysis = df_plot[(df_plot['p-value_auc']<=0.1) & (df_plot['configuration']!='best reference')].copy()

## Reference improvement (Q2)
df_analysis_2 = pd.read_csv("tables_paper/df_results_emotion_all_vs_no_aff_cv.csv")
df_analysis_2 = df_analysis_2[df_analysis_2['p-value_auc']<=0.1]
print(np.mean(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.min(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.max(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))

## Reference improvement (Q3)
df_analysis_2 = pd.read_csv("tables_paper/df_results_emotion_all_vs_disco_cv.csv")
df_analysis_2 = df_analysis_2[df_analysis_2['p-value_auc']<=0.1]
print(np.mean(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.min(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.max(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))