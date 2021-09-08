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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

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

### Load CV - Psychological
f_path = f"{PATH_RESULTS}/results_cv/psycho_aff_full"
# f_path = f"{PATH_RESULTS}/psycho_aff_test_6"
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

### Get the metrics per psychological tag
df_results_aff = (
    df_raw
    .groupby(by=['category', 'classification_model', 'semantic_model'])
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
    .groupby(by=['classification_model', 'semantic_model'])
    .median()
    .reset_index()
    .copy()
    [['classification_model', 'semantic_model', 'f1_weighted', 'kappa', 'auc']]
    .rename(columns={
        'f1_weighted': 'f1_weighted_median',
        'kappa': 'kappa_median',
        'auc': 'auc_median'
        })
    )
df_results_aff = df_results_aff[df_results_aff['auc']>0.5]

df_results_aff = df_results_aff[df_results_aff['semantic_model'].isin(list_semantic_models)]

# Remove baselines
df_results_aff = df_results_aff[
    (df_results_aff['classification_model'] != 'baseline_lightgbm') &
    (df_results_aff['classification_model'] != 'baseline_smote_lightgbm') &
    # (df_results_aff_test['classification_model'] != 'label_spreading_base') &
    (df_results_aff['classification_model'] != 'dummy_classifier') 
    ].copy()

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
    'en_name', 'semantic_model', 'classification_model',
    'n_class_0', 'n_class_1',
    'f1_weighted', 'kappa', 'auc'
    ]].copy().round(2)
df_reference = df_results
# df_reference = df_results[[
#     'en_name', 'semantic_model', 'classification_model', 
#     'f1_weighted', 'kappa', 'auc'
#     ]].copy().round(2)

df_reference = df_reference.merge(df_median_ref, how="left")

df_reference.round(2).to_csv(
    "tables_paper/df_results_psycho_reference.csv", index=False)


# =============================================================================
# Differences vs. Baselines
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

# Load best combinations
df_reference = pd.read_csv("tables_paper/df_results_psycho_reference.csv")
list_semantic_models = list(set(df_reference['semantic_model'].values))
list_prediction_models = list(set(df_reference['classification_model'].values))
list_categories = list(set(df_reference['en_name'].values))

### Load CV - Psychological
f_path = f"{PATH_RESULTS}/results_cv/psycho_aff"
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

list_baselines = [
    'baseline_lightgbm',
    'baseline_smote_lightgbm',
    'dummy_classifier'
    ]

# Iter and get results
df_metrics = pd.DataFrame()
for i, row in df_reference.iterrows():
    for baseline in list_baselines:
        df_1 = df_raw[
            (df_raw['semantic_model']==row['semantic_model']) &
            (df_raw['classification_model']==row['classification_model']) &
            (df_raw['en_name']==row['en_name'])
            ]
        
        df_2 = df_raw[
            (df_raw['semantic_model']==row['semantic_model']) &
            (df_raw['classification_model']==baseline) &
            (df_raw['en_name']==row['en_name'])
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
             'prediction_model_1': [row['classification_model']],
             'prediction_model_2': [baseline],
             'mean_1_f1': [np.mean(list_f1_df1)],
             'mean_2_f1': [np.mean(list_f1_df2)],
             'p-value_f1': [pVal_f1],
             'mean_1_kappa': [np.mean(list_kappa_df1)],
             'mean_2_kappa': [np.mean(list_kappa_df2)],
             'p-value_kappa': [pVal_kappa],
             'mean_1_auc': [np.mean(list_auc_df1)],
             'mean_2_auc': [np.mean(list_auc_df2)],
             'p-value_auc': [pVal_auc],
                }
            )
        df_metrics = df_metrics.append(df_metrics_iter)
        
df_metrics.round(2).to_csv(
    "tables_paper/df_results_psycho_cv_vs_baseline.csv", index=False)

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

df_plot = df_plot.replace("Anger", "Anger (binary)")
df_plot = df_plot.replace("Inestability", "Instability")
df_plot = df_plot.replace("Helplesness", "Helplessness")
df_plot = df_plot[df_plot['prediction_model']!='dummy_classifier']

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
plt.savefig('results/df_plot_psycho_metrics_vs_baseline.png', dpi=250)
plt.show()

# =============================================================================
# Differences vs. not using affective (Psychological)
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

# Load best combinations
df_reference = pd.read_csv("tables_paper/df_results_psycho_reference.csv")
list_semantic_models = list(set(df_reference['semantic_model'].values))
list_prediction_models = list(set(df_reference['classification_model'].values))
list_categories = list(set(df_reference['en_name'].values))

### Load CV - Psychological - Affective
f_path = f"{PATH_RESULTS}/results_cv/psycho_aff_full"
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

### Load CV - Psychological - Not Affective
f_path = f"{PATH_RESULTS}/results_cv/psycho_no_aff"
f_folders = next(walk(f_path), (None, None, []))[1]  # [] if no file
df_results_not_aff_cv = pd.DataFrame()
i = 0
for folder in f_folders:
    i += 1
    filenames = next(walk(f_path + f'/{folder}'), (None, None, []))[2]
    df_iter = pd.concat([
        pd.read_csv(f_path + f'/{folder}' + '/' + x) for x in filenames
        ])
    df_iter['iter'] = folder
    df_results_not_aff_cv = df_results_not_aff_cv.append(
        df_iter
        )
df_results_not_aff_cv = (
    df_results_not_aff_cv
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
df_results_not_aff_cv = df_results_not_aff_cv.merge(df_names, how="left").round(2).drop(columns=['es_name'])

### Iter and get results
df_metrics = pd.DataFrame()
for i, row in df_reference.iterrows():
    df_1 = df_results_aff_cv[
        (df_results_aff_cv['semantic_model']==row['semantic_model']) &
        (df_results_aff_cv['classification_model']==row['classification_model']) &
        (df_results_aff_cv['en_name']==row['en_name'])
        ]
    
    df_2 = df_results_not_aff_cv[
        (df_results_not_aff_cv['semantic_model']==row['semantic_model']) &
        (df_results_not_aff_cv['classification_model']==row['classification_model']) &
        (df_results_not_aff_cv['en_name']==row['en_name'])
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
         'prediction_model': [row['classification_model']],
         'comb_1': ['affective'],
         'comb_2': ['not affective'],
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
    "tables_paper/df_results_psycho_aff_vs_not_aff_cv.csv", index=False)   


# df_plot = df_metrics[
#     ['category', 'comb_1', 'comb_2', 
#      'mean_1_f1', 'mean_2_f1', 'p-value_f1',
#      'mean_1_kappa', 'mean_2_kappa', 'p-value_kappa',
#      'mean_1_auc', 'mean_2_auc', 'p-value_auc'
#      ]
#     ].copy()

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

df_plot = df_plot.replace("Anger", "Anger (binary)")
df_plot = df_plot.replace("Inestability", "Instability")
df_plot = df_plot.replace("Helplesness", "Helplessness")

df_plot.round(2).to_csv(
    "tables_paper/df_results_psycho_q2.csv", index=False)   


# =============================================================================
# Differences vs. original DISCO (Psychological)
# =============================================================================
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

# Load best combinations
df_reference = pd.read_csv("tables_paper/df_results_psycho_reference.csv")
list_semantic_models = list(set(df_reference['semantic_model'].values))
list_prediction_models = list(set(df_reference['classification_model'].values))
list_categories = list(set(df_reference['en_name'].values))

### Load CV - Psychological - All
f_path = f"{PATH_RESULTS}/results_cv/psycho_aff_full"
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
f_path = f"{PATH_RESULTS}/results_cv/psycho_aff_DISCO"
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
        (df_results_aff_cv['classification_model']==row['classification_model']) &
        (df_results_aff_cv['en_name']==row['en_name'])
        ]
    
    df_2 = df_results_disco_cv[
        (df_results_disco_cv['semantic_model']==row['semantic_model']) &
        (df_results_disco_cv['classification_model']==row['classification_model']) &
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
         'prediction_model': [row['classification_model']],
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
    "tables_paper/df_results_psycho_all_vs_disco_cv.csv", index=False) 


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

df_plot = df_plot.replace("Anger", "Anger (binary)")
df_plot = df_plot.replace("Inestability", "Instability")
df_plot = df_plot.replace("Helplesness", "Helplessness")
df_plot = df_plot[df_plot['configuration']!='best reference']

df_plot_previous = pd.read_csv("tables_paper/df_results_psycho_q2.csv")
df_plot_previous = df_plot_previous[df_plot_previous['configuration']!='only DISCO']
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
plt.savefig('results/df_plot_psycho_metrics_q2_q3.png', dpi=250)
plt.show()

### Analysis (manual)
df_analysis = df_plot[(df_plot['p-value_auc']<=0.1) & (df_plot['configuration']!='best reference')].copy()

## Reference improvement (Q2)
df_analysis_2 = pd.read_csv("tables_paper/df_results_psycho_aff_vs_not_aff_cv.csv")
df_analysis_2 = df_analysis_2[df_analysis_2['p-value_auc']<=0.1]
print(np.mean(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.min(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.max(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))

## Reference improvement (Q3)
df_analysis_2 = pd.read_csv("tables_paper/df_results_psycho_all_vs_disco_cv.csv")
df_analysis_2 = df_analysis_2[df_analysis_2['p-value_auc']<=0.1]
print(np.mean(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.min(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))
print(np.max(df_analysis_2['mean_1_auc'] - df_analysis_2['mean_2_auc']))






