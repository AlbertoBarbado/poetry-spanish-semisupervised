# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:00:02 2021

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
import spacy
import unidecode

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
nlp = spacy.load('es_core_news_md')

### Load Data
# ENG names
df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")
df_names['category'] = df_names['es_name']

df_add = pd.read_csv("train_dataset.csv")
df_gt = pd.read_csv("test_dataset.csv")
list_idx_ann = list(df_add['index'].values) + list(df_gt['index'].values)
list_idx_ann = list(set(list_idx_ann))

"""
# Load Data
file_to_read = open(f"{PATH_RESULTS}/dct_sonnets_input_v5", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

# Update Words per Sonnet
from common.data_processing import aux_funcion_get_words
for i, sonnet in dct_sonnets.items():
    print("Iter", i)
    dct_sonnets[i]['words'] = aux_funcion_get_words(dct_sonnets[i]['text'])

### Save Results
with open(f"{PATH_RESULTS}/dct_sonnets_input_mod_v5_analysis", 'wb') as handle:
    pickle.dump(dct_sonnets, handle)

"""

FILE_NAME = "dct_sonnets_input_mod_v5_analysis"

# Load Data
file_to_read = open(f"{PATH_RESULTS}/{FILE_NAME}", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

# Load Lexicons
"""
file_to_read = open(f"{PATH_AFF_LEXICON}/list_affective_lexicons.p", "rb")
l_new = pickle.load(file_to_read)
file_to_read.close()
[df_raw_1, df_raw_3, df_raw_4, df_raw_5, df_raw_6, df_all] = l_new

"""
file_to_read = open(f"{PATH_RESULTS}/dct_backup.p", "rb")
dct_backup = pickle.load(file_to_read)
file_to_read.close()

df_raw_1 = dct_backup['df_raw_1']
df_raw_3 = dct_backup['df_raw_3']
df_raw_4 = dct_backup['df_raw_4']
df_raw_5 = dct_backup['df_raw_5']
df_raw_5 = dct_backup['df_raw_5']
df_raw_6 = dct_backup['df_raw_6']
df_all = dct_backup['df_all']


# List of words to ignore (interjections & some stopwords)
list_not = [
    'ay', 'oh', 'cualquiera', 'cuyo', 'do', 'cuán', 'ah', 'd', 'acá', 'ey',
    'aquesta', 'loor', 'aqueste', 'vo', 's', 'ca'
    ]

"""
df_all_stem = (
    df_all.
    drop(columns=['Word', 'word_lem', 'word_stem']).
    groupby(by=['words_stem_uni']).
    median().
    reset_index()
    .rename(columns={'words_stem_uni':'word_stem'})
    )

## Remove accents
df_all['Word'] = df_all.apply(lambda x: unidecode.unidecode(x['Word']), axis = 1)
df_all['word_lem'] = df_all.apply(lambda x: unidecode.unidecode(x['word_lem']), axis = 1)
df_all['word_stem'] = df_all.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_1['Word'] = df_raw_1.apply(lambda x: unidecode.unidecode(x['Word']), axis = 1)
df_raw_1['word_lem'] = df_raw_1.apply(lambda x: unidecode.unidecode(x['word_lem']), axis = 1)
df_raw_1['word_stem'] = df_raw_1.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_3['Word'] = df_raw_3.apply(lambda x: unidecode.unidecode(x['Word']), axis = 1)
df_raw_3['word_lem'] = df_raw_3.apply(lambda x: unidecode.unidecode(x['word_lem']), axis = 1)
df_raw_3['word_stem'] = df_raw_3.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_4['Word'] = df_raw_4.apply(lambda x: unidecode.unidecode(x['Word']), axis = 1)
df_raw_4['word_lem'] = df_raw_4.apply(lambda x: unidecode.unidecode(x['word_lem']), axis = 1)
df_raw_4['word_stem'] = df_raw_4.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_5['Word'] = df_raw_5.apply(lambda x: unidecode.unidecode(x['Word']), axis = 1)
df_raw_5['word_lem'] = df_raw_5.apply(lambda x: unidecode.unidecode(x['word_lem']), axis = 1)
df_raw_5['word_stem'] = df_raw_5.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_6['Word'] = df_raw_6.apply(lambda x: unidecode.unidecode(x['Word']), axis = 1)
df_raw_6['word_lem'] = df_raw_6.apply(lambda x: unidecode.unidecode(x['word_lem']), axis = 1)
df_raw_6['word_stem'] = df_raw_6.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

"""

"""
# Update lemmatization
import re
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("spanish")


df_all['word_lem'] = df_all['Word']
df_all['word_stem'] = df_all.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
df_all['words_stem_uni'] = df_all.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_1['word_lem'] = df_raw_1['Word']
df_raw_1['word_stem'] = df_raw_1.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
df_raw_1['words_stem_uni'] = df_raw_1.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_3['word_lem'] = df_raw_3['Word']
df_raw_3['word_stem'] = df_raw_3.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
df_raw_3['words_stem_uni'] = df_raw_3.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_4['word_lem'] = df_raw_4['Word']
df_raw_4['word_stem'] = df_raw_4.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
df_raw_4['words_stem_uni'] = df_raw_4.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_5['word_lem'] = df_raw_5['Word']
df_raw_5['word_stem'] = df_raw_5.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
df_raw_5['words_stem_uni'] = df_raw_5.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_6['word_lem'] = df_raw_6['Word']
df_raw_6['word_stem'] = df_raw_6.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
df_raw_6['words_stem_uni'] = df_raw_6.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

dct_backup = {
    'df_raw_1': df_raw_1,
    'df_raw_3': df_raw_3,
    'df_raw_4': df_raw_4,
    'df_raw_5': df_raw_5,
    'df_raw_6': df_raw_6,
    'df_all': df_all
    }

### Save Results
with open(f"{PATH_RESULTS}/dct_backup.p", 'wb') as handle:
    pickle.dump(dct_backup, handle)
"""

# Only DISCO
type_used = "ANN"
# type_used = 'DISCO'
# type_used = 'NEW'

if type_used == "DISCO":
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x <= 4085}
    
elif type_used == 'NEW':
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x > 4085}
    
elif type_used == 'ANN':
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x in list_idx_ann}
    

"""
import re
from nltk.corpus import stopwords
def _get_original_words(text):
    words = re.findall(r'\w+', text,flags = re.UNICODE)
    # Upper case to lowercase
    words = [w.lower() for w in words]
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('spanish')]
    # Remove more stopwords
    words = [w for w in words if nlp.vocab[w].is_stop == False]
    # Remove non alphanumeric characters
    words = [w for w in words if w.isalpha()]
    return words

for i, sonnet in dct_sonnets.items():
    print(i)
    sonnet['words'] = _get_original_words(sonnet['text'])
    dct_sonnets[i] = sonnet

"""
print()

# =============================================================================
# Words per Lexicon
# =============================================================================
# # Get vocabulary sonnets
# total_vocab = []
# total_vocab_stem = []
# total_vocab_lem = []
# for i, sonnet in dct_sonnets.items():
#     total_vocab = list(set(total_vocab + sonnet["words"]))
#     total_vocab_stem = list(set(total_vocab_stem + sonnet["words_stem"]))
#     total_vocab_lem = list(set(total_vocab_lem + sonnet["words_lem"]))
    
# list_not = ['ay', 'oh', 'cualquiera', 'cuyo']
# total_vocab = [x for x in total_vocab if x not in list_not]
# total_vocab_stem = [x for x in total_vocab_stem if x not in list_not]
# total_vocab_lem = [x for x in total_vocab_lem if x not in list_not]
    
# See how many words are per external corpus
dct_base = {x: [] for x in ["all"]}
list_corpora = [
    "all_corpora",
    "2016_hinojosa",
    "2017_Stadthagen",
    "2016_Guasch",
    "2016_Ferre",
    "2021_PerezSanchez",
]
dct_analysis_words = {x: dct_base.copy() for x in list_corpora}
dct_analysis_stem = {x: dct_base.copy() for x in list_corpora}
dct_analysis_lem = {x: dct_base.copy() for x in list_corpora}
for i, sonnet in dct_sonnets.items():
    print("Iter", i)

    vocab = list(set(sonnet["words"]))
    vocab_stem = list(set(sonnet["words_stem_lower"]))
    vocab_lem = list(set(sonnet["words_lem_lower"]))

    # Remove interjections
    vocab = [x for x in vocab if x not in list_not]
    vocab_stem = [x for x in vocab_stem if x not in list_not]
    vocab_lem = [x for x in vocab_lem if x not in list_not]
    
    # Remove accents
    """
    vocab = [unidecode.unidecode(x) for x in vocab]
    vocab_stem = [unidecode.unidecode(x) for x in vocab_stem]
    vocab_lem = [unidecode.unidecode(x) for x in vocab_lem]
    """

    ### Words inside Lexicons
    # df_all
    vocab_iter = [x for x in vocab if x in list(df_all["Word"].values)]
    vocab_stem_iter = [x for x in vocab_stem if x in list(df_all["word_stem"].values)]
    vocab_lem_iter = [x for x in vocab_lem if x in list(df_all["word_lem"].values)]

    list_add = ["all"] 
    for psycho_tag in list_add:
        dct_analysis_words["all_corpora"][psycho_tag] = list(
            set(dct_analysis_words["all_corpora"][psycho_tag] + vocab_iter)
        )
        dct_analysis_stem["all_corpora"][psycho_tag] = list(
            set(dct_analysis_stem["all_corpora"][psycho_tag] + vocab_stem_iter)
        )
        dct_analysis_lem["all_corpora"][psycho_tag] = list(
            set(dct_analysis_lem["all_corpora"][psycho_tag] + vocab_lem_iter)
        )
        
    # df_raw_1 (2016 Hinojosa)
    vocab_iter = [x for x in vocab if x in list(df_raw_1["Word"].values)]
    vocab_stem_iter = [x for x in vocab_stem if x in list(df_raw_1["word_stem"].values)]
    vocab_lem_iter = [x for x in vocab_lem if x in list(df_raw_1["word_lem"].values)]

    list_add = ["all"]
    for psycho_tag in list_add:
        dct_analysis_words["2016_hinojosa"][psycho_tag] = list(
            set(dct_analysis_words["2016_hinojosa"][psycho_tag] + vocab_iter)
        )
        dct_analysis_stem["2016_hinojosa"][psycho_tag] = list(
            set(dct_analysis_stem["2016_hinojosa"][psycho_tag] + vocab_stem_iter)
        )
        dct_analysis_lem["2016_hinojosa"][psycho_tag] = list(
            set(dct_analysis_lem["2016_hinojosa"][psycho_tag] + vocab_lem_iter)
        )
        
    # df_raw_3 (2017 Stadthagen)
    vocab_iter = [x for x in vocab if x in list(df_raw_3["Word"].values)]
    vocab_stem_iter = [x for x in vocab_stem if x in list(df_raw_3["word_stem"].values)]
    vocab_lem_iter = [x for x in vocab_lem if x in list(df_raw_3["word_lem"].values)]

    list_add = ["all"]
    for psycho_tag in list_add:
        dct_analysis_words["2017_Stadthagen"][psycho_tag] = list(
            set(dct_analysis_words["2017_Stadthagen"][psycho_tag] + vocab_iter)
        )
        dct_analysis_stem["2017_Stadthagen"][psycho_tag] = list(
            set(dct_analysis_stem["2017_Stadthagen"][psycho_tag] + vocab_stem_iter)
        )
        dct_analysis_lem["2017_Stadthagen"][psycho_tag] = list(
            set(dct_analysis_lem["2017_Stadthagen"][psycho_tag] + vocab_lem_iter)
        )
        
    # df_raw_4 (2016 Guasch)
    vocab_iter = [x for x in vocab if x in list(df_raw_4["Word"].values)]
    vocab_stem_iter = [x for x in vocab_stem if x in list(df_raw_4["word_stem"].values)]
    vocab_lem_iter = [x for x in vocab_lem if x in list(df_raw_4["word_lem"].values)]

    list_add = ["all"]
    for psycho_tag in list_add:
        dct_analysis_words["2016_Guasch"][psycho_tag] = list(
            set(dct_analysis_words["2016_Guasch"][psycho_tag] + vocab_iter)
        )
        dct_analysis_stem["2016_Guasch"][psycho_tag] = list(
            set(dct_analysis_stem["2016_Guasch"][psycho_tag] + vocab_stem_iter)
        )
        dct_analysis_lem["2016_Guasch"][psycho_tag] = list(
            set(dct_analysis_lem["2016_Guasch"][psycho_tag] + vocab_lem_iter)
        )
        
    # df_raw_5 (2016 Ferre)
    vocab_iter = [x for x in vocab if x in list(df_raw_5["Word"].values)]
    vocab_stem_iter = [x for x in vocab_stem if x in list(df_raw_5["word_stem"].values)]
    vocab_lem_iter = [x for x in vocab_lem if x in list(df_raw_5["word_lem"].values)]

    list_add = ["all"]
    for psycho_tag in list_add:
        dct_analysis_words["2016_Ferre"][psycho_tag] = list(
            set(dct_analysis_words["2016_Ferre"][psycho_tag] + vocab_iter)
        )
        dct_analysis_stem["2016_Ferre"][psycho_tag] = list(
            set(dct_analysis_stem["2016_Ferre"][psycho_tag] + vocab_stem_iter)
        )
        dct_analysis_lem["2016_Ferre"][psycho_tag] = list(
            set(dct_analysis_lem["2016_Ferre"][psycho_tag] + vocab_lem_iter)
        )
        
    # df_raw_6 (2021 Peres-Sanchez)
    vocab_iter = [x for x in vocab if x in list(df_raw_6["Word"].values)]
    vocab_stem_iter = [x for x in vocab_stem if x in list(df_raw_6["word_stem"].values)]
    vocab_lem_iter = [x for x in vocab_lem if x in list(df_raw_6["word_lem"].values)]

    list_add = ["all"]
    for psycho_tag in list_add:
        dct_analysis_words["2021_PerezSanchez"][psycho_tag] = list(
            set(dct_analysis_words["2021_PerezSanchez"][psycho_tag] + vocab_iter)
        )
        dct_analysis_stem["2021_PerezSanchez"][psycho_tag] = list(
            set(dct_analysis_stem["2021_PerezSanchez"][psycho_tag] + vocab_stem_iter)
        )
        dct_analysis_lem["2021_PerezSanchez"][psycho_tag] = list(
            set(dct_analysis_lem["2021_PerezSanchez"][psycho_tag] + vocab_lem_iter)
        )

for psycho_tag in ['all']:
    print("Psycho tag:", psycho_tag)
    l_total = []
    l_total_stem = []
    l_total_lem = []
    for i, sonnet in dct_sonnets.items():
        if psycho_tag == "all" or sonnet[sonnet['index']==i][psycho_tag].values[0]==1:
            l_total = list(set(l_total + sonnet['words']))
            l_total_stem = list(set(l_total_stem + sonnet['words_stem_lower']))
            l_total_lem = list(set(l_total_lem + sonnet['words_lem_lower']))
        else:
            continue
    
    # Remove interjections
    l_total = [x for x in l_total if x not in list_not]
    l_total_stem = [x for x in l_total_stem if x not in list_not]
    l_total_lem = [x for x in l_total_lem if x not in list_not]
    
    """
    # Remove accents
    l_total = [unidecode.unidecode(x) for x in l_total]
    l_total_stem = [unidecode.unidecode(x) for x in l_total_stem]
    l_total_lem = [unidecode.unidecode(x) for x in l_total_lem]
    """

    for corpus in list_corpora:
        dct_analysis_words[corpus][psycho_tag] = len(dct_analysis_words[corpus][psycho_tag])/len(l_total)
        dct_analysis_stem[corpus][psycho_tag] = len(dct_analysis_stem[corpus][psycho_tag])/len(l_total_stem)
        dct_analysis_lem[corpus][psycho_tag] = len(dct_analysis_lem[corpus][psycho_tag])/len(l_total_lem)

df_analysis_words = pd.DataFrame(dct_analysis_words).round(4).reset_index()
df_analysis_stem = pd.DataFrame(dct_analysis_stem).round(4)
df_analysis_stem.columns = [x + ' stem' for x in list(df_analysis_stem.columns)]
df_analysis_stem = df_analysis_stem.reset_index()
df_analysis_lem = pd.DataFrame(dct_analysis_lem).round(4)
df_analysis_lem.columns = [x + ' lem' for x in list(df_analysis_lem.columns)]
df_analysis_lem = df_analysis_lem.reset_index()
df_analysis = df_analysis_words.merge(df_analysis_stem).merge(df_analysis_lem)
list_sorted = sorted(list(df_analysis.columns))
list_sorted.reverse()
df_analysis = df_analysis[list_sorted]

# Save
# df_analysis = df_names.rename(columns={'variable':'index'}).merge(df_analysis, how="left")
# df_analysis = df_analysis.dropna()

if type_used == "ANN":
    df_analysis_prev = pd.read_csv('tables_paper/df_table_words_per_lexicon.csv')
    df_analysis['index'] = 'DISCO (Annotated)'
    df_analysis = df_analysis.append(df_analysis_prev)
    df_analysis.round(2).to_csv('tables_paper/df_table_words_per_lexicon.csv', index=False)

elif type_used == 'DISCO':
    df_analysis_prev = pd.read_csv('tables_paper/df_table_words_per_lexicon.csv')
    df_analysis['index'] = 'DISCO (Original)'
    df_analysis = df_analysis.append(df_analysis_prev)
    df_analysis.round(2).to_csv('tables_paper/df_table_words_per_lexicon.csv', index=False)

elif type_used == 'NEW':
    df_analysis_prev = pd.read_csv('tables_paper/df_table_words_per_lexicon.csv')
    df_analysis['index'] = 'DISCO (NEW)'
    df_analysis = df_analysis.append(df_analysis_prev)
    df_analysis.round(2).to_csv('tables_paper/df_table_words_per_lexicon.csv', index=False)

else:
    df_analysis_prev = pd.read_csv('tables_paper/df_table_words_per_lexicon.csv')
    df_analysis['index'] = 'DISCO (All)'
    df_analysis = df_analysis.append(df_analysis_prev)
    df_analysis.round(2).to_csv('tables_paper/df_table_words_per_lexicon.csv', index=False)


### Relative missing words
# Load Data
file_to_read = open(f"{PATH_RESULTS}/{FILE_NAME}", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

# Get vocabulary sonnets
total_vocab = []
total_vocab_stem = []
total_vocab_lem = []

total_vocab_sxx = []
total_vocab_stem_sxx = []
total_vocab_lem_sxx = []

included_vocab = []
included_vocab_stem = []
included_vocab_lem = []

included_vocab_sxx = []
included_vocab_stem_sxx = []
included_vocab_lem_sxx = []

list_ref_vocab = list(df_all['Word'].values)
list_ref_vocab_stem = list(df_all['word_stem'].values)
list_ref_vocab_lem = list(df_all['word_lem'].values)

for i, sonnet in dct_sonnets.items():
    print(i)
    
    total_vocab = total_vocab + sonnet["words"]
    total_vocab_stem = total_vocab_stem + sonnet["words_stem"]
    total_vocab_lem = total_vocab_lem + sonnet["words_lem"]
    
    included_vocab = included_vocab + [x for x in sonnet["words"] if x in list_ref_vocab]
    included_vocab_stem = included_vocab_stem + [x for x in sonnet["words_stem"] if x in list_ref_vocab_stem]
    included_vocab_lem = included_vocab_lem + [x for x in sonnet["words_lem"] if x in list_ref_vocab_lem]
    
    if i > 4085:
        total_vocab_sxx = total_vocab_sxx + sonnet["words"]
        total_vocab_stem_sxx = total_vocab_stem_sxx + sonnet["words_stem"]
        total_vocab_lem_sxx = total_vocab_lem_sxx + sonnet["words_lem"]
        
        included_vocab_sxx = included_vocab_sxx + [x for x in sonnet["words"] if x in list_ref_vocab]
        included_vocab_stem_sxx = included_vocab_stem_sxx + [x for x in sonnet["words_stem"] if x in list_ref_vocab_stem]
        included_vocab_lem_sxx = included_vocab_lem_sxx + [x for x in sonnet["words_lem"] if x in list_ref_vocab_lem]
            

print(f"Percentage Words {100*len(included_vocab)/len(total_vocab)}")
print(f"Percentage Words Stem {100*len(included_vocab)/len(total_vocab_stem)}")            
print(f"Percentage Words Lem {100*len(included_vocab)/len(total_vocab_lem)}")
print(f"Percentage Words SXX {100*len(included_vocab_sxx)/len(total_vocab_sxx)}")
print(f"Percentage Words Stem SXX {100*len(included_vocab_stem_sxx)/len(total_vocab_stem_sxx)}")
print(f"Percentage Words Lem SXX {100*len(included_vocab_lem_sxx)/len(total_vocab_lem_sxx)}")

"""
Percentage Words 53.40140325148537
Percentage Words Stem 55.1042724766268
Percentage Words Lem 55.1042724766268
Percentage Words SXX 56.86318835337734
Percentage Words Stem SXX 92.4814486395669
Percentage Words Lem SXX 84.50086339664908

"""


# =============================================================================
# Get Statistics (II) - Top Words Missing
# =============================================================================
# Load Data
file_to_read = open(f"{PATH_RESULTS}/{FILE_NAME}", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

# Only...
# type_used = "ANN"
# type_used = 'DISCO'
type_used = 'NEW'

if type_used == "DISCO":
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x <= 4085}
    
elif type_used == 'NEW':
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x > 4085}
    
elif type_used == 'ANN':
    dct_sonnets = {x:y for x,y in dct_sonnets.items() if x in list_idx_ann}

list_words_miss = []
list_words_miss_lem = []
df_aux = pd.DataFrame()
for i, sonnet in dct_sonnets.items():
    print("Iter", i)
    
    vocab = list(sonnet['words'])
    vocab_stem = list(sonnet['words_stem'])
    vocab_lem = list(sonnet['words_lem'])
    
    # if 'sentir' in vocab_lem:
    #     break
    
    # df_all
    df_base = pd.DataFrame({
        # 'Word': vocab,
        'word_stem': vocab_stem,
        'word_lem': vocab_lem
        }
        )
    df_iter = df_base.merge(df_all[['word_stem']].drop_duplicates(), how="inner")    
    df_iter = df_base[~df_base['word_stem'].isin(list(df_iter['word_stem'].values))]
    
    # vocab_iter = list(df_iter['Word'].values)
    vocab_stem_iter = list(df_iter['word_stem'].values)
    vocab_lem_iter = list(df_iter['word_lem'].values)
    
    # vocab_iter = [x for x in vocab if x not in list(df_all['Word'].values)]
    # vocab_stem_iter = [x for x in vocab_stem if x not in list(df_all['word_stem'].values)]
    # vocab_lem_iter = [x for x in vocab_lem if x not in list(df_all['word_lem'].values)]
    
    # if 'labio' in vocab_lem_iter:break
    
    # list_words_miss = list_words_miss + vocab_iter
    list_words_miss_lem = list_words_miss_lem + vocab_lem_iter
    df_aux = df_aux.append(df_base)
    
df_aux = df_aux.drop_duplicates()
df_analysis = pd.DataFrame({'words': list_words_miss_lem}) 
df_analysis['count'] = 1
df_analysis = df_analysis.groupby(by=['words']).sum().reset_index().sort_values(by=['count'], ascending=False)
df_analysis = df_analysis[~df_analysis['words'].isin(list_not)]

# df_analysis['fixed_word'] = ""
# df_analysis = df_analysis.replace("\u0161", "").reset_index(drop=True)
# df_analysis.to_excel("fixed_words.xlsx")

"""
df_analysis.head(50).to_csv('tables_paper/df_table_top_missing.csv', index=False)

# df_analysis = df_analysis[~df_analysis['words'].isin(['holer', 'alguien', 'amir', 'déjamar'])]

COLOR_3 = "#126782"
plt.figure(figsize=(12, 8), dpi=200)
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.5)
plot_fig = sns.barplot(
    data=df_analysis.head(25), 
    x="words",
    y="count",
    color=COLOR_3
    )
plt.title("Top 25 Missing Words in Lexicons (XXth century subset)")
plt.xlabel('Words')
plt.ylabel('N Appearances')
plot_fig.set_xticklabels(
    plot_fig.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('figures/top_25_missing_lexicon', dpi=200)
"""

### WordCloud
from wordcloud import WordCloud

# Remove words that are not lemmatized correctly
df_modifications = pd.read_excel("datasets/fixed_words.xlsx").dropna(subset=['fixed_word'])
df_analysis = df_analysis[~df_analysis['words'].isin(list(df_modifications['words'].values))]

# Dummy text
text = "en"
for i, row in df_analysis.iterrows():
    text = text + f",{row['words']}"*row['count'] + '\n'

# Generate a word cloud image
# wordcloud = WordCloud().generate(text)

# take relative word frequencies into account, lower max_font_size
if type_used == 'NEW':
    wordcloud = WordCloud(
        background_color="white",
        max_words=len(df_analysis),
        max_font_size=40,
        relative_scaling=.5,
        collocations=False
        ).generate(text)
    plt.figure(figsize=(12, 8), dpi=200)
    plt.rcParams.update({'font.size': 18})
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Missing Words (XXth century subset)\n")
    plt.savefig('figures/wordcloud_missing_xxth', dpi=200)
    plt.show()
else:
    wordcloud = WordCloud(
        background_color="white",
        max_words=len(df_analysis),
        max_font_size=40,
        relative_scaling=.5,
        collocations=False
        ).generate(text)
    plt.figure(figsize=(12, 8), dpi=200)
    plt.rcParams.update({'font.size': 18})
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Missing Words (All)\n")
    plt.savefig('figures/wordcloud_missing_full', dpi=200)
    plt.show()

for i, sonnet in dct_sonnets.items():
    if 'bruno' in sonnet['words_lem']: break


# =============================================================================
# Get Statistics (III) - Number of Words per Sonnet
# =============================================================================
import re 

# Load Data
file_to_read = open(f"{PATH_RESULTS}/dct_sonnets_input_mod_v4", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

### 1. Histograms with StopWords
# Mean words: 91.8 Std: 8.7 | Non-stop: mean words: 43.5 std: 5.2
df_n_words = pd.DataFrame()
for i, sonnet in dct_sonnets.items():
    n_words = len(re.findall(r'\w+', sonnet['text'],flags = re.UNICODE))
    df_n_words = df_n_words.append(
        pd.DataFrame({
            "index": [i], 
            "n_words": [n_words],
            "n_words_nostop" : [len(sonnet['words'])]
            },
            index=[0]
            )
    )
print(
      df_n_words['n_words'].mean(), 
      df_n_words['n_words'].std()
      )
print(
      df_n_words['n_words_nostop'].mean(), 
      df_n_words['n_words_nostop'].std()
      )

# Mean words: 94 Std: 10.3 | Non-stop: mean words: 41.6 std: 6.7
dct_sonnets_new = {x:y for x,y in dct_sonnets.items() if x > 4085}
df_n_words_sxx = pd.DataFrame()
for i, sonnet in dct_sonnets_new.items():
    n_words = len(re.findall(r'\w+', sonnet['text'],flags = re.UNICODE))
    df_n_words_sxx = df_n_words_sxx.append(
        pd.DataFrame({
            "index": [i],
            "n_words": [n_words],
            "n_words_nostop" : [len(sonnet['words'])]
            }, index=[0]
            )
    )
print(
      df_n_words_sxx['n_words'].mean(), 
      df_n_words_sxx['n_words'].std()
      )
print(
      df_n_words_sxx['n_words_nostop'].mean(), 
      df_n_words_sxx['n_words_nostop'].std()
      )
    
# Mean words: 92.2 Std: 8.73 | Non-stop: mean words: 44.7 std: 5.0
dct_sonnets_ann = {x:y for x,y in dct_sonnets.items() if x in list_idx_ann}
df_n_words_ann = pd.DataFrame()
for i, sonnet in dct_sonnets_ann.items():
    n_words = len(re.findall(r'\w+', sonnet['text'],flags = re.UNICODE))
    df_n_words_ann = df_n_words_ann.append(
        pd.DataFrame({
            "index": [i],
            "n_words": [n_words],
            "n_words_nostop" : [len(sonnet['words'])]
            }, 
            index=[0]
            )
    )
print(
      df_n_words_ann['n_words'].mean(), 
      df_n_words_ann['n_words'].std()
      )
print(
      df_n_words_ann['n_words_nostop'].mean(), 
      df_n_words_ann['n_words_nostop'].std()
      )
    
df_n_words['corpus'] = 'All'
df_n_words_sxx['corpus'] = 'S.XX'
df_n_words_ann['corpus'] = 'Annotated (DISCO)'
df_plot =  df_n_words.append(df_n_words_sxx).append(df_n_words_ann)

# Plot results
# COLOR_1 = "#0077b6"
# COLOR_2 = "#ffddd2"
# COLOR_3 = "#4d908e"
# COLOR_1 = "#8e7dbe"
# COLOR_2 = "#fcaf58"
# COLOR_3 = "#87bba2"
COLOR_1 = "#5e6472"
COLOR_2 = "#ffa69e"
COLOR_3 = "#aed9e0"

plt.figure(figsize=(12, 8), dpi=200)
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.5)
sns.histplot(
    data=df_n_words, 
    x="n_words",
    stat="density",
    color=COLOR_1
    )
sns.histplot(
    data=df_n_words_sxx, 
    x="n_words",
    stat="density",
    color=COLOR_2
    )
sns.histplot(
    data=df_n_words_ann, 
    x="n_words",
    stat="density",
    color=COLOR_3
    )
plt.title("Histogram of Words per Sonnet")
plt.xlabel('Number of Words')
plt.ylabel('Number of Sonnets')

# Other [Legend]
# topbar = plt.Rectangle((0,0), 1, 1, fc = COLOR_RED, edgecolor = 'none')
# bottombar = plt.Rectangle((0,0), 1, 1, fc = COLOR_BLUE, edgecolor = 'none')
fontsize = 12
bbox_to_anchor = (10, 10)
bottombar = plt.Line2D(range(1),
                       range(1), 
                       marker = "o", 
                       color = COLOR_1,
                       markersize=fontsize-2,
                       linewidth=0)
middlebar = plt.Line2D(range(1),
                    range(1), 
                    marker = "o", 
                    color = COLOR_2,
                    markersize=fontsize-2,
                    linewidth=0)
topbar = plt.Line2D(range(1),
                    range(1), 
                    marker = "o", 
                    color = COLOR_3,
                    markersize=fontsize-2,
                    linewidth=0)
l = plt.legend([bottombar, middlebar, topbar],
                    ['All', 'XXth century', 'Annotated (DISCO)'],
                    loc = "upper right",
                    ncol = 1,
                    # bbox_to_anchor = bbox_to_anchor,
                    prop={'size':fontsize})
l.draw_frame(True)
plt.savefig('figures/histogram_words_stopwords.png', dpi=200)
# plt.savefig(output_folder + '/histogram_words.svg', dpi=500)
plt.show()


### 2. Histograms without StopWords
COLOR_1 = "#5e6472"
COLOR_2 = "#ffa69e"
COLOR_3 = "#aed9e0"

plt.figure(figsize=(12, 8), dpi=200)
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.5)
sns.histplot(
    data=df_n_words, 
    x="n_words_nostop",
    stat="density",
    color=COLOR_1
    )
sns.histplot(
    data=df_n_words_sxx, 
    x="n_words_nostop",
    stat="density",
    color=COLOR_2
    )
sns.histplot(
    data=df_n_words_ann, 
    x="n_words_nostop",
    stat="density",
    color=COLOR_3
    )
plt.title("Histogram of Words per Sonnet (without stopwords)")
plt.xlabel('Number of Words')
plt.ylabel('Number of Sonnets')

# Other [Legend]
# topbar = plt.Rectangle((0,0), 1, 1, fc = COLOR_RED, edgecolor = 'none')
# bottombar = plt.Rectangle((0,0), 1, 1, fc = COLOR_BLUE, edgecolor = 'none')
fontsize = 12
bbox_to_anchor = (10, 10)
bottombar = plt.Line2D(range(1),
                       range(1), 
                       marker = "o", 
                       color = COLOR_1,
                       markersize=fontsize-2,
                       linewidth=0)
middlebar = plt.Line2D(range(1),
                    range(1), 
                    marker = "o", 
                    color = COLOR_2,
                    markersize=fontsize-2,
                    linewidth=0)
topbar = plt.Line2D(range(1),
                    range(1), 
                    marker = "o", 
                    color = COLOR_3,
                    markersize=fontsize-2,
                    linewidth=0)
l = plt.legend([bottombar, middlebar, topbar],
                    ['All', 'XXth century', 'Annotated (DISCO)'],
                    loc = "upper right",
                    ncol = 1,
                    # bbox_to_anchor = bbox_to_anchor,
                    prop={'size':fontsize})
l.draw_frame(True)
plt.savefig('figures/histogram_words_not_stopwords.png', dpi=200)



# =============================================================================
# Dataset description (I)
# =============================================================================
# Load data
df_add = pd.read_csv("train_dataset.csv")
df_gt = pd.read_csv("test_dataset.csv")

df_names = pd.read_csv(
    f"{PATH_GROUND_TRUTH}/variable_names_en.csv", encoding="latin-1")

# Affective Categories
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

# See instances per class
list_categories = list(df_add.drop(columns=['index', 'text_original']).columns)
df_analysis_test = pd.DataFrame()
df_analysis_train = pd.DataFrame()
for category in list_categories:
    df_iter_test = df_gt[[category]].value_counts().reset_index()
    df_iter_test.columns = ['class', 'n_instances (test)']
    df_iter_test['category'] = category
    df_iter_train = df_add[[category]].value_counts().reset_index()
    df_iter_train.columns = ['class', 'n_instances (train)']
    df_iter_train['category'] = category
    
    df_analysis_test = df_analysis_test.append(df_iter_test)
    df_analysis_train = df_analysis_train.append(df_iter_train)

df_analysis_train = df_analysis_train.reset_index(drop=True)
df_analysis_test = df_analysis_test.reset_index(drop=True)

df_analysis = df_analysis_train.merge(df_analysis_test)
df_analysis = df_analysis[[
    'category', 'class', 'n_instances (train)', 'n_instances (test)'
    ]]

# Prepare tables
df_analysis_psycho = df_analysis[~df_analysis['category'].isin(list_aff)].copy()
df_analysis_emotions = df_analysis[df_analysis['category'].isin(list_aff)].copy()

df_analysis_psycho = df_names.rename(columns={'es_name':'category'}).merge(df_analysis_psycho, how="right")
df_analysis_emotions = df_names.rename(columns={'es_name':'category'}).merge(df_analysis_emotions, how="right")

df_analysis_psycho = (
    df_analysis_psycho.pivot(
        index='en_name', 
        columns='class',
        values=['n_instances (train)', 'n_instances (test)']
        )
    )

df_analysis_emotions = (
    df_analysis_emotions.pivot(
        index='en_name', 
        columns='class',
        values=['n_instances (train)', 'n_instances (test)']
        )
    )

df_analysis_psycho = df_analysis_psycho.reset_index()
df_analysis_emotions = df_analysis_emotions.reset_index()

df_analysis_psycho.to_csv("tables_paper/df_info_psychological_data.csv", index=False)
df_analysis_emotions.to_csv("tables_paper/df_info_emotions_data.csv", index=False)


# =============================================================================
# Dataset description (II)
# =============================================================================
# Load Data
file_to_read = open(f"{PATH_RESULTS}/dct_sonnets_input_mod_v4", "rb")
dct_sonnets = pickle.load(file_to_read)
file_to_read.close()

df_meta = pd.concat([
    pd.DataFrame({
        'index':[i],
        'author':[sonnet['author']],
        'period':[sonnet['year']]
        }) 
    for i, sonnet in dct_sonnets.items()
    ]
    )

# for i, sonnet in dct_sonnets.items():
#     if sonnet['author'] == 'Manuel de Zequeira y Arango' and sonnet['year']!= 'Sonetos por autor': break

df_meta_check = df_meta[df_meta['period']=='Sonetos por autor'].copy()

dct_change = {
    318: 'Sonetos del siglo XVIII',
    323: 'Sonetos del siglo XVIII',
    422: 'Sonetos del siglo XVIII',
    508: 'Sonetos del siglo XVIII',
    1713: 'Sonetos del siglo XVIII',
    3369: 'Sonetos del siglo XVIII',
    3370: 'Sonetos del siglo XVIII',
    4033: 'Sonetos del siglo XVIII'
    }

df_meta['period'] = df_meta.apply(
    lambda x: x['period'] if x['index'] not in list(dct_change.keys()) 
    else dct_change[x['index']], axis = 1
    )


df_periods = df_meta[['period']].drop_duplicates()
df_meta['n_sonnets'] = 1
df_meta_agg = df_meta.groupby(by=['period']).sum().reset_index().copy()

df_meta_agg = (
    df_meta_agg
    .replace("Sonetos del siglo XIX", "XIXth century")
    .replace("Sonetos del siglo XVIII", "XVIIIth century")
    .replace("Sonetos del Siglo XX", "XXth century")
    .replace("Sonetos del siglo XV al XVII", "XVth to XVIIth centuries")
    )
df_meta_agg['index'] = [4,3,1,2]
df_meta_agg = df_meta_agg.sort_values(by=['index'], ascending=True)
COLOR_3 = "#126782"

plt.figure(figsize=(12, 8), dpi=200)
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.5)
plot_fig = sns.barplot(
    data=df_meta_agg, 
    x="period",
    y="n_sonnets",
    color=COLOR_3
    )

plt.title("Number of Sonnets per Period")
plt.ylabel('Number of Sonnets')
plt.xlabel('Period')
# plot_fig.set_xticklabels(
#     plot_fig.get_xticklabels(), rotation=90, horizontalalignment='right')

plt.savefig('figures/sonnets_per_period', dpi=200)



for i, sonnet in dct_sonnets.items():
    print(i)
    if sonnet['year']== 'Sonetos por autor': break
        print("Changing year...")
        dct_sonnets[i]['year'] = 'Sonetos del siglo XVIII'

### Save Results
with open(f"{PATH_RESULTS}/dct_sonnets_input_mod_v4_fix", 'wb') as handle:
    pickle.dump(dct_sonnets, handle)


