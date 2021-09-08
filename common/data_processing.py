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
import stanza
import spacy_stanza
# import unidecode

from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from transformers import BertTokenizer, BertModel

from common.tools import get_files, file_presistance
from common.config import (
    PATH_POEMS, PATH_RESULTS, PATH_AFF_LEXICON, PATH_POEMS_SXX
    )

nlp = spacy.load('es_core_news_md')
stemmer = SnowballStemmer("spanish")
# stanza.download("es")
nlp_st = spacy_stanza.load_pipeline("es")

list_not = [
    'ay', 'oh', 'cualquiera', 'cuyo', 'do', 'cuán', 'ah', 'd', 'acá', 'ey',
    'aquesta', 'loor', 'aqueste', 'vo', 's', 'ca'
    ]

def word_grams(words, lim_min=2, lim_max=5):
    """
    # TODO
    Function to obtain different ngrams from a word.
    It gives back the list containing those ngrams as
    well as the original word.
   
    """
    s = []
    for n in range(lim_min, lim_max):
        for ngram in ngrams(words, n):
            s.append(''.join(str(i) for i in ngram))
            break # para coger solo el ngrama de inicio
    return s

def word_preprocessing(text):
    """
    # TODO
    Generic function that recieves a str array to be preproccesed. It performs:
        - tokenization
        - decapitalization
        - stop words removal
        - filters non-words characters
        - lemmatization
    """
   
    # Tokenize each sentence
    words = re.findall(r'\w+', text,flags = re.UNICODE)
    # Upper case to lowercase
    words = [w.lower() for w in words]
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('spanish')]
    # Remove non alphanumeric characters
    words = [w for w in words if w.isalpha()]
    # Lemmatize words for its use in affective features
    words_lem = [
        token.lemma_ if (token.tag_.split('=')[-1] != 'Sing')
        else w for w in words for token in nlp(w)
        ] # lemmatize only not-singular words and verbs
    # n-grams for those words
    words_lem_ngrams = list(
        set([x for w in words_lem for x in word_grams(w, len(w)-1, len(w)+1)]))
    # words lem with stop words and with uppercases
    words_lem_complete = [
        token.lemma_ if (token.tag_.split('=')[-1] != 'Sing')
        else w for w in re.findall(r'\w+', text,flags = re.UNICODE)
        for token in nlp(w)
        ] # lemmatize only not-singular words and verbs
   
    return words, words_lem, words_lem_ngrams, words_lem_complete

def text_preprocessing_v3(text, lemma_sentence=False):
    """
    # TODO
    Generic function that recieves a str array ato be preproccesed. It performs:
        - tokenization
        - decapitalization
        - stop words removal
        - filters non-words characters
        - lemmatization
    """
   
    # Tokenize each sentence
    if lemma_sentence:
        # Lemmatization (I)
        doc = nlp_st(text)
        words = [token.lemma_ if token.pos_=='VERB' else token for token in doc]
        words = [str(x) for x in words]
    else:
        words = re.findall(r'\w+', text,flags = re.UNICODE)
       
    # Upper case to lowercase
    words = [w.lower() for w in words]
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('spanish')]
   
    if True:
        # Remove more stopwords
        words = [w for w in words if nlp.vocab[w].is_stop == False]
     
    # Remove non alphanumeric characters
    words = [w for w in words if w.isalpha()]
   
    # Lemmatization (I)
    # words_lem = [token.lemma_ if token.pos_=='VERB' else token for w in words for token in nlp_st(w)]
    # words_pos = [token.pos_  for w in words for token in nlp_st(w)]
    # words_lem = [token.lemma_ for w in words for token in nlp_st(w)]
   
    # Lematization (II)
    words_lem = [str(x) for x in words]
    words_lem = [token.lemma_ for w in words_lem for token in nlp(w)]
   
    # Ensure one word per str
    words_lem = [w.split(" ")[0] if len(w.split(" "))>1 else w for w in words_lem]

    # Stemming
    words_stem = [stemmer.stem(i) for i in words_lem]
   
    if len(words)==0:
        words = [""]
        words_stem = [""]
        words_lem = [""]
   
    return words, words_stem, words_lem

def text_preprocessing(
        text,
        lemma_sentence=False,
        keep_upper=False,
        df_modifications=pd.DataFrame()
        ):
    """
    # TODO
    Generic function that recieves a str array ato be preproccesed. It performs:
        - tokenization
        - decapitalization
        - stop words removal
        - filters non-words characters
        - lemmatization
    """
   
    # Pipeline for whole sentences
    if lemma_sentence:
        # Lemmatization (I)
        doc = nlp_st(text)
        words = [token.lemma_ for token in doc]
       
        if keep_upper:
            dct_pos = {
                str(token.lemma_).lower():[token.lemma_, token.pos_]
                for token in doc
                }
           
        words = [str(x) for x in words]
        # Upper case to lowercase
        words = [w.lower() for w in words]
       
        if keep_upper:
            words = [
                x if dct_pos[x][1]!='PROPN' and dct_pos[x][1]!='LOC'
                else dct_pos[x][0] for x in words]
       
        # Remove stopwords
        words = [w for w in words if w not in stopwords.words('spanish')]
        # Remove more stopwords
        words = [w for w in words if nlp.vocab[w].is_stop == False]
        # Remove non alphanumeric characters
        words = [w for w in words if w.isalpha()]
        # Ensure one word per str
        words_lem = [w.split(" ")[0] if len(w.split(" "))>1 else w for w in words]
        # Remove interjections & other
        words_lem = [x for x in words_lem if x not in list_not]
       
        # Manual modifications
        if not df_modifications.empty:
            words_lem = [
                x if x not in list(df_modifications['words'])
                else df_modifications[df_modifications['words']==x]['fixed_word'].values[0]
                for x in words_lem
                ]
       
        # Stemming
        words_stem = [stemmer.stem(i) for i in words_lem]
       
    # Pipeline for individual words
    else:
        words = re.findall(r'\w+', text,flags = re.UNICODE)
        # Upper case to lowercase
        words = [w.lower() for w in words]
        # Remove stopwords
        words = [w for w in words if w not in stopwords.words('spanish')]
        # Remove more stopwords
        words = [w for w in words if nlp.vocab[w].is_stop == False]
        # Remove non alphanumeric characters
        words = [w for w in words if w.isalpha()]
        # Lematization (II)
        words_lem = [str(x) for x in words]
        words_lem = [token.lemma_ for w in words_lem for token in nlp(w)]
        # Ensure one word per str
        words_lem = [w.split(" ")[0] if len(w.split(" "))>1 else w for w in words_lem]
        # Remove interjections & other
        words_lem = [x for x in words_lem if x not in list_not]
       
        # Manual modifications
        if not df_modifications.empty:
            words_lem = [
                x if x not in list(df_modifications['words'])
                else df_modifications[df_modifications['words']==x]['fixed_word'].values[0]
                for x in words_lem
                ]
       
        # Stemming
        words_stem = [stemmer.stem(i) for i in words_lem]
   
    if len(words)==0:
        words = [""]
        words_stem = [""]
        words_lem = [""]
   
    return words, words_stem, words_lem


def aux_function_get_words(text):
    words = re.findall(r'\w+', text,flags = re.UNICODE)
    # Upper case to lowercase
    words = [w.lower() for w in words]
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('spanish')]
    # Remove more stopwords
    words = [w for w in words if nlp.vocab[w].is_stop == False]
    # Remove non alphanumeric characters
    words = [w for w in words if w.isalpha()]
    # Remove interjections & other
    words_lem = [x for x in words_lem if x not in list_not]
    return words
    


def parse_poem(document):
    """
    # TODO
    """
    text = ''
    for paragraph in  document["lg"]:
        for line in paragraph["l"]:
            # Line with content
            try:
                text += line["#"+"text"]
            # Empty line
            except:
                text += ''
            text += '\n'
        text += '\n'
    return text


def parse_stanza(document):
    """
    # TODO
    """
   
    key = 0
    stanzas = {}
    for paragraph in  document["lg"]:
        text = ''
        for line in paragraph["l"]:
            # Line with content
            try:
                text += line["#"+"text"]
            # Empty line
            except:
                text += ''
            text += '\n'
        stanzas[key] = {'text':text}
        key += 1
    return stanzas

def doc2text(doc, docs, type_doc):
    """
    # TODO
    This function recieves an xml document from the DISCO per-author files and
    retrieves the different sonnets content as well as author, sonnet name,
    id and year metadata. It gives back a dict with the different
    sonnets and metadata available for that xml document.
   
    """
   
    if type_doc=="author":
        dct_aux = {}
        j = 0
        # Iterate through poems in that doc
        iter_dict = doc["TEI"]["text"]["body"]
        # Metadata
        author = doc['TEI']['text']['front']['div']['head']
        year = doc['TEI']['text']['front']['div']['p']
                       
        for doc_poem in iter_dict.values():
            ################
            # One Sonnet per Author
            ################
            if 'lg' in doc_poem:
                # Metadata
                title = doc_poem['head']
                id_doc = doc_poem['@'+'xml:id']
               
                # Sonnet sequence
                if doc_poem['@'+'type'] == 'sonnet-sequence':
                    # with two or more parts
                    if type(doc_poem["lg"])==list:
                        for sonnet in doc_poem["lg"]:
                            text = parse_poem(sonnet)
                            dct_stanzas = parse_stanza(sonnet)
                            docs.append(text)
                            dct_aux[j] = {
                                'title': title,
                                'text': text,
                                'id_doc': id_doc,
                                'dct_stanzas': dct_stanzas
                                }
                            j += 1
                    # with one part
                    else:
                        for key, value in doc_poem.items():
                            if key=='lg':
                                sonnet=value
                                text = parse_poem(sonnet)
                                dct_stanzas = parse_stanza(sonnet)
                                docs.append(text)
                                dct_aux[j] = {
                                    'title': title,
                                    'text': text,
                                    'id_doc': id_doc,
                                    'dct_stanzas':dct_stanzas
                                    }
                                j += 1  
                               
                # In other case
                else:
                    text = parse_poem(doc_poem)
                    dct_stanzas = parse_stanza(doc_poem)
                    docs.append(text)
                    dct_aux[j] = {'title': title,
                                  'text': text,
                                  'id_doc': id_doc,
                                  'dct_stanzas': dct_stanzas
                                  }
                    j += 1
           
            ###############
            # More than one sonnet per Author
            ##############
            else:
                for doc_sonnet in doc_poem:
                    # Metadata
                    title = doc_sonnet['head']
                    id_doc = doc_sonnet['@'+'xml:id']
                   
                    # Unique sonnets
                    if 'lg' in doc_sonnet and doc_sonnet['@'+'type']=='sonnet':
                        text = parse_poem(doc_sonnet)
                        dct_stanzas = parse_stanza(doc_sonnet)
                        docs.append(text)
                        dct_aux[j] = {
                            'title': title,
                            'text': text,
                            'id_doc': id_doc,
                            'dct_stanzas': dct_stanzas
                            }
                        j += 1
               
                    # Sonnet sequence  
                    elif 'lg' in doc_sonnet and doc_sonnet['@'+'type']=='sonnet-sequence':
                        # with two or more parts
                        if type(doc_sonnet["lg"])==list:
                            for sonnet in doc_sonnet["lg"]:
                                text = parse_poem(sonnet)
                                dct_stanzas = parse_stanza(sonnet)
                                docs.append(text)
                                dct_aux[j] = {
                                    'title': title,
                                    'text': text,
                                    'id_doc': id_doc,
                                    'dct_stanzas': dct_stanzas
                                    }
                                j += 1
                        # with one part
                        else:
                            for key, value in doc_sonnet.items():
                                if key=='lg':
                                    sonnet=value
                                    text = parse_poem(sonnet)
                                    dct_stanzas = parse_stanza(sonnet)
                                    docs.append(text)
                                    dct_aux[j] = {'title': title,
                                                  'text': text,
                                                  'id_doc': id_doc,
                                                  'dct_stanzas': dct_stanzas
                                                  }
                                    j += 1                    
   
            for key, value in dct_aux.items():
                dct_aux[key].update({'author':author, 'year':year})
               
    else:
        author = doc['TEI']['teiHeader']['fileDesc']['titleStmt']['author']['#text']
        year = doc['TEI']['teiHeader']['fileDesc']['sourceDesc']['bibl']['hi']['#text']
        title = doc['TEI']['teiHeader']['fileDesc']['titleStmt']['title']['#text']
        id_doc = doc['TEI']['teiHeader']['fileDesc']['titleStmt']['@about']
                   
        # Iterate through stanzas in that doc
        sonnet = doc["TEI"]["text"]["body"]
        text = parse_poem(sonnet)
        dct_stanzas = parse_stanza(sonnet)
        docs.append(text)
        dct_aux = {0:{
            'title': title,
            'text': text,
            'id_doc': id_doc,
            'dct_stanzas': dct_stanzas,
            'author': author,
            'year': year}}
               
    return docs, dct_aux


def affectiveFeaturesText(i, df_iter):
    """
    Get affective features for a sonnet

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.
    df_iter : TYPE
        DESCRIPTION.

    Returns
    -------
    df_aux : TYPE
        DESCRIPTION.

    """
    ### Get features for this sonnet
    # Valence_Mean, Valence_SD
    valence_mean = df_iter["Valence_Mean"].mean()
    valence_sd = df_iter["Valence_SD"].mean()
   
    # Arousal_Mean, Arousal_SD
    arousal_mean = df_iter["Arousal_Mean"].mean()
    arousal_sd = df_iter["Arousal_SD"].mean()
   
    # Happiness_Mean, Happiness_SD
    happiness_mean = df_iter["Happiness_Mean"].mean()
    happiness_sd = df_iter["Happiness_SD"].mean()
   
    # Anger_Mean, Anger_SD
    anger_mean = df_iter["Anger_Mean"].mean()
    anger_sd = df_iter["Anger_SD"].mean()
   
    # Sadness_Mean, Sadness_SD
    sadness_mean = df_iter["Sadness_Mean"].mean()
    sadness_sd = df_iter["Sadness_SD"].mean()
   
    # Fear_Mean, Fear_SD
    fear_mean = df_iter["Fear_Mean"].mean()
    fear_sd = df_iter["Fear_SD"].mean()
   
    # Disgust_Mean, Disgust_SD
    disgust_mean = df_iter["Disgust_Mean"].mean()
    disgust_sd = df_iter["Disgust_SD"].mean()
   
    # Concreteness_Mean, Concreteness_SD
    concreteness_mean = df_iter["Concreteness_Mean"].mean()
    concreteness_sd = df_iter["Concreteness_SD"].mean()
   
    # Imageability_Mean, Imageability_SD
    imageability_mean = df_iter["Imageability_Mean"].mean()
    imageability_sd = df_iter["Imageability_SD"].mean()
   
    # Context_Availability_Mean, Context_Availability_SD
    ca_mean = df_iter["Context_Availability_Mean"].mean()
    ca_sd = df_iter["Context_Availability_SD"].mean()
   
    # Features: Min(Aro), Max(Aro), Min(Val), Max(Val), ValenceSpan, ArousalSpan
    max_arousal = df_iter["Arousal_Mean"].max()
    min_arousal = df_iter["Arousal_Mean"].min()
    max_valence = df_iter["Valence_Mean"].max()
    min_valence = df_iter["Valence_Mean"].min()
   
    arousal_span = max_arousal - min_arousal
    valence_span = max_valence - min_valence
   
    # Features: Nº words
    n_words = len(df_iter)
   
    # Features: Vector Correlation (Aro), Vector Correlation (Val)
    CorAro = np.round(
        df_iter.
        reset_index()[["index", "Arousal_Mean"]].
        corr()["Arousal_Mean"]["index"], 4
    )
    CorVal = np.round(
        df_iter.
        reset_index()[["index", "Valence_Mean"]].
        corr()["Valence_Mean"]["index"], 4
    )
   
    # Absolute Corr Value
    AbsCorAro = np.abs(CorAro)
    AbsCorVal = np.abs(CorVal)
   
    # Features: Sigma (Aro), Sigma (Val)
    sigma_aro = arousal_mean / (1 / np.sqrt(n_words))
    sigma_val = valence_mean / (1 / np.sqrt(n_words))
   
    # Create iter dataframe
    df_aux = pd.DataFrame(
        {
            "index": [i],
            "valence_mean": [valence_mean],
            "valence_sd": [valence_sd],
            "arousal_mean": [arousal_mean],
            "arousal_sd": [arousal_sd],
            "happiness_mean": [happiness_mean],
            "happiness_sd": [happiness_sd],
            "anger_mean": [anger_mean],
            "anger_sd": [anger_sd],
            "sadness_mean": [sadness_mean],
            "sadness_sd": [sadness_sd],
            "fear_mean": [fear_mean],
            "fear_sd": [fear_sd],
            "disgust_mean": [disgust_mean],
            "disgust_sd": [disgust_sd],
            "concreteness_mean": [concreteness_mean],
            "concreteness_sd": [concreteness_sd],
            "imageability_mean": [imageability_mean],
            "imageability_sd": [imageability_sd],
            "cont_ava_mean": [ca_mean],
            "cont_ava_sd": [ca_sd],
            "max_arousal": [max_arousal],
            "min_arousal": [min_arousal],
            "max_valence": [max_valence],
            "min_valence": [min_valence],
            "arousal_span": [arousal_span],
            "valence_span": [valence_span],
            "n_words": [n_words],
            "CorAro": [CorAro],
            "CorVal": [CorVal],
            "AbsCorAro": [AbsCorAro],
            "AbsCorVal": [AbsCorVal],
            "sigma_aro": [sigma_aro],
            "sigma_val": [sigma_val],
        },
        index=[0],
    )
   
    return df_aux


def customSentenceFunctions(
        words_stem, words_lem, model, df_weights, model_used = "sentence-transformer",
        tokenizer = ""
        ):
    """
    Create custom sentence embeddings from individual words.

    Parameters
    ----------
    words_stem : TYPE
        DESCRIPTION.
    words_lem : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    df_weights : TYPE
        DESCRIPTION.
    model_used : TYPE, optional
        DESCRIPTION. The default is "sentence-transformer".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_sentence_max : TYPE
        DESCRIPTION.
    df_sentence_span : TYPE
        DESCRIPTION.
    df_sentence_median : TYPE
        DESCRIPTION.
    df_sentence_avg_w : TYPE
        DESCRIPTION.

    """
   
   
    ### 0. Get Weights Based on the Affective Features
    # x = df_iter.drop(columns=['word']).values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df_weights = pd.DataFrame(x_scaled)
    # df_weights.columns = df_iter.drop(columns=['word']).columns
    # df_weights = df_weights.drop(
    #     columns=['Valence_SD', 'Arousal_SD', 'Concreteness_SD',
    #              'Happiness_SD', 'Anger_SD', 'Sadness_SD', 'Fear_SD',
    #              'Disgust_SD', 'Imageability_SD', 'Context_Availability_SD'],
    #     errors="ignore"
    #     )
    df_weights_iter = pd.DataFrame(
        {'word_stem': words_stem,
         'word_lem': words_lem
            }
        )
    df_weights_iter = df_weights_iter.merge(
        df_weights, how="left", left_on=['word_stem'], right_on=['word_stem']
        )
   
   
    ### 1. Get Word Embeddings
    # Sentence-Transformer Library
    if model_used == "sentence-transformer":
        df_embeddings = pd.concat(
            [
                pd.DataFrame(
                    {'index': list(model.encode(word))}
                    ).T
                for word in words_lem
                ]
            )
        df_embeddings = df_embeddings.reset_index(drop=True)
       
    # SpaCy Library
    elif model_used == 'spacy':
        df_embeddings = pd.concat(
            [
                pd.DataFrame(
                    {'index': list(model(word).tensor[0])}
                    ).T
                for word in words_lem
                ]
            )
        df_embeddings = df_embeddings.reset_index(drop=True)
       
    # HuggingFace
    elif model_used == "hugging-face":
        encoded_input = tokenizer(
            words_lem,
            padding=True,
            truncation=False,
            return_tensors='pt')
        model_output = model(**encoded_input)
       
        df_embeddings = pd.concat(
            [
                pd.DataFrame(
                    {'index': list(word[0].detach().numpy())}
                    ).T
                for word in model_output[0]
                ]
            )
        df_embeddings = df_embeddings.reset_index(drop=True)
       
    # Other
    else:
        raise ValueError()
       
   
    ### Max Pooling
    df_sentence_max = pd.DataFrame(df_embeddings.max(axis=0)).copy().T
   
    ### Span
    df_sentence_min = pd.DataFrame(df_embeddings.min(axis=0)).copy().T
    df_sentence_span = df_sentence_max - df_sentence_min
   
    ### Median
    df_sentence_median = pd.DataFrame(df_embeddings.median(axis=0)).copy().T
   
    ### Weighted Avg Based on Affective Info
    """
    We choose the greatest normalized affective value for that word,
    we use that as the weight, and we compute the average weighted mean
    based on that weight.
    """
    # cols_weight = list(df_weights_iter.drop(columns=['word_stem']))
    list_weights = list(
        df_weights_iter.
        fillna(0).
        drop(columns=['word_stem']).
        max(axis=1)
        )
   
    df_sentence_avg_w = pd.DataFrame(
        np.average(
            df_embeddings.to_numpy(),
            axis = 0,
            weights=list_weights
            )
        ).T
   
    return df_sentence_max, df_sentence_span, df_sentence_median, df_sentence_avg_w
   

##### Load affective lexicons
file_to_read = open(f"{PATH_AFF_LEXICON}/list_affective_lexicons.p", "rb")
l_new = pickle.load(file_to_read)
file_to_read.close()

[df_raw_1, df_raw_3, df_raw_4, df_raw_5, df_raw_6, df_all] = l_new


df_all['word_lem'] = df_all['Word']
df_all['word_stem'] = df_all.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
# Remove accents
# df_all['words_stem_uni'] = df_all.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_1['word_lem'] = df_raw_1['Word']
df_raw_1['word_stem'] = df_raw_1.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
# df_raw_1['words_stem_uni'] = df_raw_1.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_3['word_lem'] = df_raw_3['Word']
df_raw_3['word_stem'] = df_raw_3.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
# df_raw_3['words_stem_uni'] = df_raw_3.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_4['word_lem'] = df_raw_4['Word']
df_raw_4['word_stem'] = df_raw_4.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
# df_raw_4['words_stem_uni'] = df_raw_4.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_5['word_lem'] = df_raw_5['Word']
df_raw_5['word_stem'] = df_raw_5.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
# df_raw_5['words_stem_uni'] = df_raw_5.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

df_raw_6['word_lem'] = df_raw_6['Word']
df_raw_6['word_stem'] = df_raw_6.apply(lambda x: stemmer.stem(x['Word']), axis = 1)
# df_raw_6['words_stem_uni'] = df_raw_6.apply(lambda x: unidecode.unidecode(x['word_stem']), axis = 1)

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
df_all.to_csv("datasets/df_lexicon_mod_v4.csv", index=False)

# words = []
# words_lem = []
# words_stem = []
# for word in list(df_all['Word'].values):
#     words_iter, words_stem_iter, words_lem_iter = text_preprocessing(word)
#     words.append(words_iter)
#     words_lem.append(words_lem_iter)
#     words_stem.append(words_stem_iter)

# words_lem = [x[0] for x in words_lem]
# words_stem = [x[0] for x in words_stem]

# # Update lem with MD corpus
# df_all['word_lem'] = words_lem
# df_all['word_stem'] = words_stem

# # Manual Changes
# df_all = df_all.replace("triguir", "trigo")
# df_all = df_all.replace("trigu", "trig")
# df_all = df_all.replace("tigrir", "tigre")


df_all_stem = (
    df_all.
    drop(columns=['Word', 'word_lem']).
    groupby(by=['word_stem']).
    median().
    reset_index()
    )

# Get Weights (for affective levels)
list_drop = [
    'Valence_SD', 'Arousal_SD', 'Concreteness_SD',
    'Happiness_SD', 'Anger_SD', 'Sadness_SD', 'Fear_SD',
    'Disgust_SD', 'Imageability_SD', 'Context_Availability_SD'
    ]

x = df_all_stem.drop(
    columns=list_drop + ['word_stem']).values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
list_cols = list(df_all_stem.drop(
    columns=list_drop + ['word_stem']).columns)
df_weights = df_all_stem.drop(columns=list_drop)
df_weights[list_cols] = x_scaled

##### Load Semantic Models
model_spacy = spacy.load("es_core_news_md")
model1 = SentenceTransformer('quora-distilbert-multilingual')
model2 = SentenceTransformer('stsb-xlm-r-multilingual')
model3 = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model4 = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
model5 = SentenceTransformer('distiluse-base-multilingual-cased-v1') # buenos resultados

model_hugg1 = BertModel.from_pretrained(
    "bert-base-multilingual-cased",
    output_hidden_states = True, # Whether the model returns all hidden-states.
     )
tokenizer1 = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_hugg2 = BertModel.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased",
    output_hidden_states = True, # Whether the model returns all hidden-states.
     )
tokenizer2 = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model_hugg3 = RobertaModel.from_pretrained(
    'roberta-base',
    output_hidden_states = True, # Whether the model returns all hidden-states.
     )
tokenizer3 = RobertaTokenizer.from_pretrained('roberta-base')

## Load Sonnets SXX
df_sonnets_xx = pd.read_excel(
    f"{PATH_POEMS_SXX}/NewCorpusSXX.xlsx"
    )
df_sonnets_xx = df_sonnets_xx.dropna(subset=['text'])

# Use uppercases for proper names
keep_upper=True
if keep_upper:
    path_add_name = "_upper"
   
# Load manual modifications
df_modifications = pd.read_excel("datasets/fixed_words.xlsx").dropna(subset=['fixed_word'])

##### Load list of poems
dct_sonnets = {}
list_docs =  get_files(PATH_POEMS, "xml")
list_idx = range(len(list_docs) + len(df_sonnets_xx))
list_idx = list(list_idx)
# num_doc = 0
# num_sonnet = 0

# i = 0
for index_iter in list_idx:
    # num_doc += 1
    i = index_iter
    print("{0}".format("*"*40))
    print("Iteration {0}/{1}".format(index_iter, len(list_idx)))
       
    # ========================================================================
    # 0. Load Sonnets & Initial Processing
    # ========================================================================
   
    if index_iter < len(list_docs):
        file_path = list_docs[index_iter]
        # Load file
        doc = file_presistance(file_path, "xml", None, "load")
        docs = []
        docs, dct_data = doc2text(doc, docs, "sonnet")
       
        if len(dct_data)>1:
            continue
       
        # Stem/Lem Words
        words, words_stem, words_lem = text_preprocessing(
            dct_data[0]['text'],
            lemma_sentence=True,
            keep_upper=keep_upper,
            df_modifications=df_modifications
            )
       
        # Append
        dct_sonnets[i] = {
            'index': i,
            'author': dct_data[0]['author'],
            'id_doc': dct_data[0]['id_doc'],
            'title': dct_data[0]['title'],
            'year': dct_data[0]['year'],
            'text': dct_data[0]['text'],
            'words': words,
            'words_stem': words_stem,
            'words_lem': words_lem
            }
    else:
        index_iter += 1
        i = index_iter
        dct_data = df_sonnets_xx[df_sonnets_xx['index']==index_iter]
       
        list_stanzas = dct_data['text'].values[0].lstrip().split("\n\n")
       
        if len(list_stanzas)<4:
            list_stanzas = dct_data['text'].values[0].lstrip().split("\n \n")
       
        # Stem/Lem Words
        words, words_stem, words_lem = text_preprocessing(
            dct_data['text'].values[0],
            lemma_sentence=True,
            keep_upper=keep_upper,
            df_modifications=df_modifications
            )
       
        # Append
        dct_sonnets[i] = {
            'index': i,
            'author': dct_data['author'].values[0],
            'id_doc': "",
            'title': dct_data['title'].values[0],
            'year': dct_data['year'].values[0],
            'text': dct_data['text'].values[0],
            'words': words,
            'words_stem': words_stem,
            'words_lem': words_lem
            }
       
        # Add stanzas
        dct_data = {
            0: {'dct_data': dct_data,
                'dct_stanzas': {
                    0: {'text': list_stanzas[0]},
                    1: {'text': list_stanzas[1]},
                    2: {'text': list_stanzas[2]},
                    3: {'text': list_stanzas[3]},
                    }
                }
            }
       
    # # ========================================================================
    # # 1. Affective Features
    # # ========================================================================
    # # Get affective values for this sonnet (order is kept)
    # df_iter = (
    #     pd.DataFrame({"word_stem": words_stem})
    #     .merge(df_all_stem, how="left")
    #     .drop_duplicates(subset=["word_stem"])
    #     .rename(columns={"word_stem": "word"})
    # )
   
    # # Get Affective Features
    # df_aff = affectiveFeaturesText(i, df_iter)
   
    # # Append
    # dct_sonnets[i]['aff_features'] = df_aff
   
    # ========================================================================
    # 1. Affective Features
    # ========================================================================
    sonnet = dct_sonnets[i]
    text_lower = sonnet['text'].lower()
    _, words_stem_lower , words_lem_lower = text_preprocessing(
        text_lower,
        lemma_sentence=True,
        keep_upper=False,
        df_modifications=df_modifications
        )
   
    # Get affective values for this sonnet (order is kept)
    df_iter = (
        pd.DataFrame({"word_stem": words_stem_lower})
        .merge(df_all_stem, how="left")
        .drop_duplicates(subset=["word_stem"])
        .rename(columns={"word_stem": "word"})
    )
   
    # Get Affective Features
    df_aff = affectiveFeaturesText(i, df_iter)
   
    # Append
    dct_sonnets[i]['aff_features'] = df_aff
    dct_sonnets[i]['words_stem_lower'] = words_stem_lower
    dct_sonnets[i]['words_lem_lower'] = words_lem_lower
   
    # ========================================================================
    # 2. Semantic Features
    # ========================================================================
    str_words_lem = ""
    for word in words_lem:
        str_words_lem = str_words_lem + word + ' '
   
    ### Sentence Models
    enc_text_model1 = model1.encode(str_words_lem)
    enc_text_model2 = model2.encode(str_words_lem)
    enc_text_model3 = model3.encode(str_words_lem)
    enc_text_model4 = model4.encode(str_words_lem)
    enc_text_model5 = model5.encode(str_words_lem)
   
    # Append
    dct_sonnets[i]['enc_text_model1'] = enc_text_model1
    dct_sonnets[i]['enc_text_model2'] = enc_text_model2
    dct_sonnets[i]['enc_text_model3'] = enc_text_model3
    dct_sonnets[i]['enc_text_model4'] = enc_text_model4
    dct_sonnets[i]['enc_text_model5'] = enc_text_model5
   
    ### Sentence Models from Words
    ## Model from BERT Cased
    df_sentence_max, df_sentence_span, df_sentence_median, df_sentence_avg_w = (
            customSentenceFunctions(
                words_stem,
                words_lem,
                model_hugg1,
                df_weights,
                model_used = "hugging-face",
                tokenizer = tokenizer1
        )
        )
    dct_sonnets[i]['enc_text_model_hg_bert_max'] = df_sentence_max
    dct_sonnets[i]['enc_text_model_hg_bert_span'] = df_sentence_span
    dct_sonnets[i]['enc_text_model_hg_bert_median'] = df_sentence_median
    dct_sonnets[i]['enc_text_model_hg_bert_avg_w'] = df_sentence_avg_w
   
    ## Model from BERT Cased Spanish
    df_sentence_max, df_sentence_span, df_sentence_median, df_sentence_avg_w = (
            customSentenceFunctions(
                words_stem,
                words_lem,
                model_hugg2,
                df_weights,
                model_used = "hugging-face",
                tokenizer = tokenizer2
        )
        )
    dct_sonnets[i]['enc_text_model_hg_bert_sp_max'] = df_sentence_max
    dct_sonnets[i]['enc_text_model_hg_bert_sp_span'] = df_sentence_span
    dct_sonnets[i]['enc_text_model_hg_bert_sp_median'] = df_sentence_median
    dct_sonnets[i]['enc_text_model_hg_bert_sp_avg_w'] = df_sentence_avg_w
   
    ## Model from ROBERTA
    df_sentence_max, df_sentence_span, df_sentence_median, df_sentence_avg_w = (
            customSentenceFunctions(
                words_stem,
                words_lem,
                model_hugg3,
                df_weights,
                model_used = "hugging-face",
                tokenizer = tokenizer3
        )
        )
    dct_sonnets[i]['enc_text_model_hg_ro_max'] = df_sentence_max
    dct_sonnets[i]['enc_text_model_hg_ro_span'] = df_sentence_span
    dct_sonnets[i]['enc_text_model_hg_ro_median'] = df_sentence_median
    dct_sonnets[i]['enc_text_model_hg_ro_avg_w'] = df_sentence_avg_w
   
   
    ## Model from BERT
    ## model2 | stsb-xlm-r-multilingual
    # Model from BERT
    # customSentenceFunctions(
    #    words_stem, words_lem, model, df_weights, model_used = "sentence-transformer"
    #    )
   
    ## Model from XLM-ROBERTA
    ## model 3 | paraphrase-xlm-r-multilingual-v1
   
    # ========================================================================
    # 3. Per Stanzas
    # ========================================================================
    for keys, stanza in dct_data[0]['dct_stanzas'].items():
       
        keys = f"stanza_{keys}"
        dct_sonnets[i][keys] = {}
       
        if len(stanza)>1:
            continue
       
        ### Processing
        # Stem/Lem Words
        words, words_stem, words_lem = text_preprocessing(
            stanza['text'],
            lemma_sentence=True,
            keep_upper=keep_upper,
            df_modifications=df_modifications
            )
       
        dct_sonnets[i][keys]['text'] = stanza['text']
        dct_sonnets[i][keys]['words'] = words
        dct_sonnets[i][keys]['words_stem'] = words_stem
        dct_sonnets[i][keys]['words_lem'] = words_lem
       
        # ### Affective Features
        # # Get affective values for this stanza (order is kept)
        # df_iter = (
        #     pd.DataFrame({"word_stem": words_stem})
        #     .merge(df_all_stem, how="left")
        #     .drop_duplicates(subset=["word_stem"])
        #     .rename(columns={"word_stem": "word"})
        # )
        # # Get Affective Features
        # df_aff = affectiveFeaturesText(i, df_iter)
       
        # # Append
        # dct_sonnets[i][keys]['aff_features'] = df_aff
       
        ### Affective Features
        text_lower = sonnet[keys]['text'].lower()
        _, words_stem_lower , words_lem_lower = text_preprocessing(
            text_lower,
            lemma_sentence=True,
            keep_upper=False,
            df_modifications=df_modifications
            )
       
        # Get affective values for this stanza (order is kept)
        df_iter = (
            pd.DataFrame({"word_stem": words_stem_lower})
            .merge(df_all_stem, how="left")
            .drop_duplicates(subset=["word_stem"])
            .rename(columns={"word_stem": "word"})
        )
       
        # Get Affective Features
        df_aff = affectiveFeaturesText(i, df_iter)
       
        # Append
        dct_sonnets[i][keys]['aff_features'] = df_aff
        dct_sonnets[i][keys]['words_stem_lower'] = words_stem_lower
        dct_sonnets[i][keys]['words_lem_lower'] = words_lem_lower
       
        ### Semantic Features
        str_words_lem = ""
        for word in words_lem:
            str_words_lem = str_words_lem + word + ' '
       
        # Sentence Models
        enc_text_model1 = model1.encode(str_words_lem)
        enc_text_model2 = model2.encode(str_words_lem)
        enc_text_model3 = model3.encode(str_words_lem)
        enc_text_model4 = model4.encode(str_words_lem)
        enc_text_model5 = model5.encode(str_words_lem)
       
        # Append
        dct_sonnets[i][keys]['enc_text_model1'] = enc_text_model1
        dct_sonnets[i][keys]['enc_text_model2'] = enc_text_model2
        dct_sonnets[i][keys]['enc_text_model3'] = enc_text_model3
        dct_sonnets[i][keys]['enc_text_model4'] = enc_text_model4
        dct_sonnets[i][keys]['enc_text_model5'] = enc_text_model5
       
        ### Sentence Models from Words
       
    i += 1
    print("{0}".format("*"*40))

### Remove multiple-part sonnets
dct_sonnets = {
    x:y for x,y in dct_sonnets.items() if 'stanza_4' not in list(y.keys())
    }

### Save Results
with open(f"{PATH_RESULTS}/dct_sonnets_input_mod_v5{path_add_name}2", 'wb') as handle:
    pickle.dump(dct_sonnets, handle)


# file_to_store = open(f"{PATH_RESULTS}/dct_sonnets_input_mod", "wb")
# pickle.dump(dct_sonnets, file_to_store)
# file_to_read.close()