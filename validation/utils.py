from catboost import CatBoostClassifier, Pool
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')

from bert_utils import *

def preproccess(df, text_column, category_column):

    def lower_case(x):
        return x.lower()

    def remove_symbols(x):
        x = re.sub('\\-\\s\r\n\\s{1,}|\\-\\s\r\n|\r\n', '', x)
        x = re.sub(
            '[.,:;%©?*,!@#$%^&()\\d]|[+=]|[[]|[]]|[/]|"|\\s{2,}|-',
            ' ',
            x)
        return x

    def remove_stopwords(x):
        STOPWORDS = set(stopwords.words('russian'))
        return " ".join([word for word in str(
            x).split() if word not in STOPWORDS])

    df['preprocessed_text'] = df[text_column].apply(lambda x: lower_case(x))
    df['preprocessed_text'] = df['preprocessed_text'].apply(
        lambda x: remove_symbols(x))
    df['preprocessed_text'] = df['preprocessed_text'].apply(
        lambda x: remove_stopwords(x))

    df['preprocessed_code'] = df[category_column].apply(
        lambda x: lower_case(x))
    df['preprocessed_code'] = df['preprocessed_code'].apply(
        lambda x: remove_symbols(x))
    df['preprocessed_code'] = df['preprocessed_code'].apply(
        lambda x: remove_stopwords(x))

    return df


def get_catboost_predictions(df, model):
    df['Соответсвует_категории'] = model.predict_proba(
        df[['preprocessed_code', 'preprocessed_text']])[:, 1]
    df = df.drop(['preprocessed_code', 'preprocessed_text'], axis=1)
    return df



def get_prediction_final(df):

    model_cb = CatBoostClassifier()
    model_cb.load_model('catboost_final_model.cbm')
    
    text_column = 'Наименование'
    category_column = 'Подкатегория_текст'
    
    path_to_pretrained_model = '../model6'
    model = BertForSequenceClassification.from_pretrained(path_to_pretrained_model)
    tokenizer = BertTokenizerFast.from_pretrained('sberbank-ai/ruBert-base')
    label_df = pd.read_csv('../label_df.csv')
    
    df = preproccess(df, text_column, category_column)
    df = get_catboost_predictions(df, model_cb)
    df = bert_inference(df, text_column, tokenizer, model, label_df)
    return df
