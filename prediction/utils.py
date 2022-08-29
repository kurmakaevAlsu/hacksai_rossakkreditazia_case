import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')

import string
import re
import transformers
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification

def preprocess_text(x):
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = x.translate(str.maketrans('', '', string.digits))
    x = ''.join([w for w in x if not re.match(r'[A-Z]+', w, re.I)])
    return x.strip().lower()


def get_prediction(text, tokenizer, model, label_df):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=100, return_tensors='pt').to('cpu')
    outputs = model(**inputs)
    probs = outputs[0].softmax(1).cpu().detach().numpy()[0]
    pred_label = np.where(probs == probs.max())[0][0]
    pred_label = label_df[label_df.id == pred_label].name.iloc[0]
    return pred_label, probs.max()


def bert_inference(data, tokenizer, model, label_df):
    texts = list(pd.Series(data).apply(lambda x: preprocess_text(x)).values)
    test_preds = []
    for text in tqdm(texts):
        pred_label, pred_prob = get_prediction(text, tokenizer, model, label_df)
        pred_prob = round(pred_prob, 3)
        test_preds.append([pred_label, pred_prob])
    return f'{test_preds[0][0]}'


def get_prediction_final(df):
    path_to_pretrained_model = '../model6'
    model = BertForSequenceClassification.from_pretrained(path_to_pretrained_model)
    tokenizer = BertTokenizerFast.from_pretrained('sberbank-ai/ruBert-base')
    label_df = pd.read_csv('../label_df.csv')

    df = bert_inference(df, tokenizer, model, label_df)
    return df