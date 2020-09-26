import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def process_text(text):
    features = pd.DataFrame(index=text.index)
    text = text.apply(clean_text)
    vect = TfidfVectorizer()
    text_vect = vect.fit_transform(text)
    text_vect_df = pd.DataFrame(
        data=text_vect.toarray(),
        columns=list(vect.vocabulary_.keys()),
        index=text.index
    )
    features = features.join(text_vect_df)
    features.columns = ['text_' + c for c in features.columns]
    return features
