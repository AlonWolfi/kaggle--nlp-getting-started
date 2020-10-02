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

    features['count_urls'] = text.apply(lambda s: len(re.findall('https?://\S+|www\.\S+', s)))

    features['count_large_words'] = text.apply(lambda s: len(re.findall('(^|\s)([A-Z]\S+)', s)))
    features['count_small_words'] = text.apply(lambda s: len(re.findall('(^|\s)([a-z]\S+)', s)))
    features['count_words'] = text.apply(lambda s: len(re.findall('(^|\s)(\S+)', s)))
    features['count_large_words_frac'] = features['count_large_words'] / (features['count_words'] + 1e-1)

    text_cleaned = text.apply(clean_text)
    vect = TfidfVectorizer()
    text_vect = vect.fit_transform(text_cleaned)
    text_vect_df = pd.DataFrame(
        data=text_vect.toarray(),
        columns=list(vect.vocabulary_.keys()),
        index=text_cleaned.index
    )

    features = features.join(text_vect_df)
    features.columns = ['text_' + c for c in features.columns]
    return features
