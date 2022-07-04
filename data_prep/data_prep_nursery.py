from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/"
download_to = 'data/'
files = ['nursery.data']
column_names = [
'parents',
'has_nurs',
'form',
'children',
'housing',
'finance',
'social',
'health',
'label',
]

download(url, download_to, files)

df = clean_missing(download_to + 'nursery.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'], ['0', '0', '0', '1', '1'])

num_rows_train = int(0.8*len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/train_raw.csv', index = False)
df_test.to_csv('data/test_raw.csv', index = False)

df = tensor_transform(df)
df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/train.csv', index = False)
df_test.to_csv('data/test.csv', index = False)


