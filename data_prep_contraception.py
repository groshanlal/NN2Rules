from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/"
download_to = 'data/'
files = ['cmc.data']
column_names = [
'age',
'wife-education',
'husband-education',
'num-children',
'wife-religion',
'wife-working',
'husband-occupation',
'standard-of-living',
'media-exposure', 
'contraceptive',
]

download(url, download_to, files)

df = clean_missing(download_to + 'cmc.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'contraceptive', [1, 2, 3], [0, 1, 1])
df = discretize_continuous_columns(df, ['age', 'num-children'])


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


