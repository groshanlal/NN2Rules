from data_prep import *
from sklearn.utils import shuffle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/"
download_to = 'data/'
files = ['breast-cancer.data']
column_names = [
'label',
'age',
'menopause',
'tumor-size',
'inv-nodes',
'node-caps',
'deg-malig',
'breast',
'breast-quad',
'irradiat',
]

download(url, download_to, files)

df = clean_missing(download_to + 'breast-cancer.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['no-recurrence-events', 'recurrence-events'], ['0', '1'])

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


