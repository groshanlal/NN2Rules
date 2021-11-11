from data_prep import *
from sklearn.utils import shuffle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/"
download_to = 'data/'
files = ['agaricus-lepiota.data']
column_names = [
'label',
'cap-shape',
'cap-surface',
'cap-color',
'bruises',
'odor',
'gill-attachment',
'gill-spacing',
'gill-size',
'gill-color',
'stalk-shape',
'stalk-root',
'stalk-surface-above-ring',
'stalk-surface-below-ring',
'stalk-color-above-ring',
'stalk-color-below-ring',
'veil-type',
'veil-color',
'ring-number',
'ring-type',
'spore-print-color',
'population',
'habitat',
]

download(url, download_to, files)

df = clean_missing(download_to + 'agaricus-lepiota.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['p', 'e'], ["0", "1"])

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


