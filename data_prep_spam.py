from data_prep import *
from sklearn.utils import shuffle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/"
download_to = 'data/'
files = ['spambase.data']
column_names = [
"make", "address", "all", "3d", "our", "over", "remove", "internet",
"order", "mail", "receive", "will", "people", "report", "addresses",
"free", "business", "email", "you", "credit", "your", "font",
"000", "money", "hp", "hpl", "george", "650", "lab", "labs",
"telnet", "857", "data", "415", "85", "technology", "1999",
"parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu", "table",
"conference", "punc;", "punc(", "punc[", "punc!", "punc$", "punc#", "length-average", "length-longest",
"length-total", "spam"
]

download(url, download_to, files)

df = clean_missing(download_to + 'spambase.data', column_names, sep = ",")
df = shuffle(df)
df = discretize_continuous_columns(df, df.columns[:-1])
df = rename_label(df, 'spam', ["0", "1"], ["0", "1"])

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


