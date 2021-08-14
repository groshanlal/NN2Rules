import bs4
import requests
import os
import pandas as pd
import numpy as np

def download(url, download_to):
	if(not os.path.isdir(download_to)):
		os.mkdir(download_to)
	r = requests.get(url)
	data = bs4.BeautifulSoup(r.text, "html.parser")
	files = ['adult.data', 'adult.test']
	for f in files:
	    r = requests.get(url + f)
	    with open(download_to + f, 'w') as f_handle:
		    f_handle.write(r.text)
	
	# For test data, remove the first row and remove the extra full stop at the end of each row.
	with open(download_to + 'adult.test', 'r') as fin:
		data = fin.read().splitlines(True)
	data = [row_text[:-2] + '\n' for row_text in data]
	with open(download_to + 'adult.test', 'w') as fout:
		fout.writelines(data[1:])
    

def importer(filename, column_names):
	df = pd.read_csv(filename, sep = ", ", engine = "python")
	df.columns = column_names
	
	# Replace missing value with most frequent
	for col in df.columns:
		df[col] = df[col].replace("?", np.NaN)
	df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
	return df

def create_bins(x, low, high):
	if(x < low):
		return 'low'
	elif(x > high):
		return 'high'
	else:
		return 'mid'

def preprocess(df):
	marital_status = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
						'Married-spouse-absent', 'Never-married','Separated','Widowed']
	marital_status_transformed = ['divorced','married','married',
						'married', 'not married','not married','not married']
	df.replace(marital_status, marital_status_transformed, inplace = True)
	assert(len(df['marital-status'].value_counts()) == len(set(marital_status_transformed)))


	education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 
					'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', 
					'7th-8th', '12th', 'Masters', '1st-4th', '10th', 
					'Doctorate', '5th-6th', 'Preschool']
	education_transformed = ['Bachelors', 'Some-college', 'School', 'HS-grad', 
					'Prof-school', 'Voc', 'Voc', 'School', 
					'School', 'School', 'Masters', 'School', 'School', 
					'Doctorate', 'School', 'School']
	df.replace(education, education_transformed, inplace = True)
	assert(len(df['education'].value_counts()) == len(set(education_transformed)))

	country = list(set(df['native-country']))
	country_transformed = [c if c == 'United-States' else 'Other' for c in country]
	assert(len(set(country_transformed)) == 2)
	df.replace(country, country_transformed, inplace = True)
	assert(len(df['native-country'].value_counts()) == len(set(country_transformed)))

	continuous_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
	for key in continuous_cols:	
		print(key)
		values = df[key]
		values_min   = min(values)
		values_max   = max(values)
		# Check skew
		values_min_count   = 0
		values_max_count   = 0
		for v in values:
			if(v == values_min):
				values_min_count = values_min_count + 1
			if(v == values_max):
				values_max_count = values_max_count + 1
		if(values_min_count > len(values)/3):
			print(str(key) + " is skewed")
			values_modified = values[values > values_min]
			values_low   = np.percentile(values_modified, 33)
			values_high  = np.percentile(values_modified, 67)
		elif(values_max_count > len(values)/3):
			print(str(key) + " is skewed")
			values_modified = values[values < values_max]
			values_low   = np.percentile(values_modified, 33)
			values_high  = np.percentile(values_modified, 67)
		else:
			values_low   = 0.67*values_min + 0.33*values_max
			values_high  = 0.33*values_min + 0.67*values_max
		values_min   = min(values)
		values_max   = max(values)
		print("buckets: ", values_min, ",", values_low, ",", values_high, ",", values_max)
		df[key] = values.apply(lambda x: create_bins(x, values_low, values_high))
	
	
	df = df.drop('fnlwgt',1)
	df = df.drop('education-num',1)
	
	#df = df.drop('education',1)
	df = df.drop('workclass',1)   # dont drop
	#df = df.drop('occupation',1)
	#df = df.drop('relationship',1)
	df = df.drop('race',1)        # dont drop
	#df = df.drop('marital-status',1)
	#df = df.drop('hours-per-week',1)
	
	return df

def tensor_transform(df):
	df = pd.get_dummies(data=df, columns=df.columns)
	df = df.drop('income_<=50K', 1)
	return df

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
download_to = 'data/'

column_names = [
	'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
	'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
	'hours-per-week', 'native-country', 'income'
]

download(url, 'data/')


df_train = importer(download_to + 'adult.data', column_names)
df_train = preprocess(df_train)
df_train.to_csv('data/train_raw.csv', index = False)
df_train = tensor_transform(df_train)


df_test = importer(download_to + 'adult.test', column_names)
df_test = preprocess(df_test) 
df_test.to_csv('data/test_raw.csv', index = False)
df_test = tensor_transform(df_test)


assert(len(df_train.columns) == len(df_test.columns))
for i in range(len(df_train.columns)):
	assert(df_train.columns[i] == df_test.columns[i])


df_train.to_csv('data/train.csv', index = False)
df_test.to_csv('data/test.csv', index = False)



