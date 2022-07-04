import tensorflow as tf
import pandas as pd
import numpy as np
from nn2rules.explainer import ModelExplainer

model_explainer = ModelExplainer(model_path='trained_model', data_path='data/train.csv')
rule_list = model_explainer.explain()

with open('rule_list.txt', 'w') as f:
	print("Writing")
	for rule in rule_list:
		rule_str = " and ".join(rule)
		f.write(str(rule_str) + '\n')
	print("Done")


