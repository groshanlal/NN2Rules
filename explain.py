import tensorflow as tf
import pandas as pd
import numpy as np
import time
from util import ModelWeights, InputNeuron
from util import ForestBuilder, TreeBuilder
from util import Neuron
from util import Forest, Tree

income_model = ModelWeights()

feature_importance = income_model.get_feature_importance()
feature_importance = np.array(feature_importance)
feature_order = np.argsort(-feature_importance).tolist()

print("Features Names in Descending Order of Importance: ")
print([income_model.reshaped_feature_terms[feature_order[i]] for i in range(len(feature_order))]) 
print([len(income_model.reshaped_feature_terms[feature_order[i]]) for i in range(len(feature_order))]) 
print()

neuron_layer = []
neuron_collection = []
if(1 == len(income_model.layer_weights)):
	isLastLayer = True
else:
	isLastLayer = False

t = time.time()
for i in range(len(income_model.layer_bias[0])):
	print("Neuron " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(len(income_model.layer_bias[0]) - i))) + " min")
		t = time.time()

	neuron = Neuron(income_model.layer_weights[0][i], 
		income_model.layer_bias[0][i], 
		income_model.reshaped_feature_terms,
		feature_order)	

	print("Positive Forest: " + str(len(neuron.forest_positive)))
	print("Negative Forest: " + str(len(neuron.forest_negative)))
	print("Forest: " + str(len(neuron.forest)))

	if(isLastLayer):	
		neuron_layer.append(neuron.forest_positive)
	else:
		neuron_layer.append(neuron.forest)
	neuron_collection.append(neuron)
	print("--------------------")

with open('tree.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_collection)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_collection[i].forest:
			tree_str = tree.to_string()
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

with open('tree_pos.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_collection)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_collection[i].forest_positive:
			tree_str = tree.to_string()
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

with open('tree_neg.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_collection)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_collection[i].forest_negative:
			tree_str = tree.to_string()
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

print("Layer 1")
print(len(neuron_layer))
for i in range(len(neuron_layer)):
	print(len(neuron_layer[i]))
print()

for layer_num in range(1, len(income_model.layer_weights)):
	if(layer_num + 1 == len(income_model.layer_weights)):
		isLastLayer = True
	else:
		isLastLayer = False

	neuron_layer_weights = np.array(income_model.layer_weights[layer_num])
	neuron_layer_bias = np.array(income_model.layer_bias[layer_num])

	print("Computing Conditions: Step 1: Merging previous layer trees")
	conditions = Forest(neuron_layer[0])
	for i in range(1, len(neuron_layer)):
		conditions.logical_and(Forest(neuron_layer[i]))

	#firing = conditions.get_firing()

	print("Computing Conditions: Step 2: Extending weights to next layer")
	for i in range(len(conditions.list_of_trees)):
		w = np.array(conditions.list_of_trees[i].list_of_weights)
		b = np.array(conditions.list_of_trees[i].list_of_bias)

		w = np.matmul(neuron_layer_weights, w)
		b = np.matmul(neuron_layer_weights, b.reshape(-1, 1))
		b = b + neuron_layer_bias.reshape(-1, 1)
		b = b.reshape(-1) 

		conditions.list_of_trees[i].list_of_weights = w.tolist()
		conditions.list_of_trees[i].list_of_bias = b.tolist()

	print("Computing Conditions: Step 3: Extending trees to next layer")
	neuron_layer = []
	for j in range(len(neuron_layer_bias)):
		print("Neuron " + str(j))
		neuron_layer.append([])
		for i in range(len(conditions.list_of_trees)):
			neuron = Neuron(conditions.list_of_trees[i].list_of_weights[j], 
				conditions.list_of_trees[i].list_of_bias[j], 
				income_model.reshaped_feature_terms,
				feature_order,
				prior_terms = conditions.list_of_trees[i].terms,
				relu_activation = not isLastLayer)
			if(isLastLayer):	
				neuron_layer[-1].extend(neuron.forest_positive)
			else:
				neuron_layer[-1].extend(neuron.forest)

	
	print("Layer " + str(layer_num + 1))
	print(len(neuron_layer))
	for i in range(len(neuron_layer)):
		print(len(neuron_layer[i]))
	print()


forest_final = [tree.terms for tree in neuron_layer[0]]

with open('tree_final.txt', 'w') as f:
	print("Writing")
	for tree in forest_final:
		tree_str = " and ".join(tree)
		f.write(str(tree_str) + '\n')
	print("Done")

print("Verifying RuleList ...")

y_pred = pd.read_csv("data/test_pred.csv", header=None).to_numpy()
y_pred = y_pred.reshape(-1)

x_test = pd.read_csv("data/test.csv")
x_test = x_test.drop("income_>50K", 1)
x_test = x_test.to_numpy()

x_test_terms = pd.read_csv("data/test_raw.csv")
x_test_terms = x_test_terms.drop("income", 1)

cols = list(x_test_terms.columns)
cols = [cols[i] for i in feature_order]
x_test_terms = x_test_terms[cols]

for c in cols:
	x_test_terms[c] = x_test_terms[c].apply(lambda x: c + "_" + x)

x_test_terms = x_test_terms.to_numpy()
x_test_terms = x_test_terms.tolist()

x_train_terms = pd.read_csv("data/train_raw.csv")
x_train_terms = x_train_terms.drop("income", 1)

cols = list(x_train_terms.columns)
cols = [cols[i] for i in feature_order]
x_train_terms = x_train_terms[cols]

for c in cols:
	x_train_terms[c] = x_train_terms[c].apply(lambda x: c + "_" + x)

x_train_terms = x_train_terms.to_numpy()
x_train_terms = x_train_terms.tolist()

for i in range(len(x_test_terms)):
	j = 0 
	while(neuron_layer[0][j].terms != x_test_terms[i][:len(neuron_layer[0][j].terms)]):
		j = j + 1
		if(j == len(neuron_layer[0])):
			break
	
	if(j == len(neuron_layer[0])) and (y_pred[i] > 0.5):
		print("Wrong!")
	if(j < len(neuron_layer[0])) and (y_pred[i] < 0.5):
		print("Wrong!")
	
	if(j < len(neuron_layer[0])):
		pre_sigmoid = np.dot(x_test[i], neuron_layer[0][j].weights) + neuron_layer[0][j].bias
		post_sigmoid = 1 / (1 + np.e ** (-1 * pre_sigmoid))
		if(np.abs(y_pred[i] - post_sigmoid) > 0.005):
			print(y_pred[i])
			print(post_sigmoid)
			print("Wrong!")

print("Simplifying forest")
assert(len(neuron_layer) == 1)
feature_term_nums = [len(income_model.reshaped_feature_terms[feature_order[i]]) for i in range(len(feature_order))]
conditions = Forest(neuron_layer[0])
forest_final = conditions.simplify_forest_terms(feature_term_nums)
print()

with open('tree_final.txt', 'w') as f:
	print("Writing")
	for tree in forest_final:
		tree_str = " and ".join(tree)
		f.write(str(tree_str) + '\n')
	print("Done")


print("Counting RuleList ...")
counts = [0]*len(forest_final)

for i in range(len(x_train_terms)):
	j = 0 
	while(forest_final[j] != x_train_terms[i][:len(forest_final[j])]):
		j = j + 1
		if(j == len(forest_final)):
			break
	if(j < len(counts)):
		counts[j] = counts[j] + 1
	
counts = np.array(counts)
sum_counts = np.sum(counts)
counts = counts / sum_counts
order = np.argsort(-1 * counts)
counts = [counts[order[i]] for i in range(len(order))]
forest_final = [forest_final[order[i]] for i in range(len(order))]
i = 0 
cum = 0
while(cum < 0.90):
	cum = cum + counts[i]
	i = i + 1

forest_final = forest_final[:i]
counts = counts[:i]
print(len(forest_final))

with open('tree_final_weeded.txt', 'w') as f:
	print("Writing")
	for tree in forest_final:
		tree_str = " and ".join(tree)
		f.write(str(tree_str) + '\n')
	print("Done")

with open('tree_final_counts.txt', 'w') as f:
	print("Writing")
	for ct in counts:
		f.write(str(ct) + '\n')
	print("Done")
