import tensorflow as tf
import pandas as pd
import numpy as np
import time
from util import ModelWeights, InputNeuron
from util import ForestBuilder, TreeBuilder
from util import Neuron

income_model = ModelWeights()

feature_importance = income_model.get_feature_importance()
feature_importance = np.array(feature_importance)
feature_order = np.argsort(-feature_importance).tolist()

print("Features Names in Descending Order of Importance: ")
print([income_model.reshaped_feature_terms[feature_order[i]] for i in range(len(feature_order))]) 
print([len(income_model.reshaped_feature_terms[feature_order[i]]) for i in range(len(feature_order))]) 
print()

neuron_layer = []

t = time.time()
for i in range(len(income_model.bias_layer_1)):
	print("Neuron " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(len(income_model.bias_layer_1) - i))) + " min")
		t = time.time()

	neuron = Neuron(income_model.weight_layer_1[i], 
		income_model.bias_layer_1[i], 
		income_model.reshaped_feature_terms,
		feature_order)	

	print("Positive Forest: " + str(len(neuron.forest_positive)))
	print("Negative Forest: " + str(len(neuron.forest_negative)))
	print("Forest: " + str(len(neuron.forest)))

	neuron_layer.append(neuron.forest)
	print("--------------------")

with open('tree.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_layer[i]:
			tree_str = tree.to_string()
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

with open('tree_pos.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_layer[i]:
			tree_str = tree.to_string()
			if(sum(tree.weights) + tree.bias > 0):
				f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

with open('tree_neg.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_layer[i]:
			tree_str = tree.to_string()
			if(sum(tree.weights) + tree.bias <= 0):
				f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

class Tree:
	def __init__(self, terms, list_of_weights, list_of_bias):
		self.terms = terms[:] 
		self.list_of_weights = [w[:] for w in list_of_weights] 
		self.list_of_bias = [b for b in list_of_bias]
	
	def diff(self, tree):
		i = 0
		while(self.terms[i] == tree.terms[i]):
			i = i + 1
		if(self.terms[i] > tree.terms[i]):
			return 1
		else:
			return -1

	def logical_and(self, tree):
		if(len(self.terms) > len(tree.terms)):
			longer_tree = self
			smaller_tree = tree
		else:
			longer_tree = tree
			smaller_tree = self
		if(smaller_tree.terms != longer_tree.terms[:len(smaller_tree.terms)]):
			return None
		result_tree = Tree(longer_tree.terms, 
			self.list_of_weights + tree.list_of_weights, 
			self.list_of_bias + tree.list_of_bias)
		return result_tree

class Forest:
	def __init__(self, forest_builder):
		self.list_of_trees = [Tree(tb.terms, [tb.weights], [tb.bias]) for tb in forest_builder]

	def logical_and(self, forest):
		i = 0
		j = 0
		forest_conjunction = []
		while((i < len(self.list_of_trees)) and (j < len(forest.list_of_trees))):
			tree = self.list_of_trees[i].logical_and(forest.list_of_trees[j])
			if(tree is not None):
				forest_conjunction.append(tree)
				if(len(tree.terms) == len(self.list_of_trees[i].terms)):
					i = i + 1
				if(len(tree.terms) == len(forest.list_of_trees[j].terms)):
					j = j + 1
			else:
				if(self.list_of_trees[i].diff(forest.list_of_trees[j]) < 0):
					i = i + 1
				else:
					j = j + 1

		self.list_of_trees = forest_conjunction
		return

	def get_firing(self):
		firing = []
		for i in range(len(self.list_of_trees)):
			firing.append([])
			for j in range(len(self.list_of_trees[i].list_of_weights)):
				weight_sum = sum(self.list_of_trees[i].list_of_weights[j]) + \
									self.list_of_trees[i].list_of_bias[j]
				if(weight_sum > 0):
					weight_sum = 1.0
				firing[-1].append(weight_sum)
		return firing


print("First Layer")
print("Trees")
print(len(neuron_layer))
for i in range(len(neuron_layer)):
	print(len(neuron_layer[i]))
print()

neuron_layer_weights = np.array(income_model.weight_layer_2)
neuron_layer_bias = np.array(income_model.bias_layer_2)
print(neuron_layer_weights.shape)
print(neuron_layer_bias.shape)

conditions = Forest(neuron_layer[0])
for i in range(1, len(neuron_layer)):
	conditions.logical_and(Forest(neuron_layer[i]))

firing = conditions.get_firing()

for i in range(len(conditions.list_of_trees)):
	w = np.array(conditions.list_of_trees[i].list_of_weights)
	b = np.array(conditions.list_of_trees[i].list_of_bias)

	w = np.matmul(neuron_layer_weights, w)
	b = np.matmul(neuron_layer_weights, b.reshape(-1, 1))
	b = b + neuron_layer_bias.reshape(-1, 1)
	b = b.reshape(-1) 

	conditions.list_of_trees[i].list_of_weights = w.tolist()
	conditions.list_of_trees[i].list_of_bias = b.tolist()
	

neuron_layer = []
for j in range(len(neuron_layer_bias)):
	neuron_layer.append([])
	for i in range(len(conditions.list_of_trees)):
		neuron = Neuron(conditions.list_of_trees[i].list_of_weights[j], 
			conditions.list_of_trees[i].list_of_bias[j], 
			income_model.reshaped_feature_terms,
			feature_order,
			prior_terms = conditions.list_of_trees[i].terms,
			relu_activation = False)	
		neuron_layer[-1].extend(neuron.forest_positive)

print("Second Layer")
print("Trees")
print(len(neuron_layer))
for i in range(len(neuron_layer)):
	print(len(neuron_layer[i]))
print()

with open('tree_final.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		for tree in neuron_layer[i]:
			tree_str = tree.to_string()
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")

print("Verifying RuleList ...")

y_pred = pd.read_csv("data/pred.csv", header=None).to_numpy()
y_pred = y_pred.reshape(-1)

x_test = pd.read_csv("data/test.csv")
x_test = x_test.drop("income_<=50K", 1)
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

