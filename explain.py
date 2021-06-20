import tensorflow as tf
import pandas as pd
import numpy as np
import time

class ModelWeights:
	def __init__(self):
		self.__load_weights()
		self.__load_features()
		self.__reshape_feature_terms()
		return

	def __load_weights(self):
		model = tf.keras.models.load_model('income_model')
		self.weight_layer_1 = model.layers[0].get_weights()[0]
		self.bias_layer_1   = model.layers[0].get_weights()[1]
		self.weight_layer_2 = model.layers[1].get_weights()[0]
		self.bias_layer_2   = model.layers[1].get_weights()[1]
		return

	def __load_features(self):
		self.feature_terms = list(pd.read_csv("data/train.csv").columns[:-1])
		return

	def __reshape_feature_terms(self):
		feature_terms_root = [x.split('_')[0] for x in self.feature_terms]

		feature_names = []
		self.reshaped_feature_terms = []

		for i in range(len(feature_terms_root)):
			name = feature_terms_root[i]
			term = self.feature_terms[i]
			if(name not in feature_names):
				feature_names.append(name)
				self.reshaped_feature_terms.append([])
			self.reshaped_feature_terms[-1].append(term)
		return
	
	
class InputNeuron:
	def __init__(self, weights, bias, reshaped_feature_terms, feature_order = None):
		self.weights = weights[:]
		self.bias = bias

		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]	
		self.feature_order = feature_order	

		self.__reshape_term_weights()
		self.__shift_term_weights()
		self.__sort_term_weights()
		if(feature_order is not None):
			self.__order_features(feature_order)

	def __reshape_term_weights(self):
		self.reshaped_weights = []
		cum_sum = 0
		for ft in self.reshaped_feature_terms:
			num_terms = len(ft)
			fw = list(self.weights[cum_sum : cum_sum + num_terms])
			self.reshaped_weights.append(fw)
			cum_sum = cum_sum + num_terms

	def __shift_term_weights(self):
		self.reshaped_bias = self.bias
		for i in range(len(self.reshaped_feature_terms)):
			fw = self.reshaped_weights[i]
			fw_min = min(fw)
			fw = [fw_terms - fw_min for fw_terms in fw]
			self.reshaped_weights[i] = fw
			self.reshaped_bias = self.reshaped_bias + fw_min

	def __sort_term_weights(self):
		for i in range(len(self.reshaped_feature_terms)):
			fw = np.array(self.reshaped_weights[i])
			terms = np.array(self.reshaped_feature_terms[i])

			order = np.argsort(-fw)

			self.reshaped_weights[i] = list(fw[order])
			self.reshaped_feature_terms[i] = list(terms[order])

	def __order_features(self, order):
		self.reshaped_weights = [self.reshaped_weights[i] for i in order]
		self.reshaped_feature_terms = [self.reshaped_feature_terms[i] for i in order]

class ForestBuilder:
	def __init__(self, weights, bias, reshaped_weights, reshaped_bias, reshaped_feature_terms):
		self.weights = weights[:]
		self.bias = bias

		self.reshaped_weights = [w[:] for w in reshaped_weights]
		self.reshaped_bias = reshaped_bias
		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]

		return 
		
	def score(self, terms):
		value = 0 
		if(len(terms) > 0):
			for i in range(len(terms)):
				j = self.reshaped_feature_terms[i].index(terms[i])
				value = value + self.reshaped_weights[i][j]
		return value

	def is_always_active(self):
		weights_min = [w[ -1] for w in self.reshaped_weights]
		if(sum(weights_min) + self.reshaped_bias >= 0):
			return True
		return False

	def is_always_inactive(self):
		weights_max = [w[ 0] for w in self.reshaped_weights]		
		if(sum(weights_max) + self.reshaped_bias < 0):
			return True
		return False

	def get_forest(self, sort = True, prior = None):
		forest = self.build_forest(prior = prior)
		if(sort is True):
			forest_str = [tree.to_string() for tree in forest]
			forest_str = np.array(forest_str)
			order = np.argsort(forest_str)
			
			sorted_forest = [forest[i] for i in order]
			return sorted_forest
		else:
			return forest

	def build_forest(self, prior = None):
		if(prior is not None):
			num_terms = len(prior.terms)
			sub_neuron_forest_builder = ForestBuilder(self.weights[:], self.bias,
											self.reshaped_weights[num_terms:], 
											self.reshaped_bias + prior.value, 
											self.reshaped_feature_terms[num_terms:])
			forest = sub_neuron_forest_builder.build_forest()

			for i in range(len(forest)):
				forest[i].add_prefix(prior) 

			return forest

		if(self.is_always_inactive()):
			return []
		if(self.is_always_active()):
			return [TreeBuilder([], 0, 0, self.weights, self.bias)]
		
		root_choices = self.reshaped_feature_terms[0]
		root_weights = self.reshaped_weights[0]
		
		preconditions = [ TreeBuilder([root_choices[0]], root_weights[0], root_weights[0], self.weights, self.bias) ]
		forest = []
		
		N = len(root_choices)
		for i in range(N):
			root_term = root_choices[i]
			root_value = root_weights[i]
		
			forest_at_root = []
			for j in range(len(preconditions)):
				prior = preconditions[j].copy()
				prior.update_root(root_term, root_value)
				
				forest_at_root_with_prior = self.build_forest(prior = prior)	
				forest_at_root.extend(forest_at_root_with_prior)
			
			forest.extend(forest_at_root)		
			preconditions = forest_at_root		
		
		return forest

class TreeBuilder:
	def __init__(self, terms, value, root_value, weights, bias):
		self.terms = terms[:]
		self.value = value
		self.root_value = root_value
		self.weights = weights[:]
		self.bias = bias

	def add_prefix(self, prefix_tree):
		self.terms = prefix_tree.terms + self.terms
		self.value = prefix_tree.value + self.value 
		self.root_value = prefix_tree.root_value

	def update_root(self, new_root_term, new_root_value):
		self.terms[0] = new_root_term
		self.value = self.value - self.root_value + new_root_value 
		self.root_value = new_root_value

	def copy(self):
		return TreeBuilder(self.terms, self.value, self.root_value, self.weights, self.bias)

	def to_string(self):
		sep = " and "
		tree_str = sep.join(self.terms)
		return tree_str

	def num_terms(self):
		return len(self.terms)
	
	def scale(self, c):
		self.weights = list(np.array(self.weights) * c)
		self.bias = self.bias * c
		self.value = self.value * c
		self.root_value = self.root_value * c
		return
	
	def add(self, c):
		self.bias = self.bias + c
		self.value = self.value + c
		return
	
class Neuron:
	def __init__(self, weights, bias, reshaped_feature_terms, feature_order):
		self.weights = weights		
		self.bias = bias
		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]

		positive_neuron = InputNeuron(self.weights, self.bias, self.reshaped_feature_terms, feature_order)
		positive_forest_builder = ForestBuilder(positive_neuron.weights, positive_neuron.bias,
			positive_neuron.reshaped_weights, positive_neuron.reshaped_bias, positive_neuron.reshaped_feature_terms)
		forest_positive = positive_forest_builder.get_forest()
		self.forest_positive = forest_positive
		print(len(forest_positive))

		negative_neuron = InputNeuron(list(-1*np.array(self.weights)), -1*self.bias, self.reshaped_feature_terms, feature_order)
		negative_forest_builder = ForestBuilder(negative_neuron.weights, negative_neuron.bias,
			negative_neuron.reshaped_weights, negative_neuron.reshaped_bias, negative_neuron.reshaped_feature_terms)
		forest_negative = negative_forest_builder.get_forest()
		for tree in forest_negative:
			tree.scale(0.0)
		self.forest_negative = forest_negative
		print(len(forest_negative))

		forest = []
		i = 0
		j = 0
		while((i < len(forest_positive)) and (j < len(forest_negative))):
			if(forest_positive[i].to_string() < forest_negative[j].to_string()):
				forest.append(forest_positive[i])
				i = i + 1
			else:
				forest.append(forest_negative[j])
				j = j + 1
		while(i < len(forest_positive)):
			forest.append(forest_positive[i])
			i = i + 1
		while(j < len(forest_negative)):
			forest.append(forest_negative[j])
			j = j + 1
		self.forest = forest
		return

income_model = ModelWeights()
_ , num_hidden_neurons =  np.shape(income_model.weight_layer_1)

feature_importance = []
for i in range(num_hidden_neurons):
	neuron = InputNeuron(income_model.weight_layer_1[:, i], 
		income_model.bias_layer_1[i], 
		income_model.reshaped_feature_terms)
	feature_importance.append([])
	for i in range(len(neuron.reshaped_feature_terms)):
		feature_importance[-1].append( (max(neuron.reshaped_weights[i]) - min(neuron.reshaped_weights[i])) / len(neuron.reshaped_feature_terms[i]) )


feature_importance = np.array(feature_importance)
feature_importance = np.mean(feature_importance, axis = 0)
feature_order = list(np.argsort(-feature_importance))

print("Features Names in Descending Order of Importance: ")
print([neuron.reshaped_feature_terms[feature_order[i]] for i in range(len(feature_order))]) 
print([len(neuron.reshaped_feature_terms[feature_order[i]]) for i in range(len(feature_order))]) 
print()
neuron_layer = []

t = time.time()
for i in range(num_hidden_neurons):
	print("Neuron " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(num_hidden_neurons - i))) + " min")
		t = time.time()

	neuron = Neuron(list(income_model.weight_layer_1[:, i]), 
		income_model.bias_layer_1[i], 
		income_model.reshaped_feature_terms,
		feature_order)	

	print("Forest: " + str(len(neuron.forest)))

	neuron_layer.append(neuron)


	print("--------------------")

with open('tree_pos.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = [tree.to_string() for tree in neuron_layer[i].forest_positive]
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")
	
with open('tree_neg.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = [tree.to_string() for tree in neuron_layer[i].forest_negative]
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')
	print("Done")
'''
with open('tree.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = [tree.to_string() for tree in neuron_layer[i].forest]
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")
	
with open('behavior.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		weights_bias = [tree.weights + [tree.bias] for tree in neuron_layer[i].forest]
		for wb in weights_bias:
			f.write(str(wb) + '\n')
		f.write("----------------" + '\n')
	print("Done")
'''

def tree_and(tree_1, weights_1, bias_1, tree_2, weights_2, bias_2, w1, w2):
	if(len(tree_1) > len(tree_2)):
		longer_tree = tree_1
		smaller_tree = tree_2
	else:
		longer_tree = tree_2
		smaller_tree = tree_1
	if(smaller_tree != longer_tree[:len(smaller_tree)]):
		return None, [], 0 

	weights = list(w1 * np.array(weights_1) + w2 * np.array(weights_2))
	bias = w1 * bias_1 + w2 * bias_2


	return longer_tree, weights, bias

def tree_diff(tree_1, tree_2):
	i = 0
	while(tree_1[i] == tree_2[i]):
		i = i + 1
	if(tree_1[i] > tree_2[i]):
		return 1
	else:
		return -1

def forest_and(forest_1, weights_1, bias_1, forest_2, weights_2, bias_2, w1, w2):
	forest_conjunction = []
	weights_conjunction = []
	bias_conjunction = []

	i = 0
	j = 0
	while((i < len(forest_1)) and (j < len(forest_2))):
		tree, weights, bias = tree_and(forest_1[i], weights_1[i], bias_1[i], 
			forest_2[j], weights_2[j], bias_2[j],
			w1, w2)

		if(tree is not None):
			forest_conjunction.append(tree)
			weights_conjunction.append(weights)
			bias_conjunction.append(bias)
			if(len(tree) == len(forest_1[i])):
				i = i + 1
			if(len(tree) == len(forest_2[j])):
				j = j + 1
		else:
			if(tree_diff(forest_1[i], forest_2[j]) < 0):
				i = i + 1
			else:
				j = j + 1

	return forest_conjunction, weights_conjunction, bias_conjunction

forest_conjunction = [tree.terms for tree in neuron_layer[0].forest]
weights_conjunction = [list(np.array(tree.weights) * income_model.weight_layer_2[0, 0]) for tree in neuron_layer[0].forest] 
bias_conjunction = [tree.bias * income_model.weight_layer_2[0, 0] for tree in neuron_layer[0].forest]
print("Conditions")
print(len(forest_conjunction))

for i in range(1, 6):
	forest_conjunction, weights_conjunction, bias_conjunction = forest_and(forest_conjunction, weights_conjunction, bias_conjunction,
		[tree.terms for tree in neuron_layer[i].forest],
		[tree.weights for tree in neuron_layer[i].forest],
		[tree.bias for tree in neuron_layer[i].forest],
		1, income_model.weight_layer_2[i, 0]
		)

	print(len(forest_conjunction))

forest_final = []
for i in range(len(forest_conjunction)):
	bias_conjunction[i] = bias_conjunction[i] + income_model.bias_layer_2[0]
	positive_neuron = InputNeuron(weights_conjunction[i], bias_conjunction[i], income_model.reshaped_feature_terms, feature_order)
	positive_forest_builder = ForestBuilder(positive_neuron.weights, positive_neuron.bias, 
		positive_neuron.reshaped_weights, positive_neuron.reshaped_bias, positive_neuron.reshaped_feature_terms)
		
	value = positive_forest_builder.score(forest_conjunction[i])
	root_value = positive_forest_builder.score(forest_conjunction[i][:1])
		
	prior = TreeBuilder(forest_conjunction[i], value, root_value, weights_conjunction[i], bias_conjunction[i])
	forest_positive = positive_forest_builder.get_forest(prior = prior)
	forest_final.extend(forest_positive)

print("Trees")
print(len(forest_final))


