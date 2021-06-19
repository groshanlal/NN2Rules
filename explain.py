import tensorflow as tf
import pandas as pd
import numpy as np
import time

class ModelWeights:
	def __init__(self):
		self.__load_weights()
		self.__load_features()
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
	
	
class InputNeuron:
	def __init__(self, weights, bias, feature_terms, feature_order = None):
		self.weights = weights
		self.bias = bias
		self.feature_terms = feature_terms	
		self.feature_order = feature_order	

		self.__reshape_feature_terms()
		self.__reshape_term_weights()
		self.__shift_term_weights()
		self.__sort_term_weights()
		if(feature_order is not None):
			self.__order_features(feature_order)

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

	def copy(self):
		return InputNeuron(self.weights[:], self.bias, self.feature_terms[:], self.feature_order)

class ForestBuilder:
	def __init__(self, neuron, reshaped_weights = None, reshaped_bias = None, reshaped_feature_terms = None):
		self.neuron = neuron.copy()
		
		if(reshaped_weights is None):
			reshaped_weights = neuron.reshaped_weights
			reshaped_bias = neuron.reshaped_bias
			reshaped_feature_terms = neuron.reshaped_feature_terms

		self.reshaped_weights = [w[:] for w in reshaped_weights]
		self.reshaped_bias = reshaped_bias
		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]

		return 
		
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
			sub_neuron_forest_builder = ForestBuilder(self.neuron, 
											self.reshaped_weights[num_terms:], 
											self.reshaped_bias + prior.get_value(), 
											self.reshaped_feature_terms[num_terms:])
			forest = sub_neuron_forest_builder.build_forest()

			for i in range(len(forest)):
				forest[i].terms = prior.terms + forest[i].terms 

			return forest

		if(self.is_always_inactive()):
			return []
		if(self.is_always_active()):
			return [TreeBuilder([], self.neuron)]
		
		root_choices = self.reshaped_feature_terms[0]
		
		preconditions = [ TreeBuilder([root_choices[0]], self.neuron) ]
		forest = []
		
		N = len(root_choices)
		for i in range(N):
			root_term = root_choices[i]
		
			forest_at_root = []
			for j in range(len(preconditions)):
				prior = preconditions[j].copy()
				prior.update_root(root_term)
				
				forest_at_root_with_prior = self.build_forest(prior = prior)	
				forest_at_root.extend(forest_at_root_with_prior)
			
			forest.extend(forest_at_root)		
			preconditions = forest_at_root		
		
		return forest

class TreeBuilder:
	def __init__(self, terms, neuron):
		self.terms = terms[:]		
		self.neuron = neuron.copy()

	def scale(self, c):
		weights = self.neuron.weights
		bias = self.neuron.bias

		weights = list(np.array(weights) * c)
		bias = bias * c

		self.neuron = InputNeuron(weights, bias, self.neuron.feature_terms, self.neuron.feature_order)

	def add(self, c):
		weights = self.neuron.weights
		bias = self.neuron.bias

		bias = bias + c

		self.neuron = InputNeuron(weights, bias, self.neuron.feature_terms, self.neuron.feature_order)

	def update_root(self, new_root_term):
		self.terms[0] = new_root_term

	def copy(self):
		return TreeBuilder(self.terms[:], self.neuron.copy())

	def get_value(self):
		value = 0 
		if(len(self.terms) > 0):
			offset = 0
			while(self.terms[0] not in self.neuron.reshaped_feature_terms[offset]):
				offset = offset + 1
			for i in range(len(self.terms)):
				j = self.neuron.reshaped_feature_terms[i + offset].index(self.terms[i])
				value = value + self.neuron.reshaped_weights[i + offset][j]
		return value

	def to_string(self):
		sep = " and "
		tree_str = sep.join(self.terms)
		return tree_str

	def num_terms(self):
		return len(self.terms)

class Neuron:
	def __init__(self, weights, bias, feature_terms, feature_order):
		self.weights = weights		
		self.bias = bias
		self.feature_terms = [ft[:] for ft in feature_terms]

		positive_neuron = InputNeuron(self.weights, self.bias, self.feature_terms, feature_order)
		positive_forest_builder = ForestBuilder(positive_neuron)
		forest_positive = positive_forest_builder.get_forest()
		self.forest_positive = forest_positive

		negative_neuron = InputNeuron(list(-1*np.array(self.weights)), -1*self.bias, self.feature_terms, feature_order)
		negative_forest_builder = ForestBuilder(negative_neuron)
		forest_negative = negative_forest_builder.get_forest()
		for tree in forest_negative:
			tree.scale(0.0)
		self.forest_negative = forest_negative

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
		income_model.feature_terms)
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
		income_model.feature_terms,
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
		weights_bias = [tree.neuron.weights + [tree.neuron.bias] for tree in neuron_layer[i].forest]
		for wb in weights_bias:
			f.write(str(wb) + '\n')
		f.write("----------------" + '\n')
	print("Done")





def tree_and(tree_1, tree_2, w1, w2):
	if(len(tree_1.terms) > len(tree_2.terms)):
		longer_tree = tree_1
		smaller_tree = tree_2
	else:
		longer_tree = tree_2
		smaller_tree = tree_1
	if(smaller_tree.terms != longer_tree.terms[:len(smaller_tree.terms)]):
		return None 
	
	terms = longer_tree.terms
	
	weights_1 = tree_1.neuron.weights
	bias_1 = tree_1.neuron.bias

	weights_2 = tree_2.neuron.weights
	bias_2 = tree_2.neuron.bias

	weights = list(w1 * np.array(weights_1) + w2 * np.array(weights_2))
	bias = w1 * bias_1 + w2 * bias_2 
	assert(tree_1.neuron.feature_terms == tree_1.neuron.feature_terms)
	feature_terms = tree_1.neuron.feature_terms
	assert(tree_1.neuron.feature_order == tree_1.neuron.feature_order)
	feature_order = tree_1.neuron.feature_order

	neuron = InputNeuron(weights, bias, feature_terms, feature_order)
	return TreeBuilder(terms, neuron)

def tree_diff(tree_1, tree_2):
	i = 0
	while(tree_1.terms[i] == tree_2.terms[i]):
		i = i + 1
	if(tree_1.terms[i] > tree_2.terms[i]):
		return 1
	else:
		return -1

def get_firing(tree):
	if(sum(tree.neuron.weights) + tree.neuron.bias == 0):
		return 0 
	else:
		return 1

def forest_and(list_of_forests, weights, bias):
	trees_to_and = []
	firing = []
	for forest in list_of_forests:
		if(len(forest) == 0):
			return [] 
		else:
			tree = forest[0]
			trees_to_and.append(tree)
			firing.append(get_firing(tree))
	candidate = trees_to_and[0].copy()
	candidate.scale(weights[0])
	j = 0
	i = 1
	while(i < len(trees_to_and)):
		new_candidate = tree_and(candidate, trees_to_and[i], 1, weights[i])
		if(new_candidate == None):
			break
		if(new_candidate.terms != candidate.terms):
			j = i
		candidate = new_candidate
		i = i + 1
	if(i == len(trees_to_and)):
		candidate.add(bias)
		list_of_forests[j] = list_of_forests[j][1:]
		return [[candidate, firing]] + forest_and(list_of_forests, weights, bias)
	else:
		if(tree_diff(candidate, trees_to_and[i]) > 0):
			list_of_forests[i] = list_of_forests[i][1:]
		else:
			list_of_forests[j] = list_of_forests[j][1:]

		return forest_and(list_of_forests, weights, bias)

neuron_layer_forests = [neuron.forest for neuron in neuron_layer]
conditions = forest_and(neuron_layer_forests, income_model.weight_layer_2[:, 0], income_model.bias_layer_2[0])
result = []
for t in conditions:
	positive_neuron = t[0].neuron
	positive_forest_builder = ForestBuilder(positive_neuron)
	forest_positive = positive_forest_builder.get_forest(prior = t[0])
	result.extend(forest_positive)

print("Number of conditions: " + str(len(conditions)))
print("Number of trees: " + str(len(result)))

with open('tree_final.txt', 'w') as f:
	print("Writing")
	forest_str = [tree.to_string() for tree in result]
	for tree_str in forest_str:
		f.write(str(tree_str) + '\n')
	print("Done")

with open('behavior_final.txt', 'w') as f:
	print("Writing")
	weights_bias = [tree.neuron.weights + [tree.neuron.bias] for tree in result]
	for wb in weights_bias:
		f.write(str(wb) + '\n')
	print("Done")

