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
		column_names = pd.read_csv("data/train.csv").columns[:-1]
		column_names_root = [x.split('_')[0] for x in column_names]

		self.feature_names = []
		self.feature_terms = []

		for i in range(len(column_names_root)):
			name = column_names_root[i]
			term = column_names[i]
			if(name not in self.feature_names):
				self.feature_names.append(name)
				self.feature_terms.append([])
			self.feature_terms[-1].append(term)
		return
	
class InputNeuron:
	def __init__(self, weights, bias, feature_terms, feature_order = None):
		self.feature_terms = [ft[:] for ft in feature_terms]
				
		self.weights = weights
		self.bias = bias

		self.__reshape_term_weights()
		self.__shift_term_weights()

		self.neuron_behavior = []
		for wt in self.weights:
			self.neuron_behavior.extend(wt)
		self.neuron_behavior.extend([self.bias])

		self.neuron_terms = []
		for ft in self.feature_terms:
			self.neuron_terms.extend(ft)

		self.__sort_term_weights()
		if(feature_order is not None):
			self.__order_features(feature_order)

	def __reshape_term_weights(self):
		reshaped_weights = []
		cum_sum = 0
		for ft in self.feature_terms:
			num_terms = len(ft)
			fw = list(self.weights[cum_sum : cum_sum + num_terms])
			reshaped_weights.append(fw)
			cum_sum = cum_sum + num_terms
		self.weights = reshaped_weights


	def __shift_term_weights(self):
		for i in range(len(self.feature_terms)):
			fw = self.weights[i]
			fw_min = min(fw)
			fw = [fw_terms - fw_min for fw_terms in fw]
			self.weights[i] = fw
			self.bias = self.bias + fw_min

	def __sort_term_weights(self):
		for i in range(len(self.feature_terms)):
			fw = np.array(self.weights[i])
			terms = np.array(self.feature_terms[i])

			order = np.argsort(-fw)

			self.weights[i] = list(fw[order])
			self.feature_terms[i] = list(terms[order])

	def __order_features(self, order):
		self.weights = [self.weights[i] for i in order]
		self.feature_terms = [self.feature_terms[i] for i in order]
		

class ForestBuilder:
	def __init__(self, weights, bias, feature_terms, neuron_behavior, neuron_terms):
		self.weights = [w[:] for w in weights]
		self.bias = bias
		self.feature_terms = [ft[:] for ft in feature_terms]

		self.neuron_behavior = neuron_behavior[:]
		self.neuron_terms = neuron_terms[:]
		return 
		
	def is_always_active(self):
		weights_min = [w[ -1] for w in self.weights]
		if(sum(weights_min) + self.bias >= 0):
			return True
		return False

	def is_always_inactive(self):
		weights_max = [w[ 0] for w in self.weights]		
		if(sum(weights_max) + self.bias < 0):
			return True
		return False

	def get_forest(self, sort = True):
		forest = self.build_forest()
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
			sub_neuron_forest_builder = ForestBuilder(self.weights[num_terms:], self.bias + prior.get_value(), self.feature_terms[num_terms:], self.neuron_behavior, self.neuron_terms)
			forest = sub_neuron_forest_builder.build_forest()

			for i in range(len(forest)):
				forest[i].terms = prior.terms + forest[i].terms 

			return forest

		if(self.is_always_inactive()):
			return []
		if(self.is_always_active()):
			return [TreeBuilder([], self.neuron_behavior, self.neuron_terms)]
		
		root_choices = self.feature_terms[0]
		
		preconditions = [ TreeBuilder([root_choices[0]], self.neuron_behavior, self.neuron_terms) ]
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
	def __init__(self, terms, neuron_behavior, neuron_terms):
		self.terms = terms
		self.neuron_behavior = neuron_behavior
		self.neuron_terms = neuron_terms

	def update_root(self, new_root_term):
		self.terms[0] = new_root_term

	def copy(self):
		terms = self.terms[:]
		neuron_behavior = self.neuron_behavior[:]
		neuron_terms = self.neuron_terms[:]
		return TreeBuilder(terms, neuron_behavior, neuron_terms)

	def get_value(self):
		value = 0 
		for i in range(len(self.terms)):
			j = self.neuron_terms.index(self.terms[i])
			value = value + self.neuron_behavior[j]
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
		positive_forest_builder = ForestBuilder(positive_neuron.weights, positive_neuron.bias, positive_neuron.feature_terms, positive_neuron.neuron_behavior, positive_neuron.neuron_terms)
		forest_positive = positive_forest_builder.get_forest()
		self.forest_positive = forest_positive

		negative_neuron = InputNeuron(-1*self.weights, -1*self.bias, self.feature_terms, feature_order)
		negative_forest_builder = ForestBuilder(negative_neuron.weights, negative_neuron.bias, negative_neuron.feature_terms, negative_neuron.neuron_behavior, negative_neuron.neuron_terms)
		forest_negative = negative_forest_builder.get_forest()
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
	for i in range(len(neuron.feature_terms)):
		feature_importance[-1].append( (max(neuron.weights[i]) - min(neuron.weights[i])) / len(neuron.feature_terms[i]) )


feature_importance = np.array(feature_importance)
feature_importance = np.mean(feature_importance, axis = 0)
feature_order = list(np.argsort(-feature_importance))

print("Features Names in Descending Order of Importance: ")
print([income_model.feature_names[feature_order[i]] for i in range(len(feature_order))]) 
print([len(income_model.feature_terms[feature_order[i]]) for i in range(len(feature_order))]) 
print()

neuron_layer = []

t = time.time()
for i in range(num_hidden_neurons):
	print("Neuron " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(num_hidden_neurons - i))) + " min")
		t = time.time()

	neuron = Neuron(income_model.weight_layer_1[:, i], 
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
		forest_str = [tree.neuron_behavior for tree in neuron_layer[i].forest]
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')
	print("Done")

rule_list = [neuron.forest for neuron in neuron_layer]
for i in range(len(rule_list)):
	rule_list[i] = [tree.to_string() for tree in rule_list[i]]

for rule in rule_list[0]:
	print(rule)


