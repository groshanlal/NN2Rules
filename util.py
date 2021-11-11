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
		model = tf.keras.models.load_model('trained_model')
		self.layer_weights = []
		self.layer_bias = []

		for i in range(len(model.layers)):
			self.layer_weights.append(model.layers[i].get_weights()[0].T.tolist())
			self.layer_bias.append(model.layers[i].get_weights()[1].tolist())
		
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

	def get_feature_importance(self):
		feature_importance = []
		for i in range(len(self.layer_bias[0])):
			neuron = InputNeuron(self.layer_weights[0][i], 
				self.layer_bias[0][i], 
				self.reshaped_feature_terms)
			feature_importance.append([])
			for i in range(len(neuron.reshaped_feature_terms)):
				feature_importance[-1].append( (max(neuron.reshaped_weights[i]) - min(neuron.reshaped_weights[i])) / len(neuron.reshaped_feature_terms[i]) )


		feature_importance = np.array(feature_importance)
		feature_importance = np.mean(feature_importance, axis = 0).tolist()
		return feature_importance
		
	
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
			fw = self.weights[cum_sum : cum_sum + num_terms]
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

			self.reshaped_weights[i] = fw[order].tolist()
			self.reshaped_feature_terms[i] = terms[order].tolist()

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
		self.weights = (np.array(self.weights) * c).tolist()
		self.bias = self.bias * c
		self.value = self.value * c
		self.root_value = self.root_value * c
		return
	
	def add(self, c):
		self.bias = self.bias + c
		self.value = self.value + c
		return
	
class Neuron:
	def __init__(self, weights, bias, reshaped_feature_terms, feature_order, prior_terms = [], relu_activation = True):
		self.weights = weights		
		self.bias = bias
		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]

		positive_neuron = InputNeuron(self.weights, self.bias, self.reshaped_feature_terms, feature_order)
		positive_forest_builder = ForestBuilder(positive_neuron.weights, positive_neuron.bias,
			positive_neuron.reshaped_weights, positive_neuron.reshaped_bias, positive_neuron.reshaped_feature_terms)

		prior = None
		if(len(prior_terms) > 0):
			value = positive_forest_builder.score(prior_terms)
			root_value = positive_forest_builder.score(prior_terms[:1])			
			prior = TreeBuilder(prior_terms, value, root_value, self.weights, self.bias)

		forest_positive = positive_forest_builder.get_forest(prior = prior)
		self.forest_positive = forest_positive

		negative_neuron = InputNeuron((-1*np.array(self.weights)).tolist(), -1*self.bias, self.reshaped_feature_terms, feature_order)
		negative_forest_builder = ForestBuilder(negative_neuron.weights, negative_neuron.bias,
			negative_neuron.reshaped_weights, negative_neuron.reshaped_bias, negative_neuron.reshaped_feature_terms)

		prior = None
		if(len(prior_terms) > 0):
			value = negative_forest_builder.score(prior_terms)
			root_value = negative_forest_builder.score(prior_terms[:1])			
			prior = TreeBuilder(prior_terms, value, root_value, self.weights, self.bias)

		forest_negative = negative_forest_builder.get_forest(prior = prior)
		for tree in forest_negative:
			tree.scale(-1.0)
		if(relu_activation is True):
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

	def simplify_forest_terms(self, feature_term_nums, forest = None):
		if(forest is None):
			forest = [tree.terms for tree in self.list_of_trees]

		print(len(forest))
		grouped_forest = [[forest[0]]]
		for i in range(1, len(forest)):
			if(grouped_forest[-1][0][:-1] != forest[i][:-1]):
				grouped_forest.append([])
			grouped_forest[-1].append(forest[i])

		reduced_forest = []
		for i in range(len(grouped_forest)):
			forest_common_terms = grouped_forest[i]
			if(len(forest_common_terms) == feature_term_nums[len(forest_common_terms[0]) - 1]):
				reduced_forest.append(forest_common_terms[0][:-1])
			else:
				reduced_forest.extend(forest_common_terms)

		if(len(reduced_forest) == len(forest)):
			return reduced_forest
		else:
			return self.simplify_forest_terms(feature_term_nums, forest = reduced_forest)


