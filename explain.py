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
	def __init__(self, weights, bias, feature_terms):
		self.feature_terms = [ft[:] for ft in feature_terms]
		
		self.weights = []
		self.bias = bias

		cum_sum = 0
		for ft in feature_terms:
			num_terms = len(ft)
			fw = list(weights[cum_sum : cum_sum + num_terms])
			self.weights.append(fw)
			cum_sum = cum_sum + num_terms

		self.__shift_term_weights()
		self.__sort_term_weights()

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

			order = np.argsort(fw)

			self.weights[i] = list(fw[order])
			self.feature_terms[i] = list(terms[order])

	def order_features(self, order):
		self.weights = [self.weights[i] for i in order]
		self.feature_terms = [self.feature_terms[i] for i in order]

	def get_forest(self):
		forest_builder = ForestBuilder(self.weights, 
			self.bias, self.feature_terms)
		return forest_builder.get_forest()
		

class ForestBuilder:
	def __init__(self, weights, bias, feature_terms):
		self.weights = [w[:] for w in weights]
		self.bias = bias
		self.feature_terms = [ft[:] for ft in feature_terms]
		return 

	def is_always_active(self):
		weights_min = [w[ 0] for w in self.weights]
		if(sum(weights_min) + self.bias >= 0):
			return True
		return False

	def is_always_inactive(self):
		weights_max = [w[-1] for w in self.weights]		
		if(sum(weights_max) + self.bias < 0):
			return True
		return False

	def get_forest(self):
		list_of_tree_builders = self.build_forest()
		list_of_trees = [Tree(tb.terms) for tb in list_of_tree_builders]
		return Forest(list_of_trees)

	def build_forest(self):
		if(self.is_always_inactive()):
			return []
		if(self.is_always_active()):
			return [TreeBuilder([], 0, 0)]
		
		root_choices = self.feature_terms[-1]
		root_weights = self.weights[-1]
		
		guiding_forest = [ TreeBuilder([root_choices[-1]], root_weights[-1], root_weights[-1]) ]
		forest = []
		
		N = len(root_choices)
		for i in range(N):
			root_term = root_choices[N - 1 - i]
			root_value = root_weights[N - 1 - i]

			forest_at_root = []
			for j in range(len(guiding_forest)):
				tree_builder = guiding_forest[j].copy()
				tree_builder.update_root(root_term, root_value)
				
				tree_growth = self.grow(tree_builder)	
				forest_at_root.extend(tree_growth)
			
			forest.extend(forest_at_root)		
			guiding_forest = forest_at_root		
		
		return forest
	
	def grow(self, tree_builder):
		num_terms = len(tree_builder.terms)
		sub_neuron_forest_builder = ForestBuilder(self.weights[:-num_terms], self.bias + tree_builder.value, self.feature_terms[:-num_terms])
		tree_growth = sub_neuron_forest_builder.build_forest()

		for k in range(len(tree_growth)):
			tree_growth[k].terms = tree_builder.terms + tree_growth[k].terms 
			tree_growth[k].value = tree_builder.value + tree_growth[k].value 
			tree_growth[k].root_value = tree_builder.root_value

		return tree_growth
		


class TreeBuilder:
	def __init__(self, terms, value, root_value):
		self.terms = terms
		self.value = value
		self.root_value = root_value

	def update_root(self, new_root_term, new_root_value):
		self.terms[0] = new_root_term
		self.value = self.value - self.root_value + new_root_value 
		self.root_value = new_root_value

	def copy(self):
		terms = self.terms[:]
		return TreeBuilder(terms, self.value, self.root_value)


class Tree:
	def __init__(self, list_of_terms):
		self.list_of_terms = list_of_terms
		return

	def to_string(self):
		sep = " and "
		tree_str = sep.join(self.list_of_terms)
		return tree_str

	def num_terms(self):
		return len(self.list_of_terms)

	def logical_and(self, tree):
		i = 0
		while(self.list_of_terms[i] == tree.list_of_terms[i]):
			i = i + 1
			if(i == self.num_terms()):
				return tree, i
			if(i == tree.num_terms()):
				return self, i
		return None, i

	def make_builder(self, weights, bias, feature_terms):
		value = 0
		root_value = 0
		for i in range(len(self.list_of_terms)):
			j = 0
			while(self.list_of_terms[i] != feature_terms[-1 - i][j]):
				j = j + 1
			value = value + weights[-1 - i][j]
			if(i == 0):
				root_value = root_value + weights[-1 - i][j]
		return TreeBuilder(self.list_of_terms[:], value, root_value)

	def diff(self, tree):
		if(self.list_of_terms[:-1] == tree.list_of_terms[:-1]):
			if(self.list_of_terms[-1].split("_")[0] == tree.list_of_terms[-1].split("_")[0]):
				return 0
		return 1



class Forest:
	def __init__(self, list_of_trees):
		self.list_of_trees = list_of_trees
		
		forest_str = self.to_string()
		forest_str.sort()
		
		forest = []
		for tree_str in forest_str:
			tree = Tree(tree_str.split(" and "))
			forest.append(tree)
		self.list_of_trees = forest

	def num_trees(self):
		return len(self.list_of_trees)

	def to_string(self):
		forest_str = [tree.to_string() for tree in self.list_of_trees]
		return forest_str

	def logical_and(self, forest):
		forest_conjunction = []

		i = 0
		j = 0
		while((i < self.num_trees()) and (j < forest.num_trees())):
			tree_0 = self.list_of_trees[i]
			tree_1 = forest.list_of_trees[j]

			tree_conjunction, k = tree_0.logical_and(tree_1)

			if(tree_conjunction is not None):
				forest_conjunction.append(tree_conjunction)
				if(k == tree_0.num_terms()):
					j = j + 1
				if(k == tree_1.num_terms()):
					i = i + 1
			else:
				if(tree_0.list_of_terms[k] < tree_1.list_of_terms[k]):
					i = i + 1
				else:
					j = j + 1

		return Forest(forest_conjunction)	

	def search_similar(self, tree, starting_pos = 0):
		tree_at_start = self.list_of_trees[starting_pos]
		assert(tree_at_start.diff(tree) == 0)

		terms = set()

		start = starting_pos
		tree_at_start = self.list_of_trees[start]
		while(tree.diff(tree_at_start) == 0):
			terms.add(tree_at_start.list_of_terms[-1])
			start = start - 1
			if(start < 0):
				break
			else:
				tree_at_start = self.list_of_trees[start]
		start = start + 1

		end = starting_pos
		tree_at_end = self.list_of_trees[end]
		while(tree.diff(tree_at_end) == 0):
			terms.add(tree_at_end.list_of_terms[-1])
			end = end + 1
			if(end == self.num_trees()):
				break
			else:
				tree_at_end = self.list_of_trees[end]

		return [start, end], terms




	def optimize(self, feature_terms):
		checked = [0]*len(self.list_of_trees)

		calls = 0
		while(sum(checked) != len(checked)):
			print("Reduce Call " + str(calls))
			calls = calls + 1
			checked = self.combine_trees(checked, feature_terms)
		return



	def combine_trees(self, checked, feature_terms):
		reduced_list_of_trees = []
		reduced_checked = []

		i = 0
		while(i < self.num_trees()):
			tree = self.list_of_trees[i]
			if(checked[i] == 1):
				reduced_list_of_trees.append(tree)
				reduced_checked.append(1)
				i = i + 1
			else:
				[start, end], terms = self.search_similar(tree, starting_pos = i)

				count = 0
				while(count < i - start):
					reduced_list_of_trees.pop()
					reduced_checked.pop()				
					count = count + 1			
				
				if(set(terms) == set(feature_terms[-tree.num_terms()])):
					reduced_list_of_trees.append(Tree(tree.list_of_terms[:-1]))
					reduced_checked.append(0)
				else:
					reduced_list_of_trees.extend(self.list_of_trees[start : end])
					reduced_checked.extend([1]*(end - start))
				i = end
		
		self.list_of_trees = reduced_list_of_trees
		return reduced_checked


class Neuron:
	def __init__(self, weights, bias, feature_terms, order):
		self.weights = weights		
		self.bias = bias
		self.feature_terms = [ft[:] for ft in feature_terms]

		positive = InputNeuron(self.weights, self.bias, self.feature_terms)
		positive.order_features(order)
		self.forest_positive = positive.get_forest()

		negative = InputNeuron(-1*self.weights, -1*self.bias, self.feature_terms)
		negative.order_features(order)
		self.forest_negative = negative.get_forest()
		return


income_model = ModelWeights()

_ , num_hidden_neurons =  np.shape(income_model.weight_layer_1)

neuron_orders = []
for i in range(num_hidden_neurons):
	neuron = InputNeuron(income_model.weight_layer_1[:, i], 
		income_model.bias_layer_1[i], 
		income_model.feature_terms)
	ranges = []
	for i in range(len(neuron.feature_terms)):
		ranges.append(max(neuron.weights[i]) - min(neuron.weights[i]))
		ranges[i] = ranges[i] / len(neuron.feature_terms[i])
	neuron_orders.append(ranges)


neuron_orders = np.array(neuron_orders)
feature_order = np.mean(neuron_orders, axis = 0)
feature_order = list(np.argsort(np.array(feature_order)))

print("Features Names in Descending Order of Importance: ")
print([income_model.feature_names[feature_order[-1-i]] for i in range(len(feature_order))]) 
print([len(income_model.feature_terms[feature_order[-1-i]]) for i in range(len(feature_order))]) 
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

	print("Positive: " + str(neuron.forest_positive.num_trees()))
	print("Negative: " + str(neuron.forest_negative.num_trees()))

	neuron_layer.append(neuron)


	print("--------------------")

with open('tree_pos.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = neuron_layer[i].forest_positive.to_string()
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')	
	print("Done")
	
with open('tree_neg.txt', 'w') as f:
	print("Writing")
	for i in range(len(neuron_layer)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = neuron_layer[i].forest_negative.to_string()	
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')
	print("Done")

def get_partitions(neuron_layer, combo_weights, combo_bias):
	num_neurons = len(neuron_layer)
	if(num_neurons == 1):
		neg_weights = [0*combo_weights[0]*neuron_layer[0].weights]
		neg_bias = [0*combo_weights[0]*neuron_layer[0].bias + combo_bias]
		neg_conditions = [neuron_layer[0].forest_negative]
		neg_firing = [[0]]

		pos_weights = [1*combo_weights[0]*neuron_layer[0].weights]
		pos_bias = [1*combo_weights[0]*neuron_layer[0].bias + combo_bias]
		pos_conditions = [neuron_layer[0].forest_positive]
		pos_firing = [[1]]

		return neg_weights + pos_weights, neg_bias + pos_bias, neg_conditions + pos_conditions, neg_firing + pos_firing
	else:
		weights, bias, conditions, firing = get_partitions(neuron_layer[1:], combo_weights[1:], combo_bias)
		neg_weights = [0*combo_weights[0]*neuron_layer[0].weights + w for w in weights]
		neg_bias = [0*combo_weights[0]*neuron_layer[0].bias + b for b in bias]
		neg_conditions = [neuron_layer[0].forest_negative.logical_and(c) for c in conditions]
		neg_firing = [[0] + f for f in firing]

		pos_weights = [1*combo_weights[0]*neuron_layer[0].weights + w for w in weights]
		pos_bias = [1*combo_weights[0]*neuron_layer[0].bias + b for b in bias]
		pos_conditions = [neuron_layer[0].forest_positive.logical_and(c) for c in conditions]
		pos_firing = [[1] + f for f in firing]

		return neg_weights + pos_weights, neg_bias + pos_bias, neg_conditions + pos_conditions, neg_firing + pos_firing

t = time.time()
print("Computing Conditions ... ")
weights_final, bias_final, conditions_final, firing_final = get_partitions(neuron_layer, income_model.weight_layer_2, income_model.bias_layer_2)
sec = time.time() - t
print("Computed Conditions in " + str(int(sec/60)) + " min")


forest_final = []
count = 0

t = time.time()
for i in range(len(firing_final)):
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(len(firing_final) - i))) + " min")
		t = time.time()
	
	print(firing_final[i])
	print("Conditions " + str(len(conditions_final[i].list_of_trees)))	

	neuron = InputNeuron(weights_final[i], bias_final[i], income_model.feature_terms)	
	neuron.order_features(feature_order)
	forest_builder = ForestBuilder(neuron.weights, neuron.bias, neuron.feature_terms)
	
	forest = []
	for tree in conditions_final[i].list_of_trees:
		tree_builder = tree.make_builder(neuron.weights, neuron.bias, neuron.feature_terms)
		list_of_tree_builders = forest_builder.grow(tree_builder)
		list_of_trees = [Tree(tb.terms) for tb in list_of_tree_builders]
		forest.extend(list_of_trees)
	
	print("Trees: " + str(len(forest)))

	forest_final.append(forest)	
	count = count + len(forest)
	print("Total : " + str(count))
	
	print('--------------------')


forest_final = [tree for forest in forest_final for tree in forest]
forest_final = Forest(forest_final)
print(forest_final.num_trees())
forest_final.optimize(neuron.feature_terms)
print(forest_final.num_trees())


with open('tree_final.txt', 'w') as f:
	print("Writing")
	forest_str = forest_final.to_string()	
	for tree_str in forest_str:
		f.write(str(tree_str) + '\n')
	print("Done")

print("Verifying RuleList")

x_test = pd.read_csv("data/test.csv").to_numpy()[:,:-1] 
column_names = list(pd.read_csv("data/test.csv").columns[:-1])

y_pred = np.matmul(x_test, income_model.weight_layer_1) + income_model.bias_layer_1
y_pred[y_pred < 0.0] = 0.0

y_pred = np.matmul(y_pred, income_model.weight_layer_2) + income_model.bias_layer_2
y_pred[y_pred < 0.0] = 0.0
y_pred[y_pred > 0.0] = 1.0

y_pred = y_pred.reshape(-1)


x_test_trees = []
for i in range(len(x_test)):
	x = InputNeuron(x_test[i], 0, income_model.feature_terms)
	x.order_features(feature_order)		

	xt = []
	for j in range(len(x.weights)):
		for k in range(len(x.weights[j])):
			if(x.weights[j][k] == 1):
				xt.append(x.feature_terms[j][k])
	xt.reverse()
	x_test_trees.append(Tree(xt))

x_test_result = Forest(x_test_trees).logical_and(forest_final)

x_test_result = x_test_result.to_string() 
x_test_result = set(x_test_result)

for i in range(len(x_test_trees)):
	if(x_test_trees[i].to_string() in x_test_result):
		if(y_pred[i] == 0):
			print("Wrong!")
	else:
		if(y_pred[i] == 1):
			print("Wrong!")
