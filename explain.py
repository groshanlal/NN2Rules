import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import time

def get_weights():
	model = tf.keras.models.load_model('income_model')

	weight_layer_1 = model.layers[0].get_weights()[0]
	bias_layer_1   = model.layers[0].get_weights()[1]
	weight_layer_2 = model.layers[1].get_weights()[0]
	bias_layer_2   = model.layers[1].get_weights()[1]

	return [weight_layer_1, bias_layer_1, weight_layer_2, bias_layer_2]

def get_features():
	feature_columns = pd.read_csv("data/train.csv").columns[:-1]
	feature_root_names = [x.split('_')[0] for x in feature_columns]

	feature_names = []
	feature_terms = []
	feature_num_terms = []

	for i in range(len(feature_root_names)):
		name = feature_root_names[i]
		term = feature_columns[i]
		if(name not in feature_names):
			feature_names.append(name)
			feature_terms.append([term])
			feature_num_terms.append(1)
		if(term not in feature_terms[-1]):
			feature_terms[-1].append(term)
			feature_num_terms[-1] = feature_num_terms[-1] + 1 

	return [feature_names, feature_terms, feature_num_terms] 

def transform_weights(feature_weights, feature_terms, feature_num_terms, neuron_bias, feature_order = None):
	feature_weights_ordered = []
	feature_terms_ordered = []
	cum_sum = 0
	for i in range(len(feature_num_terms)):
		num_terms = feature_num_terms[i]
		weights = feature_weights[cum_sum : cum_sum + num_terms]
		terms = np.array(feature_terms[i])
		order = np.argsort(weights)

		weights = list(weights[order])
		terms = list(terms[order])
		feature_weights_ordered.append(weights)
		feature_terms_ordered.append(terms)

		cum_sum = cum_sum + num_terms
	
	feature_weights_min = [min(fw) for fw in feature_weights_ordered]
	for i in range(len(feature_weights_ordered)):
		for j in range(len(feature_weights_ordered[i])):
			feature_weights_ordered[i][j] = feature_weights_ordered[i][j] - feature_weights_min[i]
		neuron_bias = neuron_bias + feature_weights_min[i]
	
	if(feature_order is not None):
		feature_weights_ordered = [feature_weights_ordered[i] for i in feature_order]
		feature_terms_ordered = [feature_terms_ordered[i] for i in feature_order]
	
	return [feature_weights_ordered, feature_terms_ordered, neuron_bias]

def tree2str(tree):	
	sep = " and "
	tree_str = sep.join(tree)
	return tree_str

def forest2str(forest):	
	forest_str = [tree2str(tree) for tree in forest]
	forest_str_sorted = forest_str[:]
	forest_str_sorted.sort()
	return forest_str_sorted

def get_forest(feature_weights_ordered, feature_terms_ordered, neuron_bias):	
	if(len(feature_weights_ordered) == 0):
		if(neuron_bias < 0):
			return [], [], []
		else:
			return [[]], [0], [0]
	feature_weights_max = [fw[-1] for fw in feature_weights_ordered]
	feature_weights_min = [fw[0] for fw in feature_weights_ordered]
	
	if(sum(feature_weights_max) + neuron_bias < 0):
		return [], [], []
	if(sum(feature_weights_min) + neuron_bias >= 0):
		return [[]], [0], [0]

	forest = []
	tree_sum = []
	depth = []

	pivot_node = feature_terms_ordered[-1][-1]
	pivot_weight = feature_weights_ordered[-1][-1]

	forest_1 = [[pivot_node]]
	tree_sum_1 = [pivot_weight]
	depth_1 = [1] 


	N = len(feature_weights_ordered[-1])
	for index in range(N):
		forest_2 = [x[:] for x in forest_1]
		tree_sum_2 = tree_sum_1[:]
		depth_2 = depth_1[:]

		forest_1 = []
		tree_sum_1 = []
		depth_1 = []

		for i in range(len(forest_2)):
			if(index > 0):
				forest_2[i][0] = feature_terms_ordered[-1][N - 1 - index]
				tree_sum_2[i] = feature_weights_ordered[-1][N - 1 - index] - feature_weights_ordered[-1][N - index] + tree_sum_2[i] 

		for i in range(len(forest_2)):
			if(neuron_bias + tree_sum_2[i] >=0):
				forest.append(forest_2[i][:])
				tree_sum.append(tree_sum_2[i])
				depth.append(depth_2[i])

				forest_1.append(forest_2[i][:])
				tree_sum_1.append(tree_sum_2[i])
				depth_1.append(depth_2[i])
			else:
				forest_2i, tree_sum_2i, depth_2i = get_forest(feature_weights_ordered[:-depth_2[i]], feature_terms_ordered[:-depth_2[i]], neuron_bias + tree_sum_2[i])
				for j in range(len(forest_2i)):
					forest_2i[j] = forest_2[i] + forest_2i[j] 
					tree_sum_2i[j] = tree_sum_2[i] + tree_sum_2i[j] 
					depth_2i[j] = depth_2[i] + depth_2i[j] 
					

					forest.append(forest_2i[j][:])
					tree_sum.append(tree_sum_2i[j])
					depth.append(depth_2i[j])

					forest_1.append(forest_2i[j][:])
					tree_sum_1.append(tree_sum_2i[j])
					depth_1.append(depth_2i[j])


	return  forest, tree_sum, depth

def deserialize_forest(forest_str):
	forest = []
	for tree_str in forest_str:
		tree = deserialize_tree(tree_str)
		forest.append(tree)
	return forest

def deserialize_tree(tree_str):
	tree = tree_str.split(" and ")
	return tree


def logical_and(forest_0, forest_1):
	forest = []

	forest_len_0 = len(forest_0)
	forest_len_1 = len(forest_1)

	forest_str_0 = forest2str(forest_0)
	forest_str_1 = forest2str(forest_1)

	i = 0
	j = 0
	while((i < forest_len_0) and (j < forest_len_1)):
		tree_str_0 = forest_str_0[i]
		tree_str_1 = forest_str_1[j]

		tree_0 = deserialize_tree(tree_str_0)
		tree_1 = deserialize_tree(tree_str_1)

		N0 = len(tree_0)
		N1 = len(tree_1)

		k = 0
		while((k < N0) and (k < N1) and (tree_0[k] == tree_1[k])):
			k = k + 1
		if(k == N0):
			forest.append(tree_1)
			j = j + 1
		elif(k == N1):
			forest.append(tree_0)
			i = i + 1
		elif(tree_0[k] < tree_1[k]):
			i = i + 1
		else:
			j = j + 1

	return forest

def reduce_forest(forest_final):
	forest_final_2 = []
	i = 0
	j = 0
	while(i < len(forest_final)):
		tree_final = forest_final[j][:-1]
		fn = forest_final[j][-1].split("_")[0]
		while((j < len(forest_final)) and (tree2str(tree_final) == tree2str(forest_final[j][:-1]))):
			j = j + 1
		if(j - i == feature_num_terms[feature_names.index(fn)]):
			forest_final_2.append(tree_final)
		else:
			forest_final_2 = forest_final_2 + forest_final[i : j]
		i = j
	return forest_final_2

def recursive_reduce_forest(forest_final):
	forest_final = deserialize_forest(list(set(forest2str(forest_final))))
	forest_final = deserialize_forest((forest2str(forest_final)))
	forest_final_len = len(forest_final)
	forest_final = reduce_forest(forest_final)
	while(forest_final_len != len(forest_final)):
		forest_final_len = len(forest_final)
		forest_final = reduce_forest(forest_final)
	return forest_final

def binary_seq(n):
	assert(n > 0)

	if(n == 1):
		return [[0], [1]]
	if(n > 1):
		smaller_seq = binary_seq(n - 1)
		seq_0 = [[0] + s for s in smaller_seq[:]]
		seq_1 = [[1] + s for s in smaller_seq[:]]

		return seq_0 + seq_1

def get_conditions(n, forest_pos, forest_neg):
	assert(n > 0)

	if(n == 1):
		return [[0], [1]], [forest_neg[-1], forest_pos[-1]]
	if(n > 1):		
		smaller_seq, smaller_cond = get_conditions(n - 1, forest_pos, forest_neg)

		seq_0 = [[0] + s for s in smaller_seq[:]]
		seq_1 = [[1] + s for s in smaller_seq[:]]

		smaller_cond_0 = [logical_and(forest_neg[-n], c) for c in smaller_cond[:]]
		smaller_cond_1 = [logical_and(forest_pos[-n], c) for c in smaller_cond[:]]
		
		return seq_0 + seq_1, smaller_cond_0 + smaller_cond_1

[weight_layer_1, bias_layer_1, weight_layer_2, bias_layer_2] = get_weights()
num_hidden_neurons = len(bias_layer_1)
[feature_names, feature_terms, feature_num_terms] = get_features()

neuron_orders = []
for i in range(num_hidden_neurons):
	neuron_weights = weight_layer_1[:, i]
	neuron_bias = bias_layer_1[i]

	[feature_weights_ordered, feature_terms_ordered, neuron_bias] = transform_weights(neuron_weights, feature_terms, feature_num_terms, neuron_bias)
	ranges = []
	for i in range(len(feature_weights_ordered)):
		ranges.append(max(feature_weights_ordered[i]) - min(feature_weights_ordered[i]))
		ranges[i] = ranges[i] / feature_num_terms[i]
	neuron_orders.append(ranges)

neuron_orders = np.array(neuron_orders)
feature_order = np.mean(neuron_orders, axis = 0)

feature_order = list(np.argsort(np.array(feature_order)))

print("Features Names in Descending Order of Importance: ")
print([feature_names[feature_order[-1-i]] for i in range(len(feature_order))]) 
print([feature_num_terms[feature_order[-1-i]] for i in range(len(feature_order))]) 
print()

forest_pos = []
t = time.time()
for i in range(num_hidden_neurons):
	print("Positive Neuron " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(num_hidden_neurons - i))) + " min")
		t = time.time()

	neuron_weights = weight_layer_1[:, i]
	neuron_bias = bias_layer_1[i]
	
	[feature_weights_ordered, feature_terms_ordered, neuron_bias] = transform_weights(neuron_weights, feature_terms, feature_num_terms, neuron_bias, feature_order)
	
	feature_weights_min = [fw[0] for fw in feature_weights_ordered]
	feature_weights_max = [fw[-1] for fw in feature_weights_ordered]
	if(sum(feature_weights_min) + neuron_bias > 0):
		print("Neuron " + str(i) + " Useless: Always Positive")
	if(sum(feature_weights_max) + neuron_bias < 0):
		print("Neuron " + str(i) + " Useless: Always Negative")

	neuron_forest, _, _ = get_forest(feature_weights_ordered, feature_terms_ordered, neuron_bias)
	print(len(neuron_forest))
	forest_pos.append(neuron_forest)
	print("--------------------")

with open('tree_pos.txt', 'w') as f:
	for i in range(len(forest_pos)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = forest2str(forest_pos[i])	
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')

forest_neg = []
t = time.time()
for i in range(num_hidden_neurons):
	print("Negative Neuron " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(num_hidden_neurons - i))) + " min")
		t = time.time()

	neuron_weights = -1 * weight_layer_1[:, i]
	neuron_bias = -1 * bias_layer_1[i]

	[feature_weights_ordered, feature_terms_ordered, neuron_bias] = transform_weights(neuron_weights, feature_terms, feature_num_terms, neuron_bias, feature_order)

	neuron_forest, _, _ = get_forest(feature_weights_ordered, feature_terms_ordered, neuron_bias)
	print(len(neuron_forest))
	forest_neg.append(neuron_forest)
	print("--------------------")

with open('tree_neg.txt', 'w') as f:
	for i in range(len(forest_neg)):
		f.write("Neuron " + str(i) + ":" + '\n')
		forest_str = forest2str(forest_neg[i])	
		for tree_str in forest_str:
			f.write(str(tree_str) + '\n')
		f.write("----------------" + '\n')

t = time.time()
firings, conditions = get_conditions(num_hidden_neurons, forest_pos, forest_neg)
sec = time.time() - t
print("Computed Conditions in " + str(int(sec/60)) + " min")

weight_diagonal = np.diag(weight_layer_2[:,0])
weight_firing = np.matmul(weight_diagonal, np.array(firings).T)
weight_firing = weight_firing.T
neuron_firing_weights = np.matmul(weight_firing, weight_layer_1.T)
neuron_firing_bias = np.matmul(weight_firing, bias_layer_1.reshape(6, 1)) + bias_layer_2[0]

forest_final = []
t = time.time()
for i in range(len(firings)):
	print("Firing " + str(i))
	if(i > 0):
		sec = time.time() - t
		print("ETA: " + str(int(sec/60*(len(firings) - i))) + " min")
		t = time.time()
	
	precondition = conditions[i]
	neuron_weights = neuron_firing_weights[i]
	neuron_bias = neuron_firing_bias[i]

	
	if(len(precondition) > 0):
		print(len(precondition))
		print(str(int(time.time() - t)) + " sec")

		[feature_weights_ordered, feature_terms_ordered, neuron_bias] = transform_weights(neuron_weights, feature_terms, feature_num_terms, neuron_bias, feature_order)
		neuron_forest, _, _ = get_forest(feature_weights_ordered, feature_terms_ordered, neuron_bias)
		
		print(len(precondition))
		print(str(int(time.time() - t)) + " sec")
		
		neuron_forest = logical_and(precondition, neuron_forest)
		forest_str = forest2str(neuron_forest)	
		forest_final = forest_final + neuron_forest
		
		print(len(neuron_forest))
		print(str(int(time.time() - t)) + " sec")
	
			
	print(len(forest_final))

	print("--------------------")

forest_final = deserialize_forest((forest2str(forest_final)))
forest_final = recursive_reduce_forest(forest_final)
print(len(forest_final))

with open('tree_final.txt', 'w') as f:
	forest_str = forest2str(forest_final)	
	for tree_str in forest_str:
		f.write(str(tree_str) + '\n')

data_test = pd.read_csv("data/test.csv").to_numpy() 

x_test = data_test[:,:-1]
y_test = data_test[:, -1]

y_pred = np.matmul(x_test, weight_layer_1) + bias_layer_1
y_pred[y_pred < 0.0] = 0.0

y_pred = np.matmul(y_pred, weight_layer_2) + bias_layer_2
y_pred[y_pred < 0.0] = 0.0
y_pred[y_pred > 0.0] = 1.0
np.savetxt('ypred.csv', y_pred, fmt='%.3f', delimiter=', ')

[feature_names, feature_terms, feature_num_terms]  = get_features()
feature_names = [feature_names[feature_order[i]] for i in range(len(feature_order))]
feature_terms = [feature_terms[feature_order[i]] for i in range(len(feature_order))]


data_forest = []
data_forest_str = []
for index in range(len(y_pred)):
	x = x_test[index]
	x_parsed = []
	cum_sum = 0
	for i in range(len(feature_num_terms)):
		num_terms = feature_num_terms[i]
		x_terms = list(x[cum_sum : cum_sum + num_terms])

		x_parsed.append(x_terms)
		cum_sum = cum_sum + num_terms


	x_parsed = [x_parsed[feature_order[i]] for i in range(len(feature_order))]

	x_tree = []
	for i in range(len(x_parsed)):
		x_tree = [feature_terms[i][x_parsed[i].index(1)]] + x_tree
	data_forest.append(x_tree)
	data_forest_str.append(tree2str(x_tree))



result = logical_and(data_forest, forest_final)
result_str = forest2str(result)
for i in range(len(data_forest)):
	if(y_pred[i] == 0) and (data_forest_str[i] in result_str):
		print("Wrong")
	if(y_pred[i] == 1) and (data_forest_str[i] not in result_str):
		print("Wrong")

