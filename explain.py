import tensorflow as tf
import pandas as pd
import numpy as np
import time
from nn2rules.importer import Model
from nn2rules.neuron import Neuron
from nn2rules.rule import RuleList

trained_model = Model(model_path='trained_model', data_path='data/train.csv')

for layer_num in range(len(trained_model.layer_weights)):
	print()
	print("Layer " + str(layer_num + 1))
	print(len(trained_model.layer_bias[layer_num]))

	if(len(trained_model.layer_weights) == layer_num + 1):
		isLastLayer = True
	else:
		isLastLayer = False


	neuron_weights_list = np.array(trained_model.layer_weights[layer_num])
	neuron_bias_list = np.array(trained_model.layer_bias[layer_num])


	if(layer_num == 0):
		neuron_rules = []
		for i in range(len(neuron_bias_list)):
			neuron = Neuron(neuron_weights_list[i], 
				neuron_bias_list[i], 
				trained_model.reshaped_feature_terms,
				trained_model.feature_order)	
			
			if(isLastLayer):	
				neuron_rules.append(neuron.rule_list_positive)
			else:
				neuron_rules.append(neuron.rule_list)
			
			print("Neuron " + str(i) + ": " + str(len(neuron_rules[-1])) + " rules")
		neuron_rules = [RuleList(nr) for nr in neuron_rules]
	else:

		print("Computing Conditions: Step 1: Merging previous layer rules")
		conditions = neuron_rules[0]
		for i in range(1, len(neuron_rules)):
			conditions.logical_and(neuron_rules[i])


		print("Computing Conditions: Step 2: Extending weights to next layer")
		for i in range(len(conditions.list_of_rules)):
			w = np.array(conditions.list_of_rules[i].list_of_weights)
			b = np.array(conditions.list_of_rules[i].list_of_bias)

			w = np.matmul(neuron_weights_list, w)
			b = np.matmul(neuron_weights_list, b.reshape(-1, 1))
			b = b + neuron_bias_list.reshape(-1, 1)
			b = b.reshape(-1) 

			conditions.list_of_rules[i].list_of_weights = w.tolist()
			conditions.list_of_rules[i].list_of_bias = b.tolist()

		print("Computing Conditions: Step 3: Extending rules to next layer")
		neuron_rules = []
		for j in range(len(neuron_bias_list)):
			neuron_rules.append([])
			for i in range(len(conditions.list_of_rules)):
				neuron = Neuron(conditions.list_of_rules[i].list_of_weights[j], 
					conditions.list_of_rules[i].list_of_bias[j], 
					trained_model.reshaped_feature_terms,
					trained_model.feature_order,
					prior_terms = conditions.list_of_rules[i].terms,
					relu_activation = not isLastLayer)
				if(isLastLayer):	
					neuron_rules[-1].extend(neuron.rule_list_positive)
				else:
					neuron_rules[-1].extend(neuron.rule_list)
			print("Neuron " + str(j) + ": " + str(len(neuron_rules[-1])) + " rules")

		neuron_rules = [RuleList(nr) for nr in neuron_rules]
	
	

rule_list_final = [rule.terms for rule in neuron_rules[0].list_of_rules]

print()
print("Simplifying rule list")
assert(len(neuron_rules) == 1)
feature_term_nums = [len(trained_model.reshaped_feature_terms[trained_model.feature_order[i]]) for i in range(len(trained_model.feature_order))]
conditions = neuron_rules[0]
rule_list_final = conditions.simplify_rule_list_terms(feature_term_nums)

print()
with open('rule_final.txt', 'w') as f:
	print("Writing")
	for rule in rule_list_final:
		rule_str = " and ".join(rule)
		f.write(str(rule_str) + '\n')
	print("Done")


