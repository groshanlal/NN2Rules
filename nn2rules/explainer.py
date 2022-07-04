import tensorflow as tf
import pandas as pd
import numpy as np
from nn2rules.neuron import Neuron
from nn2rules.rule import RuleList

class ModelExplainer:
	def __init__(self, model_path, data_path):
		self.model_path = model_path
		self.data_path = data_path

		self.__load_weights()
		self.__load_features()
		self.__reshape_feature_terms()

		self.__get_feature_order()
		return

	def __load_weights(self):
		model = tf.keras.models.load_model(self.model_path)
		self.layer_weights = []
		self.layer_bias = []

		for i in range(len(model.layers)):
			self.layer_weights.append(model.layers[i].get_weights()[0].T.tolist())
			self.layer_bias.append(model.layers[i].get_weights()[1].tolist())
		
		return

	def __load_features(self):
		self.feature_terms = list(pd.read_csv(self.data_path).columns[:-1])
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

	def __get_feature_order(self):
		feature_importance = []
		for i in range(len(self.layer_bias[0])):

			weights = self.layer_weights[0][i]
			bias = self.layer_bias[0][i]
			reshaped_feature_terms = [ft[:] for ft in self.reshaped_feature_terms]	
			
			reshaped_weights = []
			cum_sum = 0
			for ft in reshaped_feature_terms:
				num_terms = len(ft)
				fw = weights[cum_sum : cum_sum + num_terms]
				reshaped_weights.append(fw)
				cum_sum = cum_sum + num_terms

			feature_importance.append([])
			for i in range(len(reshaped_feature_terms)):
				feature_importance[-1].append( (max(reshaped_weights[i]) - min(reshaped_weights[i])) / len(reshaped_feature_terms[i]) )

		feature_importance = np.array(feature_importance)
		feature_importance = np.mean(feature_importance, axis = 0)

		self.feature_order = np.argsort(-feature_importance).tolist()
		return 

	def explain(self):
		for layer_num in range(len(self.layer_weights)):
			print()
			print("Layer " + str(layer_num + 1))
			print(len(self.layer_bias[layer_num]))

			if(len(self.layer_weights) == layer_num + 1):
				isLastLayer = True
			else:
				isLastLayer = False


			neuron_weights_list = np.array(self.layer_weights[layer_num])
			neuron_bias_list = np.array(self.layer_bias[layer_num])


			if(layer_num == 0):
				neuron_rules = []
				for i in range(len(neuron_bias_list)):
					neuron = Neuron(neuron_weights_list[i], 
						neuron_bias_list[i], 
						self.reshaped_feature_terms,
						self.feature_order)	
					
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
							self.reshaped_feature_terms,
							self.feature_order,
							prior_terms = conditions.list_of_rules[i].terms,
							relu_activation = not isLastLayer)
						if(isLastLayer):	
							neuron_rules[-1].extend(neuron.rule_list_positive)
						else:
							neuron_rules[-1].extend(neuron.rule_list)
					print("Neuron " + str(j) + ": " + str(len(neuron_rules[-1])) + " rules")

				neuron_rules = [RuleList(nr) for nr in neuron_rules]
			
			

		rule_list = [rule.terms for rule in neuron_rules[0].list_of_rules]

		print()
		print("Simplifying rule list")
		assert(len(neuron_rules) == 1)
		feature_term_nums = [len(self.reshaped_feature_terms[self.feature_order[i]]) for i in range(len(self.feature_order))]
		conditions = neuron_rules[0]
		rule_list = conditions.simplify_rule_list_terms(feature_term_nums)

		return rule_list

