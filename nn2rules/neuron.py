import numpy as np
from nn2rules.builder import RuleListBuilder, RuleBuilder

class Linear:
	def __init__(self, weights, bias, reshaped_feature_terms, feature_order):
		self.weights = weights[:]
		self.bias = bias

		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]	
		self.feature_order = feature_order	

		self.__reshape_term_weights()
		self.__shift_term_weights()
		self.__sort_term_weights()
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

	
class Neuron:
	def __init__(self, weights, bias, reshaped_feature_terms, feature_order, prior_terms = [], relu_activation = True):
		self.weights = weights		
		self.bias = bias
		self.reshaped_feature_terms = [ft[:] for ft in reshaped_feature_terms]

		positive_neuron = Linear(self.weights, self.bias, self.reshaped_feature_terms, feature_order)
		positive_rule_list_builder = RuleListBuilder(positive_neuron.weights, positive_neuron.bias,
			positive_neuron.reshaped_weights, positive_neuron.reshaped_bias, positive_neuron.reshaped_feature_terms)

		prior = None
		if(len(prior_terms) > 0):
			value = positive_rule_list_builder.score(prior_terms)
			root_value = positive_rule_list_builder.score(prior_terms[:1])			
			prior = RuleBuilder(prior_terms, value, root_value, self.weights, self.bias)

		rule_list_positive = positive_rule_list_builder.get_rule_list(prior = prior)
		self.rule_list_positive = rule_list_positive

		negative_neuron = Linear((-1*np.array(self.weights)).tolist(), -1*self.bias, self.reshaped_feature_terms, feature_order)
		negative_rule_list_builder = RuleListBuilder(negative_neuron.weights, negative_neuron.bias,
			negative_neuron.reshaped_weights, negative_neuron.reshaped_bias, negative_neuron.reshaped_feature_terms)

		prior = None
		if(len(prior_terms) > 0):
			value = negative_rule_list_builder.score(prior_terms)
			root_value = negative_rule_list_builder.score(prior_terms[:1])			
			prior = RuleBuilder(prior_terms, value, root_value, self.weights, self.bias)

		rule_list_negative = negative_rule_list_builder.get_rule_list(prior = prior)
		for rule in rule_list_negative:
			rule.scale(-1.0)
		if(relu_activation is True):
			for rule in rule_list_negative:
				rule.scale(0.0)
		self.rule_list_negative = rule_list_negative

		rule_list = []
		i = 0
		j = 0
		while((i < len(rule_list_positive)) and (j < len(rule_list_negative))):
			if(rule_list_positive[i].to_string() < rule_list_negative[j].to_string()):
				rule_list.append(rule_list_positive[i])
				i = i + 1
			else:
				rule_list.append(rule_list_negative[j])
				j = j + 1
		while(i < len(rule_list_positive)):
			rule_list.append(rule_list_positive[i])
			i = i + 1
		while(j < len(rule_list_negative)):
			rule_list.append(rule_list_negative[j])
			j = j + 1
		self.rule_list = rule_list
		return

