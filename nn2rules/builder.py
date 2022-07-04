import tensorflow as tf
import pandas as pd
import numpy as np
import time

class RuleListBuilder:
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

	def get_rule_list(self, sort = True, prior = None):
		rule_list = self.build_rule_list(prior = prior)
		if(sort is True):
			rule_list_str = [rule.to_string() for rule in rule_list]
			rule_list_str = np.array(rule_list_str)
			order = np.argsort(rule_list_str)
			
			sorted_rule_list = [rule_list[i] for i in order]
			return sorted_rule_list
		else:
			return rule_list

	def build_rule_list(self, prior = None):
		if(prior is not None):
			num_terms = len(prior.terms)
			sub_neuron_rule_list_builder = RuleListBuilder(self.weights[:], self.bias,
											self.reshaped_weights[num_terms:], 
											self.reshaped_bias + prior.value, 
											self.reshaped_feature_terms[num_terms:])
			rule_list = sub_neuron_rule_list_builder.build_rule_list()

			for i in range(len(rule_list)):
				rule_list[i].add_prefix(prior) 

			return rule_list

		if(self.is_always_inactive()):
			return []
		if(self.is_always_active()):
			return [RuleBuilder([], 0, 0, self.weights, self.bias)]
		
		root_choices = self.reshaped_feature_terms[0]
		root_weights = self.reshaped_weights[0]
		
		preconditions = [ RuleBuilder([root_choices[0]], root_weights[0], root_weights[0], self.weights, self.bias) ]
		rule_list = []
		
		N = len(root_choices)
		for i in range(N):
			root_term = root_choices[i]
			root_value = root_weights[i]
		
			rule_list_with_root = []
			for j in range(len(preconditions)):
				prior = preconditions[j].copy()
				prior.update_root(root_term, root_value)
				
				rule_list_with_prior = self.build_rule_list(prior = prior)	
				rule_list_with_root.extend(rule_list_with_prior)
			
			rule_list.extend(rule_list_with_root)		
			preconditions = rule_list_with_root		
		
		return rule_list


class RuleBuilder:
	def __init__(self, terms, value, root_value, weights, bias):
		self.terms = terms[:]
		self.value = value
		self.root_value = root_value
		self.weights = weights[:]
		self.bias = bias

	def add_prefix(self, prefix_rule):
		self.terms = prefix_rule.terms + self.terms
		self.value = prefix_rule.value + self.value 
		self.root_value = prefix_rule.root_value

	def update_root(self, new_root_term, new_root_value):
		self.terms[0] = new_root_term
		self.value = self.value - self.root_value + new_root_value 
		self.root_value = new_root_value

	def copy(self):
		return RuleBuilder(self.terms, self.value, self.root_value, self.weights, self.bias)

	def to_string(self):
		sep = " and "
		rule_str = sep.join(self.terms)
		return rule_str

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

