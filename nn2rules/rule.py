class Rule:
	def __init__(self, terms, list_of_weights, list_of_bias):
		self.terms = terms[:] 
		self.list_of_weights = [w[:] for w in list_of_weights] 
		self.list_of_bias = [b for b in list_of_bias]
	
	def diff(self, rule):
		i = 0
		while(self.terms[i] == rule.terms[i]):
			i = i + 1
		if(self.terms[i] > rule.terms[i]):
			return 1
		else:
			return -1

	def logical_and(self, rule):
		if(len(self.terms) > len(rule.terms)):
			longer_rule = self
			smaller_rule = rule
		else:
			longer_rule = rule
			smaller_rule = self
		if(smaller_rule.terms != longer_rule.terms[:len(smaller_rule.terms)]):
			return None
		result_rule = Rule(longer_rule.terms, 
			self.list_of_weights + rule.list_of_weights, 
			self.list_of_bias + rule.list_of_bias)
		return result_rule

class RuleList:
	def __init__(self, rule_list_builder):
		self.list_of_rules = [Rule(tb.terms, [tb.weights], [tb.bias]) for tb in rule_list_builder]

	def logical_and(self, rule_list):
		i = 0
		j = 0
		rule_list_conjunction = []
		while((i < len(self.list_of_rules)) and (j < len(rule_list.list_of_rules))):
			rule = self.list_of_rules[i].logical_and(rule_list.list_of_rules[j])
			if(rule is not None):
				rule_list_conjunction.append(rule)
				if(len(rule.terms) == len(self.list_of_rules[i].terms)):
					i = i + 1
				if(len(rule.terms) == len(rule_list.list_of_rules[j].terms)):
					j = j + 1
			else:
				if(self.list_of_rules[i].diff(rule_list.list_of_rules[j]) < 0):
					i = i + 1
				else:
					j = j + 1

		self.list_of_rules = rule_list_conjunction
		return

	def get_firing(self):
		firing = []
		for i in range(len(self.list_of_rules)):
			firing.append([])
			for j in range(len(self.list_of_rules[i].list_of_weights)):
				weight_sum = sum(self.list_of_rules[i].list_of_weights[j]) + \
									self.list_of_rules[i].list_of_bias[j]
				if(weight_sum > 0):
					weight_sum = 1.0
				firing[-1].append(weight_sum)
		return firing

	def simplify_rule_list_terms(self, feature_term_nums, rule_list = None):
		if(rule_list is None):
			rule_list = [rule.terms for rule in self.list_of_rules]

		print(len(rule_list))
		grouped_rule_list = [[rule_list[0]]]
		for i in range(1, len(rule_list)):
			if(grouped_rule_list[-1][0][:-1] != rule_list[i][:-1]):
				grouped_rule_list.append([])
			grouped_rule_list[-1].append(rule_list[i])

		reduced_rule_list = []
		for i in range(len(grouped_rule_list)):
			rule_list_common_terms = grouped_rule_list[i]
			if(len(rule_list_common_terms) == feature_term_nums[len(rule_list_common_terms[0]) - 1]):
				reduced_rule_list.append(rule_list_common_terms[0][:-1])
			else:
				reduced_rule_list.extend(rule_list_common_terms)

		if(len(reduced_rule_list) == len(rule_list)):
			return reduced_rule_list
		else:
			return self.simplify_rule_list_terms(feature_term_nums, rule_list = reduced_rule_list)


