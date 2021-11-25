import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rule_counts = pd.read_csv("tree_final_counts_train.txt", header = None).to_numpy()
rule_counts = rule_counts[:-1,0]
rule_counts = rule_counts[rule_counts > 0]
num_effective_rules = len(rule_counts)

rule_counts = pd.read_csv("tree_final_counts_test.txt", header = None).to_numpy()
rule_support = [0]*len(rule_counts[:-1, 0])
for i in range(len(rule_support)):
	if(i == 0):
		rule_support[i] = rule_counts[i][0]
	else:
		rule_support[i] = rule_support[i - 1] + rule_counts[i][0]
rule_support = np.array(rule_support)
num_rules = len(rule_support)

print("Num Rules")
print(num_rules)

print("Num Effective Rules")
print(num_effective_rules)

print("Support")
print(rule_support[num_effective_rules - 1])
plt.figure()
plt.plot(list(range(1, 1 + num_rules)), rule_support[:num_rules])
plt.xlabel("Num Rules")
plt.ylabel("Support")
plt.title("Support as a function of Num Rules")
plt.savefig("Support.png")

rule_fidelity = [0]*len(rule_counts[:-1, 0])
for i in range(len(rule_fidelity)):
	if(i == 0):
		rule_fidelity[i] = rule_counts[-1][0] + rule_counts[i][0]
	else:
		rule_fidelity[i] = rule_fidelity[i - 1] + rule_counts[i][0]
tot_num_data = rule_fidelity[-1]
rule_fidelity = rule_fidelity / tot_num_data

print("Fidelity")
print(rule_fidelity[num_effective_rules - 1])
plt.figure()
plt.plot(list(range(1, 1 + num_rules)), rule_fidelity[:num_rules])
plt.xlabel("Num Rules")
plt.ylabel("Fidelity")
plt.title("Fidelity as a function of Num Rules")
plt.savefig("Fidelity.png")

rule_error = [0]*len(rule_counts[:-1, 0])
for i in range(len(rule_error)):
	if(i == 0):
		rule_error[i] = rule_counts[-1][1]
		for j in range(len(rule_counts[:-1])):
			rule_error[i] = rule_error[i] + rule_counts[j][1]
		rule_error[i] = rule_error[i] - rule_counts[i][1] + rule_counts[i][2] 
	else:
		rule_error[i] = rule_error[i - 1] - rule_counts[i][1] + rule_counts[i][2] 
rule_error = rule_error / tot_num_data

print("Accuracy")
print(1 - rule_error[num_effective_rules - 1])
plt.figure()
plt.plot(list(range(1, 1 + num_rules)), 1 - rule_error[:num_rules])
plt.xlabel("Num Rules")
plt.ylabel("Accuracy")
plt.title("Accuracy as a function of Num Rules")
plt.savefig("Accuracy.png")


print("Fidelity on correct samples")
rule_fidelity = [0]*len(rule_counts[:-1, 0])
for i in range(len(rule_fidelity)):
	if(i == 0):
		rule_fidelity[i] = rule_counts[-1][2] + rule_counts[i][1]
	else:
		rule_fidelity[i] = rule_fidelity[i - 1] + rule_counts[i][1]
tot_num_data = rule_fidelity[-1]
rule_fidelity = rule_fidelity / tot_num_data

print(rule_fidelity[num_effective_rules - 1])

print("Fidelity on error samples")
rule_fidelity = [0]*len(rule_counts[:-1, 0])
for i in range(len(rule_fidelity)):
	if(i == 0):
		rule_fidelity[i] = rule_counts[-1][1] + rule_counts[i][2]
	else:
		rule_fidelity[i] = rule_fidelity[i - 1] + rule_counts[i][2]
tot_num_data = rule_fidelity[-1]
rule_fidelity = rule_fidelity / tot_num_data

print(rule_fidelity[num_effective_rules - 1])

