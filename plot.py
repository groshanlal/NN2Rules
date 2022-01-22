import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

adult = pd.read_csv("results/adult/sup_fid_err.csv", header = None).to_numpy()
contraception = pd.read_csv("results/contraception/sup_fid_err.csv", header = None).to_numpy()
nursery = pd.read_csv("results/nursery/sup_fid_err.csv", header = None).to_numpy()
cars = pd.read_csv("results/cars/sup_fid_err.csv", header = None).to_numpy()

plt.figure()
plt.plot(adult[:,0]*100, adult[:,2], label='adult')
plt.plot(contraception[:,0]*100, contraception[:,2], label='contraception')
plt.plot(nursery[:,0]*100, nursery[:,2], label='nursery')
plt.plot(cars[:,0]*100, cars[:,2], label='cars')
plt.legend()
plt.xlabel("Num Rules (Percentage)")
plt.ylabel("Fidelity")
plt.title("Fidelity as a function of Num Rules")
plt.savefig("Fidelity.pdf")

plt.figure()
plt.plot(adult[:,0]*100, adult[:,3], label='adult')
plt.plot(contraception[:,0]*100, contraception[:,3], label='contraception')
plt.plot(nursery[:,0]*100, nursery[:,3], label='nursery')
plt.plot(cars[:,0]*100, cars[:,3], label='cars')
plt.legend()
plt.xlabel("Num Rules (Percentage)")
plt.ylabel("Accuracy")
plt.title("Accuracy as a function of Num Rules")
plt.savefig("Accuracy.pdf")

fig = plt.figure("Line plot")
legendFig = plt.figure("Legend plot")
ax = fig.add_subplot(111)

line1, = ax.plot(adult[:,0]*100, adult[:,3], label='adult')
line2, = ax.plot(contraception[:,0]*100, contraception[:,3], label='contraception')
line3, = ax.plot(nursery[:,0]*100, nursery[:,3], label='nursery')
line4, = ax.plot(cars[:,0]*100, cars[:,3], label='cars')

legendFig.legend([line1, line2, line3, line4],['adult', 'contraception', 'nursery', 'cars'], loc='center', fontsize=20)
legendFig.savefig('legend.pdf')

tree = [0.941, 0.82, 0.984, 0.931]
trepan = [0.949, 0.936, 0.984, 0.951]
nn2rules_full = [1, 1, 1, 1]
nn2rules_support = [0.992, 0.959, 0.985, 0.945]

plt.figure()
fig, ax = plt.subplots()
barWidth = 0.20
#fig = plt.subplots(figsize =(12, 8))
 
# Set position of bar on X axis
br1 = np.arange(len(tree))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
 
# Make the plot
plt.bar(br1, tree, width = barWidth, label ='Decision Tree')
plt.bar(br2, trepan, width = barWidth, label ='TREPAN')
plt.bar(br3, nn2rules_full, width = barWidth, label ='NN2Rules(Full)')
plt.bar(br4, nn2rules_support, width = barWidth, label ='NN2Rules(Support)')

# Adding Xticks
plt.xlabel('Datsets', fontweight ='bold', fontsize = 15)
plt.ylabel('Fidelity', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(tree))],
        ['Adult', 'Contraception', 'Nursery', 'Cars'])
plt.ylim([0.75, 1.15]) 
plt.legend(ncol=2)
plt.savefig("Fidelity_Bar.pdf")

tree = [0.834, 0.798, 0.875, 0.462]
trepan = [0.855, 0.904, 0.875, 0.385]
nn2rules_full = [1, 1, 1, 1]
nn2rules_support = [0.982, 0.933, 0.531, 0.538]

plt.figure()
fig, ax = plt.subplots()
barWidth = 0.20
#fig = plt.subplots(figsize =(12, 8))
 
# Set position of bar on X axis
br1 = np.arange(len(tree))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
 
# Make the plot
plt.bar(br1, tree, width = barWidth, label ='Decision Tree')
plt.bar(br2, trepan, width = barWidth, label ='TREPAN')
plt.bar(br3, nn2rules_full, width = barWidth, label ='NN2Rules(Full)')
plt.bar(br4, nn2rules_support, width = barWidth, label ='NN2Rules(Support)')

# Adding Xticks
plt.xlabel('Datsets', fontweight ='bold', fontsize = 15)
plt.ylabel('Fidelity on Error', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(tree))],
        ['Adult', 'Contraception', 'Nursery', 'Cars'])
plt.ylim([0.30, 1.15]) 
plt.legend(ncol=2)
plt.savefig("Fidelity_Error_Bar.pdf")

