# NN2Rules

## Introduction
We present an algorithm, NN2Rules, to convert a trained neural network into a rule list. Rule lists are more interpretable since they align better with the way humans make decisions. NN2Rules is a decompositional approach to rule extraction, i.e., it extracts a set of decision rules from the parameters of the trained neural network model. We show that the decision rules extracted have the same prediction as the neural network on any input presented to it, and hence the same accuracy. A key contribution of NN2Rules is that it allows hidden neuron behavior to be either soft-binary (eg. sigmoid activation) or rectified linear (ReLU) as opposed to existing decompositional approaches that were developed with the assumption of soft-binary activation. 

For more detailed introduction of NN2Rules, please check out our paper.

## Demo

Run the follwing python scripts to decompose a simple neural network into rules:

Step 1: Generate data:
```
python3 data_prep/data_prep_contraception.py
```
Step 2: Train a neural network model:
```
python3 train.py 3
```
Step 3: Explain the neural network using NN2Rules:
```
python3 explain.py
```

## Citation
Please cite [NN2Rules](https://arxiv.org/abs/2207.12271) in your publications if it helps your research:
```
@article{nn2rules2022,
  title={NN2Rules: Extracting Rule List from Neural Networks},
  author={Lal, G Roshan and Mithal, Varun},
  journal={arXiv preprint arXiv:2207.12271},
  year={2022}
}
```
