import math

# ELU
def elu(x, alpha=1.0):
    return x if x > 0 else alpha * (math.exp(x) - 1)

elu_in = [-2,-1,0,1,2]
print("elu: ", [elu(e) for e in elu_in])

def elu_1(x, alpha=1.0):
    return 1 if x > 0 else alpha * math.exp(x)

elu_1_in = [-2,-1,0,1,2]
print("elu_1: ", [elu_1(e) for e in elu_1_in])

# Sigmoid
def sigmoid(x, alpha=1.0):
    return 1 / (1 + math.exp(-x))

sigmoid_in = [-10.0, 0.0, 1.0, 7.0, 10000.0]
print("sigmoid: ", [sigmoid(e) for e in sigmoid_in])

def sigmoid_1(x, alpha=1.0):
    return sigmoid(x) * (1 - sigmoid(x))

sigmoid_1_in = [-10.0, 0.0, 1.0, 7.0, 10000.0]
print("sigmoid_1: ", [sigmoid_1(e) for e in sigmoid_1_in])