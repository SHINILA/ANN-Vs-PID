import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1.0 - x)

class NeuralNetworkV1:
    def __init__(self, x, y, hln):
        np.random.seed(1)
        self.W1 = np.random.rand(x.shape[1], hln) - 0.5  # Input-hidden weights
        self.W2 = np.random.rand(hln, y.shape[1]) - 0.5  # Hidden-output weights
        self.inp = x
        self.y = y  # Desired output, y
        self.out = np.zeros(self.y.shape)  # Actual output, ŷ
        self.k = 2.0  # Constant for ∂Loss(y,ŷ)/∂ŷ calculation
    
    def training(self, x, y):
        self.inp = x.reshape(1, x.size)  # Convert vector [...] to one-row matrix [[...]]
        self.y = y
        
        # Feedforward
        self.hid = sigmoid(np.dot(self.inp, self.W1))  # Hidden layer values
        self.out = sigmoid(np.dot(self.hid, self.W2))  # Output layer values

        # Backpropagation (starting from output to hidden layer)
        out_err = self.k * (self.y - self.out)  # ∂Loss(y,ŷ)/∂ŷ at the ANN output
        out_delta = out_err * sigmoid_der(self.out)  # ∂Loss(y,ŷ)/∂ŷ * ϐ’

        # ∂Loss(h,ĥ)/∂ĥ at the hidden layer output has to be calculated before W2 correction
        hid_err = np.dot(out_delta, self.W2.T)

        # Correct W2 weights
        self.W2 += np.dot(self.hid.T, out_delta)  # W2 += DW2

        # Backpropagation from hidden to input layer
        hid_delta = hid_err * sigmoid_der(self.hid)  # ∂Loss(h,ĥ)/∂ĥ * ϐ’

        # Correct W1 weights
        self.W1 += np.dot(self.inp.T, hid_delta)  # W1 += DW1

    def __call__(self, Mi):
        hid = sigmoid(np.dot(Mi, self.W1))  # Hidden layer values
        out = sigmoid(np.dot(hid, self.W2))  # Output layer values
        return out

# Initialize input and output
inp = np.array([[1.0]])  # Reshape to 2D array with one sample and one feature
oup = np.array([[1.0]])  # Reshape to 2D array with one sample and one output

# Create neural network
Ann = NeuralNetworkV1(inp, oup, 22)

# Scaling factors
xScale = 1 / np.pi
yScale = 0.5

# Number of samples and training iterations
samples = 25
trainings = 1000

print('\nTraining of Ann\n')
dx = np.pi / samples  # 25 values in the range <0 .. Pi>
for i in range(trainings):
    x = 0.0  # initial cos() argument
    for j in range(samples):
        Xin = np.array([[x * xScale]])  # limit inp to <0.0 .. 1.0>
        Yout = np.array([[(np.cos(x) + 1.0) * yScale]])  # limit out to <0.0 .. 1.0>
        Ann.training(Xin, Yout)
        x += dx

print('\nTesting of Ann\n')
dx = np.pi / (2 * samples)  # 50 values in the range <0 .. Pi>
x = 0.0
for j in range(2 * samples):
    Xin = np.array([[x * xScale]])  # limit inp to <0.0 .. 1.0>
    Yout = Ann(Xin) / yScale - 1.0  # re-scale out to <-1.0 .. +1.0>
    Yref = np.cos(x)
    print(np.array([x]), Yout, np.array([Yref]))
    x += dx
