import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1.0 - x)

class NeuralNetwork1:
    def __init__(self, x, y, hln):
        np.random.seed(1)
        self.W1 = np.random.rand(x.shape[1], hln) - 0.5  # Input-hidden weights
        self.W2 = np.random.rand(hln, y.shape[1]) - 0.5  # Hidden-output weights
        self.inp = x
        self.y = y  # Desired output, y
        self.out = np.zeros(self.y.shape)  # Actual output, ŷ
        self.k = 2.0  # Constant for ∂Loss(y,ŷ)/∂ŷ calculation

    def training(self):
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

if __name__ == "__main__":
    np.set_printoptions(suppress=True)  # Suppress scientific notation in print()
    np.set_printoptions(precision=3)  # Set precision to 3 digits after decimal point

    X = np.array([[0,0,0,0,0,0,0,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0, # 0
                   0,1,0,0,0,0,0,1,0,
                   0,1,0,0,0,0,0,1,0,
                   0,1,0,0,0,0,0,1,0,
                   0,1,0,0,0,0,0,1,0,
                   0,1,0,0,0,0,0,1,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,0,0,0,0,0,0,0],
                  
                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,
                   0,0,0,0,1,0,0,0,0, # 1
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0, # 2
                   0,0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,1,0,0,0,0,0,
                   0,0,1,0,0,0,0,0,0,
                   0,1,0,0,0,0,0,0,0,
                   0,1,1,1,1,1,1,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0, # 3
                   0,0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,0,1,0,0,0,
                   0,0,0,0,0,0,1,0,0,
                   0,0,1,0,0,0,0,1,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,
                   0,0,0,0,1,0,0,0,0, # 4
                   0,0,0,1,1,0,0,0,0,
                   0,0,1,0,1,0,0,0,0,
                   0,1,0,0,1,0,0,0,0,
                   0,1,1,1,1,1,1,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,1,0,0,
                   0,0,1,0,0,0,0,0,0, # 5
                   0,0,1,1,1,1,0,0,0,
                   0,0,0,0,0,0,1,0,0,
                   0,0,0,0,0,0,1,0,0,
                   0,0,0,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,0,1,0,0,1,0,0,
                   0,0,0,0,1,1,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,1,1,1,1,0,0,
                   0,0,1,0,0,0,0,0,0, # 6
                   0,0,1,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,0,1,1,0,0,0,
                   0,0,0,1,0,0,1,0,0, # 7
                   0,0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,1,0,0,0,0,0,
                   0,0,0,0,1,0,0,0,0,
                   0,0,0,1,0,0,0,0,0,
                   0,0,0,1,0,0,0,0,0,
                   0,0,0,1,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0, # 8
                   0,0,1,0,0,0,1,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,0,0,0,0,0,0,0],

                  [0,0,0,0,0,0,0,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,1,0,0,0,1,0,0, # 9
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,1,1,1,1,1,0,0,
                   0,0,0,0,0,0,1,0,0,
                   0,0,0,0,0,0,1,0,0,
                   0,0,1,0,0,0,1,0,0,
                   0,0,0,1,1,1,0,0,0,
                   0,0,0,0,0,0,0,0,0]])
                 
    y = np.array([[0,0,0,0],
                  [0,0,0,1],
                  [0,0,1,0],
                  [0,0,1,1],
                  [0,1,0,0],
                  [0,1,0,1],
                  [0,1,1,0],
                  [0,1,1,1],
                  [1,0,0,0],
                  [1,0,0,1]])
  # Desired output, y

    Ann = NeuralNetwork1(X, y, 7)  # Creating an ANN instance with 7 hidden neurons

    # Train the network for 1000 epochs
    for i in range(1000):
        Ann.training()

    # Test the network with all patterns
    # Test it to recognize digit 3 
    Xin = np.array(
        [0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,
         0,0,0,1,1,1,0,0,0, # 3
         0,0,1,0,0,0,1,0,0,
         0,0,0,0,0,1,0,0,0,
         0,0,0,0,1,0,0,0,0,
         0,0,0,0,0,1,0,0,0,
         0,0,0,0,0,0,1,0,0,
         0,0,1,0,0,0,0,1,0,
         0,0,0,1,1,1,1,0,0,
         0,0,0,0,0,0,0,0,0])

    Yout = Ann(Xin)
    print("\nResult for a digit 3:")
    print(Yout)
