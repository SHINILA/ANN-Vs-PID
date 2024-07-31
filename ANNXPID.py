import math
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function and its derivative
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1.0 - x)

# Neural Network class
class NeuralNetworkV1:
    def __init__(self, x, y, hln):
        np.random.seed(1)
        self.W1 = np.random.rand(x.size, hln) - 0.5  # Input-hidden weights
        self.W2 = np.random.rand(hln, y.size) - 0.5  # Hidden-output weights
        self.inp = x
        self.y   = y  # Desired output, y
        self.out = np.zeros(self.y.size)  # Actual output, ŷ
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

# SecondOrderSystem class
class SecondOrderSystem: 
    def __init__(self, d1, d2, L):
        if (d1 > 0):
            e1 = -1.0 / d1
            x1 = math.exp(e1)
        else:
            x1 = 0
        if (d2 > 0):
            e2 = -1.0 / d2
            x2 = math.exp(e2)
        else:
            x2 = 0
        a = 1.0 - x1
        c = 1.0 - x2
        self.ac = a * c
        self.bplusd = x1 + x2
        self.bd = x1 * x2
        self.L = L
        self.Yppr = 0
        self.Ypr = 0
 
    def __call__(self, X): 
        if X > self.L:
            X = self.L
        elif X < -self.L:
            X = -self.L
        Y = self.ac * X + self.bplusd * self.Ypr - self.bd * self.Yppr
        self.Yppr = self.Ypr
        self.Ypr = Y
        return Y

# FirstOrderSystem class
class FirstOrderSystem:
    def __init__(self, d):
        if d > 0:
            e = -1.0 / d
            x = math.exp(e)
        else:
            x = 0
        self.a0 = 0.5 * (1 + x)
        self.a1 = -0.5 * (1 + x)
        self.b1 = x
        self.Ypr = 0
        self.Xpr = 0
 
    def __call__(self, X):
        Y = self.a0 * X + self.a1 * self.Xpr + self.b1 * self.Ypr
        self.Xpr = X
        self.Ypr = Y
        return Y

# PIDControlBlock class
class PIDControlBlock:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
 
    def __call__(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# HighPassFilter class
class HighPassFilter:
    def __init__(self, d):
        if d > 0:
            e = -1.0 / d
            x = math.exp(e)
        else:
            x = 0
        self.a0 = 0.5 * (1 + x)
        self.a1 = -0.5 * (1 + x)
        self.b1 = x
        self.Ypr = 0
        self.Xpr = 0
 
    def __call__(self, X):
        Y = self.a0 * X + self.a1 * self.Xpr + self.b1 * self.Ypr
        self.Xpr = X
        self.Ypr = Y
        return Y

# FFplusFBsystem class
class FFplusFBsystem:
    def __init__(self, Fy, Pid, Fu, plant):
        self.P = plant
        self.C = Pid
        self.Fy = Fy
        self.Fu = Fu
        self.Ypr = 0
 
    def __call__(self, X):
        E = self.Fy(X) - self.Ypr
        U = self.C(E) + self.Fu(X)
        Y = self.P(U)
        self.Ypr = Y
        return Y, U

# Parameters
maxVal = 1000.0
errScale = 1 / 1000.0
uOffset = 1000.0
uScale = 1 / 4000.0
trainings = 1000
samples = 1000
trainVal = 500.0
stepVal = 750.0
T1 = 250
T2 = 100
Ty = 250
Tu = 125
Kp = 2.0
Ki = 0.004
Kd = 175

# Create and train ANN
inp = np.array([0.0])
oup = np.array([0.0])
Ann1 = NeuralNetworkV1(inp, oup, 11)

print('\nTraining of Ann1 \n')
for i in range(trainings):
    SOSref = SecondOrderSystem(T1, T2, maxVal)  # maxVal is passed to Limiter class 
    PIDref = PIDControlBlock(Kp, Ki, Kd) 
    FINref = FirstOrderSystem(Ty)
    FUNref = HighPassFilter(Tu)
    RefContrSys = FFplusFBsystem(FINref, PIDref, FUNref, SOSref)
    for j in range(samples):
        Y, U = RefContrSys(trainVal)
        Xin = np.array([(trainVal - Y) * errScale])
        Yout = np.array([(U + uOffset) * uScale])
        Ann1.training(Xin, Yout)

# AnnFbFfControl class
class AnnFbFfControl:
    def __init__(self, ann, plant):
        self.Plt = plant
        self.Ann = ann
        self.Ypr = 0
        self.Scale = 0
        self.Xpr = 0
        self.Orig = 0

    def __call__(self, X, trainVal):
        if X != self.Xpr:  # just once when new X issued
            self.Scale = trainVal / (X - self.Xpr)
            self.Orig = self.Xpr
            self.Xpr = X
        Xi = [(X - self.Ypr) * self.Scale * errScale]
        Xin = np.array([Xi])
        Yout = self.Ann(Xin) / uScale - uOffset
        U = Yout / self.Scale + self.Orig
        Y = self.Plt(U)
        self.Ypr = Y
        return Y

# Create ANN-based control system and test it
Plant1 = SecondOrderSystem(T1, T2, maxVal)
Ann1ContrSys = AnnFbFfControl(Ann1, Plant1)

# PID Control System for comparison
PIDContrSys = PIDControlBlock(Kp, Ki, Kd)

# Initialize lists to store results for plotting
ann_outputs = []
pid_outputs = []
setpoint = stepVal

# Simulate both control systems
print("\nSimulating control systems\n")
for j in range(samples):
    ann_output = Ann1ContrSys(stepVal, trainVal)
    pid_output = Plant1(PIDContrSys(setpoint - Plant1.Ypr))

    ann_outputs.append(ann_output)
    pid_outputs.append(pid_output)

    # Print the output Y values
    print(f"Sample {j+1} - ANN Output: {ann_output}, PID Output: {pid_output}")

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(ann_outputs,'r-', label='ANN Control Output')
plt.plot(pid_outputs,'y', label='PID Control Output')
plt.axhline(y=setpoint, color='g', linestyle='-', label='Setpoint')
plt.xlabel('Sample')
plt.ylabel('Output')
plt.title('ANN vs PID Control Outputs')
plt.legend()
plt.show()
