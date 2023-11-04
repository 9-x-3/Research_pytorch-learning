import numpy as np

# f = w * x

# example our weight is 2 so the equation will be f = 2 * x

X = np.array([2,3,5,6], dtype = np.float32)
Y = np.array([4,6,10,12], dtype = np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss
def loss(y,y_pred):
    return ((y_pred -y)**2).mean()

#gradient
#MSE formula = 1/N * (w*x -y)**2
#dJ/dw = 1/N 2x (w*y -y)
def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f"Prediction before train f(5) {forward(4):.3f}")

# training

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    y_pred  = forward(X)
    loss  = loss (Y,y_pred)
    dw = gradient(X,Y,y_pred)