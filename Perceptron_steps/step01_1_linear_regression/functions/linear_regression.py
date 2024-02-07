import numpy as np



# START:predict
def predict(X, w):
    return X * w
# END:predict


# START:loss
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)
# END:loss


# START:train
def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w

    raise Exception("Couldn't converge within %d iterations" % iterations)
# END:train


# START:main
# Import the dataset
#X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
#w = train(X, Y, iterations=10000, lr=0.01)
#print("\nw=%.3f" % w)

# Predict the number of sales
#print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))
# END:main
