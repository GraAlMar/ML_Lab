import numpy as np


# START_HIGHLIGHT
def predict(X, w, b):
    return X * w + b
# END_HIGHLIGHT


# START_HIGHLIGHT
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)
# END_HIGHLIGHT


def train(X, Y, iterations, lr):
    # START_HIGHLIGHT
    w = b = 0
    # END_HIGHLIGHT
    for i in range(iterations):
        # START_HIGHLIGHT
        current_loss = loss(X, Y, w, b)
        # END_HIGHLIGHT
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        # START_HIGHLIGHT
        if loss(X, Y, w + lr, b) < current_loss:
            # END_HIGHLIGHT
            w += lr
        # START_HIGHLIGHT
        elif loss(X, Y, w - lr, b) < current_loss:
            # END_HIGHLIGHT
            w -= lr
        # START_HIGHLIGHT
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        # END_HIGHLIGHT
        else:
            # START_HIGHLIGHT
            return w, b
            # END_HIGHLIGHT

    raise Exception("Couldn't converge within %d iterations" % iterations)

