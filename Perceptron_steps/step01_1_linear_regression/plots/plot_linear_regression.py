import matplotlib.pyplot as plt
import seaborn as sns
from logic.dataloader import DataLoader
from Perceptron_steps.step01_1_linear_regression.functions import linear_regression


data_loader = DataLoader("../../../data/sales.txt")
X = data_loader.get_X()
Y = data_loader.get_Y()

w = linear_regression.train(X,Y,iterations=10000, lr=0.01)

sns.set()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("clicks", fontsize=30)
plt.ylabel("sales", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [0, linear_regression.predict(x_edge, w)], linewidth=1.0, color="g")
plt.show()

