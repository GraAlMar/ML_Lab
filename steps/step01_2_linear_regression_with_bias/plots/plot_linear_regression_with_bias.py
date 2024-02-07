import matplotlib.pyplot as plt
import seaborn as sns
from logic.dataloader import DataLoader
from steps.step01_2_linear_regression_with_bias.functions import linear_regression_with_bias

data_loader = DataLoader("../../../data/sales.txt")
X = data_loader.get_X()
Y = data_loader.get_Y()

w, b = linear_regression_with_bias.train(X, Y, iterations=10000, lr=0.01)

sns.set()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
# START_HIGHLIGHT
plt.plot([0, x_edge], [b, linear_regression_with_bias.predict(x_edge, w, b)], linewidth=1.0, color="g")
# END_HIGHLIGHT
plt.show()
