import matplotlib.pyplot as plt
import seaborn as sns
from logic.dataloader import DataLoader

data_loader = DataLoader("../../../data/sales.txt")
X = data_loader.get_X()
Y = data_loader.get_Y()

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("clicks", fontsize=30)
plt.ylabel("sales", fontsize=30)
plt.plot(X, Y, "bo")
plt.show()

