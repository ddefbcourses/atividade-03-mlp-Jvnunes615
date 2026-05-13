import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_curve(values, title):
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def compare_models(names, scores, metric_name):
    plt.figure(figsize=(8, 4))
    plt.bar(names, scores)
    plt.title(f"Comparison of {metric_name}")
    plt.ylabel(metric_name)
    plt.xticks(rotation=15)
    plt.show()
