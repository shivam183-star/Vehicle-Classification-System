from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("model/model.pkl")

X_test = pd.read_csv("data/Xtest.csv")
Y_test = pd.read_csv("data/Ytest.csv")

result = model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, result)*100:.2f}%")
print("\nConfusion Matrix: ")
print(confusion_matrix(Y_test, result))
print("\nClassification Report: ")
print(classification_report(Y_test, result))

true_counts = pd.Series(Y_test["vehicle_class"]).value_counts().sort_index()
pred_counts = pd.Series(result.ravel()).value_counts().sort_index()

labels = true_counts.index
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, true_counts, width, label='True Count')
plt.bar(x + width/2, pred_counts, width, label='Predicted Count')

plt.xlabel("Vehicle Class", fontsize=14, fontweight='bold', font = "Times New Roman", color = "#3F3FE3")
plt.ylabel("Number of Samples", fontsize=14, fontweight='bold', font = "Times New Roman", color = "#3F3FE3")
plt.title("True vs Predicted Vehicle Class Distribution", fontsize=16, fontweight='bold', font = "Times New Roman", color = "#CE2121")
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.savefig("plots/true_vs_predicted_distribution.png", dpi=300)
plt.show()
