import matplotlib.pyplot as plt
import pandas as pd
import joblib

model = joblib.load("model/model.pkl")

df = pd.read_csv("data/Cars_Data.csv")
X = df.drop("vehicle_class", axis= 1)
importances = model.get_feature_importance()
features = X.columns

fi = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(9,6))
plt.barh(fi["Feature"], fi["Importance"], color="#84BDF6", edgecolor="black")
plt.gca().invert_yaxis()
plt.xlabel("Importance Score", fontsize=14, fontweight='bold', font = "Times New Roman", color = "#3150EB")
plt.title("Feature Importance", fontsize=16, fontweight='bold', font = "Times New Roman", color = "#CE2121")
plt.xticks(fontsize=12, font = "Times New Roman", color = "#512066")
plt.yticks(fontsize=12, font = "Times New Roman", color = "#512066")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300)
plt.show()
