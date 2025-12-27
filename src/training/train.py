from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df = pd.read_csv("data/Cars_Data.csv")

X = df.drop("vehicle_class", axis= 1)
Y = df["vehicle_class"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42)

X_test.to_csv("data/Xtest.csv", index= False)
Y_test.to_csv("data/Ytest.csv", index= False)

model = CatBoostClassifier(iterations= 300, 
                           learning_rate= 0.1,
                           depth= 6,
                           loss_function="MultiClass",
                           eval_metric= "Accuracy",
                           verbose= False)

model.fit(X_train, Y_train, cat_features= ["drivetrain", "brand"])

joblib.dump(model, "model/model.pkl")
print("Model Training Completed")