# Vehicle Classification using Machine Learning

This project classifies vehicles into multiple categories using **CatBoost** based on realistic vehicle features.  
---

## Vehicle Classes
The model predicts the following 5 classes:

1. **Sports / Hyper**
2. **SUV / MPV**
3. **Sedan / Hatchback**
4. **Truck**
5. **Off-Roader**

---

## Features Used
Some of the key input features include:
- Engine Power (HP)
- Torque (Nm)
- Vehicle Weight (kg)
- Ground Clearance (mm)
- Seating Capacity
- Top Speed (km/h)
- 0–100 km/h Time (sec)
- Brand
- Drive Train (AWD, FWD, RWD)

These features are chosen to ensure **clear class separability** with realistic overlap.

---

## Project Structure

```
.
├── data/
│   ├── Cars_Data.csv
│   ├── Xtest.csv
│   ├── Ytest.csv
│   └── generate_dataset.py
│
├── model/
│   └── model.pkl
│
├── plots/
│   ├── feature_importance.png
│   └── true_vs_predicted_distribution.png
│
├── src/
│   ├── training/
│   │   └── train.py
│   └── evaluation/
│       ├── evaluate.py
│       └── feature_importance.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

### Clone the repository
```bash
git clone https://github.com/shivam183-star/Vehicle-Classification-System.git
cd Vehicle-Classification-System
```
---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python src/training/train.py
```

This will:
- Load the dataset
- Train a CatBoost classifier
- Save the trained model as `model/model.pkl`

---

## Evaluate the Model

```bash
python src/evaluation/evaluate.py
```

Evaluation includes:
- Classification report
- Confusion matrix
- Test set predictions

---

## Feature Importance

```bash
python src/evaluation/feature_importance.py
```

Output is saved in:
```
plots/feature_importance.png
```

---

## Streamlit App

Run the web app locally:

```bash
python -m streamlit run app.py
```

The app:
- Loads the pretrained model
- Takes user inputs
- Predicts vehicle class instantly

---

## Deployment Notes
- `model.pkl` is included **only for deployment**
- Training artifacts and logs are ignored via `.gitignore`
- Dataset generation code is included for reproducibility

---

## Academic Note
This project uses **synthetic but domain-driven data** with realistic ranges and controlled overlap to avoid rule-based prediction and ensure valid ML learning.

---

## Requirements
See `requirements.txt`

---

## Author
**Shivam Singh**
- Student | Python Learner
- @shivam183-star

---
## Support
If you find this project useful, consider giving it a star on GitHub!

---

## License

For educational and academic use.
