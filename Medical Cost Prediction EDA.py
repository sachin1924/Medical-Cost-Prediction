import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# DATA LOADING 

df = pd.read_csv(r"C:\Users\SACHIN\Downloads\archive (7)\medical_cost_prediction_dataset.csv")

print(df.head())
print(df.info())


# BASIC VISUALIZATION 

df['gender'].value_counts().plot(kind='bar')
plt.title("Number of Patients by Gender")
plt.xlabel("Gender (Male / Female)")
plt.ylabel("Number of Patients")
plt.show()

df['smoker'].value_counts().plot(kind='bar')
plt.title("Smoking Status of Patients")
plt.xlabel("Smoking Habit (Yes / No)")
plt.ylabel("Number of Patients")
plt.show()

plt.hist(df['age'], bins=20)
plt.title("Age Distribution of Patients")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Patients")
plt.show()

plt.hist(df['annual_medical_cost'], bins=20)
plt.title("Distribution of Annual Medical Cost")
plt.xlabel("Annual Medical Cost")
plt.ylabel("Number of Patients")
plt.show()



# PREPROCESSING 

df.ffill(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('annual_medical_cost', axis=1)
y_reg = df['annual_medical_cost']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# LINEAR REGRESSION 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\n--- LINEAR REGRESSION RESULTS ---")
print("Mean Absolute Error (Average Error):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score (Model Accuracy):", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.title("Actual Cost vs Predicted Cost")
plt.xlabel("Actual Annual Medical Cost")
plt.ylabel("Predicted Annual Medical Cost")
plt.show()



# CREATE CLASSIFICATION TARGET 

median_cost = df['annual_medical_cost'].median()
df['High_Cost'] = (df['annual_medical_cost'] > median_cost).astype(int)

y_class = df['High_Cost']

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)


#  LOGISTIC REGRESSION 

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

log_acc = accuracy_score(y_test, log_pred)
print("\nLogistic Regression Accuracy (High vs Low Cost):", log_acc)



# KNN 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)
print("\nKNN Accuracy (High vs Low Cost):", knn_acc)
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))



# DECISION TREE 

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
print("\nDecision Tree Accuracy (High vs Low Cost):", dt_acc)
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))



#  NAIVE BAYES 

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
print("\nNaive Bayes Accuracy (High vs Low Cost):", nb_acc)
print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))



# ACCURACY COMPARISON BAR GRAPH 

models = [
    'Logistic Regression',
    'KNN',
    'Decision Tree',
    'Naive Bayes'
]

accuracies = [
    log_acc,
    knn_acc,
    dt_acc,
    nb_acc
]

plt.bar(models, accuracies)
plt.title("Comparison of Classification Model Accuracy")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (Higher Value = Better Model)")
plt.ylim(0, 1)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc*100:.2f}%", ha='center')

plt.show()
