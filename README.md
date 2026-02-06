# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIJAYASHANKAR N
RegisterNumber: 25002971
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data.csv")
print(data.head())

data1 = data.copy()

data1.drop(['sl_no', 'salary'], axis=1, inplace=True)

print("\nMissing values:\n", data1.isnull().sum())
print("\nDuplicate values:", data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data1['gender'] = le.fit_transform(data1['gender'])
data1['ssc_b'] = le.fit_transform(data1['ssc_b'])
data1['hsc_b'] = le.fit_transform(data1['hsc_b'])
data1['hsc_s'] = le.fit_transform(data1['hsc_s'])
data1['degree_t'] = le.fit_transform(data1['degree_t'])
data1['workex'] = le.fit_transform(data1['workex'])
data1['specialisation'] = le.fit_transform(data1['specialisation'])
data1['status'] = le.fit_transform(data1['status'])

x = data1.iloc[:, :-1]
y = data1['status']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score
print("\nAccuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", confusion)

from sklearn.metrics import classification_report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn import metrics

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['Not Placed', 'Placed']
)

cm_display.plot()
plt.show()
*/
```

## Output:
<img width="572" height="88" alt="Screenshot 2026-02-06 111715" src="https://github.com/user-attachments/assets/f88e5b17-0031-46e0-b143-f13c4c55604a" />
<img width="341" height="47" alt="Screenshot 2026-02-06 111657" src="https://github.com/user-attachments/assets/af2c3107-7bc7-4433-8222-67cdd0da1052" />
<img width="293" height="82" alt="Screenshot 2026-02-06 111638" src="https://github.com/user-attachments/assets/14fcec35-2d8c-430c-ad8b-fc40c8018228" />
<img width="695" height="223" alt="Screenshot 2026-02-06 111620" src="https://github.com/user-attachments/assets/a91d7c30-cf6c-4a0a-a004-cdf78a0c5aca" />
<img width="751" height="560" alt="Screenshot 2026-02-06 111554" src="https://github.com/user-attachments/assets/0c79fb8b-f9f8-4f5a-871f-8bebcfebc9b1" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
