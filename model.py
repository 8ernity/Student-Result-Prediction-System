import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("data.csv")

# Convert result to numbers
data['result'] = data['result'].map({'Fail': 0, 'Pass': 1})

# Features and target
X = data[['study_hours', 'attendance', 'previous_marks']]
y = data['result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# User input prediction
print("\nEnter Student Details:")
study = float(input("Study Hours: "))
attendance = float(input("Attendance (%): "))
marks = float(input("Previous Marks: "))

prediction = model.predict([[study, attendance, marks]])

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
