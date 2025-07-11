import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('data/diabetes.csv')
# profile = ProfileReport(df, title="Diabetes Profiling Report", explorative=True)
# profile.to_file("output/diabetes_report.html")

# Split
target = 'Outcome'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Processing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Train model
model_svm = SVC()
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
metrics_svm = classification_report(y_test, y_pred_svm)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
metrics_lr = classification_report(y_test, y_pred_lr)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
metrics_rf = classification_report(y_test, y_pred_rf)

print("SVM: ", metrics_svm)
print("LR: ", metrics_lr)
print("RF: ", metrics_rf)