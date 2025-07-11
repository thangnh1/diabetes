import pandas as pd
from sklearn.linear_model import LogisticRegression
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = classification_report(y_test, y_pred)

print(metrics)