import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('winequality-red.csv', delimiter=';')


#print(df.head())


#print(df.isnull().sum())


print(df.describe())



plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df)
plt.title('Şarap Kalitesi Dağılımı')
plt.xlabel('Kalite')
plt.ylabel('Frekans')
plt.show()


features = df.columns[:-1]
plt.figure(figsize=(20, 15))

for i, feature in enumerate(features):
    plt.subplot(4, 3, i+1)
    sns.boxplot(x='quality', y=feature, data=df)
    plt.title(f'Quality vs {feature}')

plt.tight_layout()
plt.show()



plt.figure(figsize=(15, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()



plt.figure(figsize=(20, 20))
sns.pairplot(df, hue='quality', diag_kind='kde', markers='+')
plt.show()



X = df.drop(columns=['quality'])
y = df['quality']
y = (y >= 6).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)




y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)



plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()
