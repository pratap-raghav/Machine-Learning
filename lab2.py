import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_set = pd.read_csv('salary_data.csv')

x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color="blue", label="Actual Salary")
plt.plot(x_train, y_train_pred, color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction (Training Dataset)")
plt.legend()
plt.show()

plt.scatter(x_test, y_test, color="green", label="Actual Salary")
plt.plot(x_train, y_train_pred, color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction (Test Dataset)")
plt.legend()
plt.show()

mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared Score (R2): {r2}")

x_sample = [[0]]
y_sample_pred = regressor.predict(x_sample)
print(f"Predicted Salary for 3.6 years of experience: {y_sample_pred[0]:.2f}")
