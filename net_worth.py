
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge, ElasticNet,RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from joblib import dump
from joblib import load

# Load the dataset
df = pd.read_csv('Net_Worth_Data.csv')

# Display the first 5 rows of the DataFrame
print(df.head())

# Display the last 5 rows of the DataFrame
print(df.tail())

# Get the shape of the DataFrame
shape = df.shape
print(f'The dataset has {shape[0]} rows and {shape[1]} columns.')

# Print a concise summary of the dataset
df.info()

def preprocess_data(data):
    # Drop irrelevant columns from input features (X)
    X = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth'], axis=1)
    
    # Extract the target variable (Y)
    Y = data['Net Worth']
    
    # Scale the input features (X)
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    # Scale the target variable (Y)
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape).ravel()
    
    # Print first few rows of scaled input dataset
    print("Scaled Input Features (X_scaled):")
    print(X_scaled[:5])  # Print first 5 rows
    
    # Print first few rows of scaled output dataset
    print("\nScaled Target Variable (y_scaled):")
    print(y_scaled[:5])  # Print first 5 values
    
    # Return the scaled input features, scaled target variable, and scaler objects
    return X_scaled, y_scaled, sc, sc1

# Call preprocess_data function to get scaled data
X_scaled, y_scaled, sc, sc1 = preprocess_data(df)


#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split (X_scaled,y_scaled, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

# Print the shapes of the resulting splits
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Print the first few rows of each
print("\nFirst few rows of X_train:\n", X_train[:5])
print("First few rows of X_test:\n", X_test[:5])
print("First few rows of y_train:\n", y_train[:5])
print("First few rows of y_test:\n", y_test[:5])


#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split (X_scaled,y_scaled, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

# Print the shapes of the resulting splits
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Print the first few rows of each
print("\nFirst few rows of X_train:\n", X_train[:5])
print("First few rows of X_test:\n", X_test[:5])
print("First few rows of y_train:\n", y_train[:5])
print("First few rows of y_test:\n", y_test[:5])
#model used: Linear Regress, Support Vector Machine, Random Forest Regressor,Gradient Boosting Regressor,Ridge Regression, Elastic Net, Robust Regression, Decision Tree Regressor, Artificial Neural Network,Extra Trees Regressor

# Initialize the model
linear_model = LinearRegression()
svm= SVR()
rf= RandomForestRegressor()
gbr = GradientBoostingRegressor()
ridge= Ridge()
en= ElasticNet()
rr= RANSACRegressor()
dtr= DecisionTreeRegressor()
ann=  MLPRegressor(max_iter=1000)
etr= ExtraTreesRegressor()


#train the models using training set
linear_model.fit (X_train, y_train)
svm.fit(X_train,y_train)
rf.fit (X_train,y_train)
gbr.fit (X_train,y_train)
ridge.fit (X_train,y_train)
en.fit (X_train,y_train)
rr.fit (X_train,y_train)
dtr.fit (X_train,y_train)
ann.fit (X_train,y_train)
etr.fit (X_train,y_train)


#prediction on the validation / test data
linear_model_preds = linear_model.predict(X_test)
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)
gbr_preds = gbr.predict(X_test)
ridge_preds = ridge.predict(X_test)
en_preds = en.predict(X_test)
rr_preds = rr.predict(X_test)
dtr_preds = dtr.predict(X_test)
ann_preds = ann.predict(X_test)
etr_preds = etr.predict(X_test)


#evaluate model performance
#RMSE is a measure of the difference between the prdicted values by the model and the actual values
linear_model_rmse = mean_squared_error(y_test, linear_model_preds, squared= False)
svm_rmse = mean_squared_error(y_test, svm_preds, squared= False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared= False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared= False)
ridge_rmse = mean_squared_error(y_test, ridge_preds, squared= False)
en_rmse = mean_squared_error(y_test, en_preds, squared= False)
rr_rmse = mean_squared_error(y_test, rr_preds, squared= False)
dtr_rmse = mean_squared_error(y_test, dtr_preds, squared= False)
ann_rmse = mean_squared_error(y_test, ann_preds, squared= False)
etr_rmse = mean_squared_error(y_test, etr_preds, squared= False)
#model used: Linear Regress, Support Vector Machine, Random Forest Regressor,Gradient Boosting Regressor,Ridge Regression, Elastic Net, Robust Regression, Decision Tree Regressor, Artificial Neural Network,Extra Trees Regressor

#display the evaluation results
print(f"Linear Regression RMSE : {linear_model_rmse}")
print(f"Support Vector Machine RMSE : {svm_rmse}")
print(f"Randome Forest Regressor : {rf_rmse}")
print(f"Gradient Boosting Regressor : {gbr_rmse}")
print(f"Ridge Regression : {ridge_rmse}")
print (f"Elastic Net : {en_rmse}")
print(f"RANSAC Regressor :{rr_rmse}")
print(f"Decision Tree Regressor : {dtr_rmse}")
print(f"Artificial Neural Network : {ann_rmse}")
print(f"Extra Trees Regressor : {etr_rmse}")

#CHOOSE the best model

model_objects = [linear_model,svm, rf, gbr,ridge,en,rr,dtr,ann,etr]
rmse_value = [linear_model_rmse,svm_rmse,rf_rmse,gbr_rmse,ridge_rmse,en_rmse,rr_rmse,dtr_rmse,ann_rmse,etr_rmse]

best_model_index = rmse_value.index(min(rmse_value))
best_model_object = model_objects[best_model_index]

#visualize the model results
#create a bar chart
models = ['Linear Regress', 'Support Vector Machine', 'Random Forest Regressor','Gradient Boosting Regressor','Ridge Regression', 'Elastic Net', 'RANSACRegressor', 'Decision Tree Regressor', 'Artificial Neural Network','Extra Trees Regressor']

plt.figure(figsize=(10,7))
bars = plt.bar (models, rmse_value, color = ['blue', 'green', 'orange', 'red', 'yellow', 'grey','purple', 'pink','black', 'brown'])

#add rmse values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+ bar.get_width()/2, yval + 0.00001, round(yval, 10), ha= 'center', va='bottom', fontsize = 10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparision')
plt.xticks(rotation = 45) #rotate model names for better visibility
plt.tight_layout()

#display the chart
plt.show()

# Save the model to the file
dump(best_model_object, "net_worth.joblib")

# Load the model from the file
loaded_model = load('net_worth.joblib')

# User input for prediction
gender = int(input("Enter gender (0 for female, 1 for male): "))
age = float(input("Enter age: "))
annual_salary = float(input("Enter annual salary: "))
credit_card_debt = float(input("Enter credit card debt: "))
healthcare_cost = float(input("Enter healthcare cost: "))
inherited_amount = float(input("Enter inherited amount: "))
stocks = float(input("Enter stocks: "))
bonds = float(input("Enter bonds: "))
mutual_funds = float(input("Enter mutual funds: "))
etfs = float(input("Enter ETFs: "))
reits = float(input("Enter REITs: "))

# Prepare input data as a 2D array
input_data = [[gender, age, annual_salary, credit_card_debt, healthcare_cost, inherited_amount, stocks, bonds, mutual_funds, etfs, reits]]

# Transform input data using the scaler
scaled_input_data = sc.transform(input_data)

# Make prediction using the loaded model
predicted_net_worth = loaded_model.predict(scaled_input_data)

# Inverse transform the predicted value to get original scale
predicted_net_worth_original_scale = sc1.inverse_transform(predicted_net_worth.reshape(-1, 1))

print("Predicted Net Worth based on input:", predicted_net_worth_original_scale)

