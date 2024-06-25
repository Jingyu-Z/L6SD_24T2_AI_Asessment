import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import  GradientBoostingRegressor
from joblib import load

# Load the dataset
df = pd.read_csv('Net_Worth_Data.csv')

# Preprocessing function
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
    
    # Return the scaled input features, scaled target variable, and scaler objects
    return X_scaled, y_scaled, sc, sc1

# Preprocess the data
X_scaled, y_scaled, sc, sc1 = preprocess_data(df)

# Train the best model (as per your previous code)
best_model = GradientBoostingRegressor()  # Example model, replace with your best model
best_model.fit(X_scaled, y_scaled)

# Load the best model from file (as per your previous code)
loaded_model = load('net_worth.joblib')

# Function to handle prediction
def predict_net_worth():
    # Get user input values from entry fields
    gender = int(gender_var.get())
    age = float(age_var.get())
    annual_salary = float(annual_salary_var.get())
    credit_card_debt = float(credit_card_debt_var.get())
    healthcare_cost = float(healthcare_cost_var.get())
    inherited_amount = float(inherited_amount_var.get())
    stocks = float(stocks_var.get())
    bonds = float(bonds_var.get())
    mutual_funds = float(mutual_funds_var.get())
    etfs = float(etfs_var.get())
    reits = float(reits_var.get())
    
    # Prepare input data as a 2D array
    input_data = [[gender, age, annual_salary, credit_card_debt, healthcare_cost, inherited_amount, stocks, bonds, mutual_funds, etfs, reits]]
    
    # Scale input data using the scaler used during preprocessing
    scaled_input_data = sc.transform(input_data)
    
    # Make prediction using the loaded model
    predicted_net_worth_scaled = loaded_model.predict(scaled_input_data)
    
    # Inverse transform the predicted value to get original scale
    predicted_net_worth = sc1.inverse_transform(predicted_net_worth_scaled.reshape(-1, 1))
    
    # Display the predicted net worth in the result label
    result_label.config(text=f"Predicted Net Worth: ${predicted_net_worth[0][0]:,.2f}")


# Create the main application window
root = tk.Tk()
root.title("Net Worth Prediction")

# Create a frame to hold input fields
input_frame = ttk.Frame(root, padding="20")
input_frame.grid(row=0, column=0)

# Create labels and entry fields for user input
ttk.Label(input_frame, text="Gender (0 for female, 1 for male): ").grid(row=0, column=0, sticky="w")
gender_var = tk.StringVar()
gender_entry = ttk.Entry(input_frame, textvariable=gender_var)
gender_entry.grid(row=0, column=1)

ttk.Label(input_frame, text="Age: ").grid(row=1, column=0, sticky="w")
age_var = tk.StringVar()
age_entry = ttk.Entry(input_frame, textvariable=age_var)
age_entry.grid(row=1, column=1)

ttk.Label(input_frame, text="Annual Salary: ").grid(row=2, column=0, sticky="w")
annual_salary_var = tk.StringVar()
annual_salary_entry = ttk.Entry(input_frame, textvariable=annual_salary_var)
annual_salary_entry.grid(row=2, column=1)

ttk.Label(input_frame, text="Credit Card Debt: ").grid(row=3, column=0, sticky="w")
credit_card_debt_var = tk.StringVar()
credit_card_debt_entry = ttk.Entry(input_frame, textvariable=credit_card_debt_var)
credit_card_debt_entry.grid(row=3, column=1)

ttk.Label(input_frame, text="Healthcare Cost: ").grid(row=4, column=0, sticky="w")
healthcare_cost_var = tk.StringVar()
healthcare_cost_entry = ttk.Entry(input_frame, textvariable=healthcare_cost_var)
healthcare_cost_entry.grid(row=4, column=1)

ttk.Label(input_frame, text="Inherited Amount: ").grid(row=5, column=0, sticky="w")
inherited_amount_var = tk.StringVar()
inherited_amount_entry = ttk.Entry(input_frame, textvariable=inherited_amount_var)
inherited_amount_entry.grid(row=5, column=1)

ttk.Label(input_frame, text="Stocks: ").grid(row=6, column=0, sticky="w")
stocks_var = tk.StringVar()
stocks_entry = ttk.Entry(input_frame, textvariable=stocks_var)
stocks_entry.grid(row=6, column=1)

ttk.Label(input_frame, text="Bonds: ").grid(row=7, column=0, sticky="w")
bonds_var = tk.StringVar()
bonds_entry = ttk.Entry(input_frame, textvariable=bonds_var)
bonds_entry.grid(row=7, column=1)

ttk.Label(input_frame, text="Mutual Funds: ").grid(row=8, column=0, sticky="w")
mutual_funds_var = tk.StringVar()
mutual_funds_entry = ttk.Entry(input_frame, textvariable=mutual_funds_var)
mutual_funds_entry.grid(row=8, column=1)

ttk.Label(input_frame, text="ETFs: ").grid(row=9, column=0, sticky="w")
etfs_var = tk.StringVar()
etfs_entry = ttk.Entry(input_frame, textvariable=etfs_var)
etfs_entry.grid(row=9, column=1)

ttk.Label(input_frame, text="REITs: ").grid(row=10, column=0, sticky="w")
reits_var = tk.StringVar()
reits_entry = ttk.Entry(input_frame, textvariable=reits_var)
reits_entry.grid(row=10, column=1)

# Create a button to trigger prediction
predict_button = ttk.Button(input_frame, text="Predict Net Worth", command=predict_net_worth)
predict_button.grid(row=11, columnspan=2)

# Create a label to display the prediction result
result_label = ttk.Label(input_frame, text="")
result_label.grid(row=12, columnspan=2)

# Start the main event loop
root.mainloop()


