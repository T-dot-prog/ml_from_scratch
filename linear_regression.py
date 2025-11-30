from databank import data_bank

path: str = data_bank.download_from_website()


def cost_function(x, y , w, b):
        """Cost function for linear regression"""
        m = len(x)
        cost_sum = 0

        for i in range(m):
            y_pred = w * x[i] + b
            cost = (y_pred - y[i]) ** 2
            cost_sum += cost

        total_cost = (1/(2 * m)) * cost_sum # (1/2m) * (y-y_pred) **2
        return total_cost
    
def gradient_function(x, y, w, b):
    """Gradient function for Gradient descent"""
    m = len(x)
    dw = db = 0

    for i in range(m):
        y_pred = w * x[i] + b

        dw += (y_pred - y[i]) * x[i]
        db += (y_pred - y[i])

    dw = (1/m) * dw
    db = (1/m) * db

    return dw, db
    
def gradient_descent(x, y, alpha , iterations ):
    """Function to implement gradient descent"""
    w = b = 0

    for i in range(iterations):
        dw , db = gradient_function(x, y, w, b)

        w = w - alpha * dw
        b = b - alpha * db
    
    return w, b

learning_rate: float = 0.01
iteartions: int = 10000

import pandas as pd 
def get_xtrain_ytrain(dataset_path: str) -> pd.Series:
    """Function to get X train and y train values"""
    try:
        training_set = pd.read_csv(f'{dataset_path}/Salary Data.csv')

        x_train = training_set["YearsExperience"].values
        y_train = training_set["Salary"].values

        return x_train, y_train
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    x_train , y_train = get_xtrain_ytrain(dataset_path= path)

    final_w , final_b = gradient_descent(x_train, y_train, learning_rate, iteartions)

    print(f"w: {final_w:.4f}, b: {final_b:.4f}")

    import matplotlib.pyplot as plt
    import numpy as np

    # Plot the scatter plot 
    plt.scatter(x_train, y_train, label='Data Points')

    x_vals = np.linspace(min(x_train), max(x_train), 100)
    y_vals = final_w * x_vals + final_b
    plt.plot(x_vals, y_vals, color='red', label='Regression Line')

    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()