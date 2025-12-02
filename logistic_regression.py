import numpy as np 
import pandas as pd 



np.random.seed(42)
feature_1 = np.random.rand(100) * 10
feature_2 = np.random.rand(100) * 5
label = np.random.randint(0, 2, 100)

data = pd.DataFrame({
    'feature_1': feature_1,
    'feature_2': feature_2,
    'label': label
})

X_train = data[['feature_1', 'feature_2']].values
y_train = data['label']

m ,n = X_train.shape

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, w, b):
    cost_sum = 0

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        cost_sum += -y[i] * np.log(g) - (1 - y[i]) * np.log(1 - g) # - y_true * np.log(y_pred) - (1 - y_true) * np.log(1- y_pred)

    return (1/m) * cost_sum

def gradient_function(X, y, w, b):
    grad_w = np.zeros(n)
    grad_b = 0

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        grad_b += (g - y[i])
        for j in range(n):
            grad_w[j] += (g - y[i]) * X[i, j]

    grad_b = (1/m) * grad_b
    grad_w = (1/m) * grad_w

    return grad_b, grad_w

def gradient_descent(X, y, alpha, iterations):
    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        grad_b, grad_w = gradient_function(X, y, w, b)

        w = w - alpha * grad_w
        b = b - alpha * grad_b

        if i % 1000 == 0:
            print(f"Iteration {i}: Cost {cost_function(X, y, w, b)}")
    
    return w, b

def predict(X, w, b):
    preds = np.zeros(m)

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        preds[i] = 1 if g >= 0.5 else 0

    return preds

if __name__ == "__main__":
    learning_rate = 0.01
    iterations = 10000

    final_w, final_b = gradient_descent(X_train, y_train, learning_rate, iterations)

    predictions = predict(X_train, final_w, final_b)
    accuracy = np.mean(predictions == y_train) * 100
    print(f"training accuracy: {accuracy:.2f}%")