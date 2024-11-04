from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


app = Flask(__name__)

# Ensure the 'static' directory exists for saving plots
if not os.path.exists('static'):
    os.makedirs('static')

def generate_plots(N, mu, sigma2, S):
    # Generate the first dataset
    X = np.random.rand(N, 1) * 10  # Random X values between 0 and 10
    noise = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = 2 + 3 * X.flatten() + noise  # Random Y with noise and no true relationship

    # Fit the linear regression model for the first dataset
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Create scatter plot with regression line
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', label=f'Fitted Line (slope={slope:.2f}, intercept={intercept:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.title('Scatter Plot with Regression Line')
    plt.title(f'Linear Fit: y = {intercept:.2f} + {slope:.2f}x')
    plt.legend()
    plot1_path = 'static/plot1.png'
    plt.savefig(plot1_path)
    plt.close()

    # Simulate S datasets and collect slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        noise_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = 2 + 3 * X.flatten() + noise_sim
        model.fit(X, Y_sim)
        slopes.append(model.coef_[0])
        intercepts.append(model.intercept_)

    # Create histograms of slopes and intercepts
    plt.figure(figsize=(10, 6))
    plt.hist(slopes, bins=30, alpha=0.6, label='Slopes', color='blue')
    plt.hist(intercepts, bins=30, alpha=0.6, label='Intercepts', color='orange')
    plt.axvline(slope, color='blue', linestyle='--', label=f'Slope: {slope:.2f}')
    plt.axvline(intercept, color='orange', linestyle='--', label=f'Intercept: {intercept:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Slopes and Intercepts')
    plt.legend()
    plot2_path = 'static/plot2.png'
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions for more extreme values
    slope_more_extreme = sum(s > slope for s in slopes) / S  # Already provided
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S  # Already provided

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            N = int(request.form["N"])
            mu = float(request.form["mu"])
            sigma2 = float(request.form["sigma2"])
            S = int(request.form["S"])

            plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

            return render_template("index.html", plot1=plot1, plot2=plot2,
                                   slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)
        except ValueError:
            return "Invalid input. Please enter valid numbers."

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)