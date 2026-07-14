from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask Application
app = Flask(__name__)

# Load the trained model
with open("../Training/HDI_Model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("home.html")


# ==========================
# Prediction Page
# ==========================
@app.route("/prediction")
def prediction():
    return render_template("indexnew.html")


# ==========================
# Predict HDI
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        life_expectancy = float(request.form["life_expectancy"])
        expected_schooling = float(request.form["expected_schooling"])
        mean_schooling = float(request.form["mean_schooling"])
        gni = float(request.form["gni"])

        # Arrange input in the same order used during training
        features = np.array([[life_expectancy,
                              expected_schooling,
                              mean_schooling,
                              gni]])

        # Predict HDI
        prediction = model.predict(features)

        output = round(float(prediction[0]), 3)

        return render_template(
            "indexnew.html",
            prediction_text=f"Predicted Human Development Index (HDI): {output}"
        )

    except Exception as e:
        return render_template(
            "indexnew.html",
            prediction_text=f"Error: {str(e)}"
        )


# ==========================
# Run Flask Application
# ==========================
if __name__ == "__main__":
    app.run(debug=True)