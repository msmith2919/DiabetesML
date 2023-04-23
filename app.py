import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

# Load the diabetes dataset
diabetes_df = pd.read_csv('https://raw.githubusercontent.com/msmith2919/DiabetesDataset/main/diabetes.csv')

# Extract glucose and diabetes outcome as features
features = ['Glucose', 'Outcome']
data = diabetes_df[features]

# Set the feature names explicitly
data.columns = ['glucose', 'outcome']

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(train_data[['glucose']], train_data['outcome'])

# Initialize the Flask application
app = Flask(__name__)

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the result route
@app.route('/result', methods=['POST'])
def result():
    glucose_values = [int(request.form['glucose1']), int(request.form['glucose2']),
                      int(request.form['glucose3']), int(request.form['glucose4'])]
    glucose_array = pd.Series(glucose_values).values.reshape(-1, 1)  # Reshape the input to have one column
    prediction = model.predict(glucose_array)[0]
    confidence = model.predict_proba(glucose_array)[0][prediction]
    if prediction == 1:
        result_text = 'The user is predicted to have diabetes with a confidence of {:.2f}%'.format(confidence * 100)
    else:
        result_text = 'The user is predicted to not have diabetes with a confidence of {:.2f}%'.format((1 - confidence) * 100)
    return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
