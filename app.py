from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import numpy as np
import pickle
import logging
import os
import requests
import sqlite3

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# importing model
model = pickle.load(open(os.path.join(base_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(base_dir, 'standscaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(base_dir, 'minmaxscaler.pkl'), 'rb'))

# flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Database setup
def init_db():
    conn = sqlite3.connect(os.path.join(base_dir, 'users.db'))
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(os.path.join(base_dir, 'users.db'))
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(os.path.join(base_dir, 'users.db'))
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return 'Username already exists'
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/crop_prediction')
def crop_prediction():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route("/get_weather")
def get_weather():
    city = request.args.get('city')
    api_key = 'd67d47efdad8ad9dffd9f9430e9ffdc5'  # Replace with your OpenWeatherMap API key
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(weather_url)
    weather_data = response.json()

    if weather_data['cod'] == 200:
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        return jsonify(success=True, temp=temp, humidity=humidity)
    else:
        return jsonify(success=False)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        city = request.form['City']
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])

        logging.debug(f"Received input: N={N}, P={P}, K={K}, temp={temp}, humidity={humidity}, ph={ph}, rainfall={rainfall}")

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        logging.debug(f"Prediction: {prediction}")

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
            crop_image = f"{crop.lower()}.jpg"  # Assuming the image filenames are in lowercase and match the crop names
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            crop_image = "default.jpg"  # Default image if no crop is found
            crop = "default"  # Default crop name if no crop is found
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        result = f"An error occurred: {str(e)}"
        crop_image = "default.jpg"  # Default image in case of an error
        crop = "default"  # Default crop name in case of an error
    
    return render_template('index.html', result=result, crop_image=crop_image, crop=crop)

if __name__ == "__main__":
    app.run(debug=True)