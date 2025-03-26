from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import numpy as np
import pickle
import logging
import os
import sqlite3
import re
import requests
from datetime import datetime
from functools import wraps

logging.basicConfig(level=logging.DEBUG)

base_dir = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(base_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(base_dir, 'standscaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(base_dir, 'minmaxscaler.pkl'), 'rb'))

# app
app = Flask(__name__, template_folder=base_dir)
app.secret_key = 'your_secret_key'

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

def get_db_connection():
    conn = sqlite3.connect(os.path.join(base_dir, 'users.db'), timeout=30)  # Increase timeout to 30 seconds
    conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode
    return conn

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/reset_db')
@login_required
def reset_db():
    conn = sqlite3.connect(os.path.join(base_dir, 'users.db'))
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS users')
    conn.commit()
    conn.close()
    init_db()
    return 'Database reset and reinitialized'

@app.route('/')
def crop_pred():
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        if user:
            cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
            user = cursor.fetchone()
            conn.close()
            if user:
                session['username'] = username
                return jsonify(success=True)
            else:
                return jsonify(success=False, message='Incorrect password')
        else:
            conn.close()
            return jsonify(success=False, message='User not found, please register')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
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
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/crop_prediction')
@login_required
def crop_prediction():
    return render_template('crop_pred.html', username=session['username'])

@app.route("/predict", methods=['POST'])
def predict():
    try:
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
            crop_image = f"{crop.lower()}.jpg"  
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            crop_image = "default.jpg"  
            crop = "default"  
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        result = f"An error occurred: {str(e)}"
        crop_image = "default.jpg"  
        crop = "default"  
    
    return render_template('crop_pred.html', result=result, crop_image=crop_image, crop=crop)

@app.route('/crop_info/<crop>')
@login_required
def crop_info(crop):
    crop_details = {
        "rice": {
            "title": "Rice Cultivation",
            "details": [
                {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like IR-64, BPT-5204, or hybrid varieties based on climate. 2. ➢Seed rate: 40-50 kg per hectare (for transplanted) and 80-100 kg per hectare (for direct seeding)."},
                {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Main nitrogen source. 2. ➢Diammonium Phosphate (DAP) (18% N, 46% P₂O₅) – Provides nitrogen and phosphorus. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Potassium-rich fertilizer. 4. ➢ Apply farmyard manure (10-15 tons/ha) before planting."},
                {"heading": "🌱 Proper Plantation", "content": "1. ➢Transplanting Method: Nursery raised, seedlings transplanted at 2-3 leaf stage. 20 x 15 cm spacing between plants. 2. ➢Direct Seeding Method: Seeds are directly sown in wet or dry conditions."},
                {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when 80-90% of grains turn golden yellow. 2. ➢Use a sickle or combine harvester. 3. ➢Dry grains to 12-14% moisture before storage."},
                {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires standing water (5-7 cm) during vegetative growth. 2. <b>Weed Control:</b> Apply herbicides like Butachlor or use mechanical weeding. 3. <b>Pest Control:</b> Monitor for stem borers, leafhoppers, and brown planthoppers."},
                {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (June-July planting, harvested in October-November). 2. <b>Rabi season</b> (November-December planting, harvested in April-May in some regions)."},
                {"heading": "🚜 When to Cut the Crop", "content": "1. ➢Around 120-150 days after planting, ➢Depending on the variety."}
            ]
        },
        "maize": {
            "title": "Maize Cultivation",
            "details": [
                {"heading": "🌾 Seeds Required", "content": "1. ➢Hybrid varieties like HQPM-1, PIONEER 30V92, or DKC 9125. 2. ➢Seed rate: 18-25 kg per hectare."},
                {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Essential for nitrogen supply. 2. ➢Diammonium Phosphate (DAP) (18% N, 46% P₂O₅) – Supports root growth. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Boosts grain formation. ➢Apply farmyard manure (10-15 tons/ha) before planting."},
                {"heading": "🌱 Proper Plantation", "content": "1. ➢Sowing depth: 3-5 cm. 2. ➢Row spacing: 60-75 cm; Plant spacing: 18-25 cm. 3. ➢Sown by dibbling or mechanized planters."},
                {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when cobs are fully mature, dry, and kernels hard. 2. ➢Kernel moisture should be 18-20% at harvest and dried to 12-14% for storage. 3. ➢Use mechanical harvesters or handpicking."},
                {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Needs irrigation at critical stages—vegetative, flowering, and grain filling. 2. <b>Weed Control:</b> Use herbicides like Atrazine or manual weeding. 3. <b>Pest Control:</b> Watch for Fall Armyworm, stem borer, and cutworms."},
                {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (June-July sowing, harvested in September-October). 2. <b>Rabi season</b> (October-November sowing, harvested in February-March in some regions)."},
                {"heading": "🚜 When to Cut the Crop", "content": "1. ➢Around 90-120 days after planting, ➢When kernels are fully matured."}
            ]
        },
         "jute": {
    "title": "Jute Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like JRO-524, JRO-7835, and JRO-8432. 2. ➢Seed rate: 5-6 kg per hectare for capsularis and 4-5 kg per hectare for olitorius species."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Supports vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Helps in root and fiber development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances fiber quality. 4. ➢Apply farmyard manure (10 tons/ha) before sowing."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Sow seeds at a depth of 1-2 cm in well-drained, loamy soil. 2. ➢Row spacing: 25-30 cm for better plant growth. 3. ➢Requires warm, humid conditions for better fiber development."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when plants start shedding leaves (100-120 days after sowing). 2. ➢Cut plants at the base and bundle them for retting. 3. ➢Ret fibers in water for 2-3 weeks for separation."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires frequent irrigation in dry areas. 2. <b>Weed Control:</b> Use pre-emergence herbicides like Fluchloralin. 3. <b>Pest Control:</b> Monitor for jute semilooper and stem weevil."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Summer season</b> (March-May sowing, harvested in June-September)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢Around 100-120 days after planting, ➢Before flowering for better fiber quality."}
    ]
},
"cotton": {
    "title": "Cotton Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Bt Cotton (BG-II), Hybrid Cotton (H-4, H-6), or local varieties (Suvin, MCU-5). 2. ➢Seed rate: 2-3 kg per hectare for hybrids, 12-15 kg per hectare for non-hybrids."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Provides nitrogen for vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root and flower development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves fiber quality. 4. ➢Apply farmyard manure (10-15 tons/ha) before planting for better soil health."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Sow seeds at a depth of 2-3 cm in well-prepared loamy soil. 2. ➢Row spacing: 60-90 cm depending on variety. 3. ➢Use ridge and furrow method to ensure proper drainage."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when cotton bolls fully open (about 150-180 days after sowing). 2. ➢Pick cotton manually or use mechanical pickers. 3. ➢Store in dry conditions to maintain fiber quality."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires 6-7 irrigations, critical at flowering and boll formation. 2. <b>Weed Control:</b> Use pre-emergence herbicides like Pendimethalin. 3. <b>Pest Control:</b> Monitor for bollworms, aphids, and whiteflies."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (April-May sowing, harvested in October-January)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢Around 150-180 days after planting, ➢Depending on variety and climate."}
    ]
},
"coconut": {
    "title": "Coconut Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Dwarf Orange, Tall (West Coast Tall, East Coast Tall), and hybrid varieties (Chowghat Orange Dwarf × West Coast Tall). 2. ➢Seed rate: 175-200 nuts per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Provides nitrogen for growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root and nut formation. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves nut size and quality. 4. ➢Apply farmyard manure (25-30 kg per tree annually) for better growth."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Plant in well-drained sandy loam or alluvial soil. 2. ➢Spacing: 7.5 x 7.5 m for tall varieties, 6.5 x 6.5 m for dwarf varieties. 3. ➢Use pit size of 1m x 1m x 1m and fill with compost before planting."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when coconuts turn brown and have a hard shell (10-12 months after pollination). 2. ➢Use skilled climbers or mechanical coconut harvesters. 3. ➢Harvest tender coconuts for water at 6-7 months old."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires 30-40 liters per tree per day during dry season. 2. <b>Weed Control:</b> Mulch with coconut husk or organic matter. 3. <b>Pest Control:</b> Monitor for rhinoceros beetle, red palm weevil, and leaf-eating caterpillars."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting season:</b> June-July (monsoon planting) or September-October (post-monsoon planting)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢Around 10-12 months after pollination, ➢Tender nuts at 6-7 months."}
    ]
},
"papaya": {
    "title": "Papaya Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Red Lady, Pusa Dwarf, and Solo. 2. ➢Seed rate: 250-300 g per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Supports vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Improves root and fruit development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances fruit quality. 4. ➢Apply farmyard manure (10-15 kg per plant) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Plant in well-drained sandy loam soil. 2. ➢Spacing: 1.8 x 1.8 m. 3. ➢Use pits of 50 x 50 x 50 cm filled with compost."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when fruits turn yellow (5-6 months after flowering). 2. ➢Use a sharp knife to cut fruits carefully. 3. ➢Store at 10-13°C to extend shelf life."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Regular irrigation, especially in dry seasons. 2. <b>Weed Control:</b> Manual weeding or mulching. 3. <b>Pest Control:</b> Protect from aphids, mealybugs, and papaya fruit fly."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> June-September or February-March."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢5-6 months after flowering."}
    ]
},
"orange": {
    "title": "Orange Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Nagpur Orange, Kinnow, and Valencia. 2. ➢Seed rate: 250-300 g per hectare (for seedlings)."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Promotes vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves fruit size and taste. 4. ➢Apply farmyard manure (15-20 kg per tree) annually."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Plant in well-drained sandy loam soil. 2. ➢Spacing: 6 x 6 m. 3. ➢Use pits of 1m x 1m x 1m filled with compost."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when fruits turn deep orange (7-8 months after flowering). 2. ➢Handpick or use clippers to avoid damage. 3. ➢Store at 5-8°C for longer shelf life."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires deep irrigation during dry periods. 2. <b>Weed Control:</b> Use organic mulching. 3. <b>Pest Control:</b> Protect from citrus psylla and fruit flies."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> July-August or February-March."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢7-8 months after flowering."}
    ]
},
"apple": {
    "title": "Apple Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Red Delicious, Golden Delicious, and Fuji. 2. ➢Propagation is done using grafted saplings."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Essential for growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Improves root system. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances fruit color and quality. 4. ➢Apply farmyard manure (20-25 kg per tree) annually."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Plant in well-drained loamy soil in hilly regions. 2. ➢Spacing: 4 x 4 m. 3. ➢Use pits of 1m x 1m x 1m filled with compost."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when fruits reach full color (120-150 days after flowering). 2. ➢Use hand-picking or mechanical harvesters. 3. ➢Store at 0-4°C to extend shelf life."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires frequent irrigation in summer. 2. <b>Weed Control:</b> Use mulching to retain soil moisture. 3. <b>Pest Control:</b> Protect from apple scab and codling moth."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting season:</b> December-February (winter dormancy)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢120-150 days after flowering."}
    ]
},
"muskmelon": {
    "title": "Muskmelon Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Hara Madhu, Punjab Hybrid, and Arka Jeet. 2. ➢Seed rate: 1.5-2 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Essential for vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances fruit development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves fruit sweetness. 4. ➢Apply farmyard manure (15-20 tons/ha) before planting."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> February-April."}
    ]
},
"watermelon": {
    "title": "Watermelon Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Sugar Baby, Arka Jyoti, and Kiran. 2. ➢Seed rate: 3-5 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Supports vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Improves fruit set. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances fruit sweetness. 4. ➢Apply farmyard manure (15-20 tons/ha) before planting."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> January-March (summer crop)."}
    ]
},
"lentil": {
    "title": "Lentil Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like L-4076, Pusa Vaibhav, and IPL-316. 2. ➢Seed rate: 30-40 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Boosts nitrogen supply. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root growth. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves seed formation. 4. ➢Apply farmyard manure (5-10 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Plant in well-drained loamy soil. 2. ➢Spacing: 30 x 10 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when pods turn yellow and dry. 2. ➢Use sickles or threshing machines. 3. ➢Dry seeds to 10-12% moisture before storage."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires 2-3 irrigations at flowering and pod filling. 2. <b>Weed Control:</b> Use hand weeding or pre-emergence herbicides. 3. <b>Pest Control:</b> Protect from aphids and pod borers."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Rabi season</b> (October-November planting, harvested in February-March)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢110-120 days after planting."}
    ]
},
"blackgram": {
    "title": "Black Gram (Urad) Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use varieties like Pant U-19, T-9, or PDU-1. 2. ➢Seed rate: 20-25 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Provides nitrogen. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances early root growth. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves seed setting. 4. ➢Apply farmyard manure (5-10 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Grows best in sandy loam soil with good drainage. 2. ➢Spacing: 30 x 10 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when pods turn black and dry. 2. ➢Cut plants and sun-dry before threshing."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires irrigation at flowering and pod filling stages. 2. <b>Weed Control:</b> Hand weeding at 20-25 days after sowing. 3. <b>Pest Control:</b> Monitor for whiteflies and pod borers."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (June-July planting, harvested in October-November)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢80-90 days after planting."}
    ]
},
"mungbean": {
    "title": "Mung Bean (Green Gram) Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use varieties like PDM-139, K-851, and SML-668. 2. ➢Seed rate: 15-20 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Needed in small amounts. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Strengthens roots and pod formation. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances seed quality. 4. ➢Apply farmyard manure (5-10 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Plant in well-drained sandy loam soil. 2. ➢Spacing: 30 x 10 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when 80% of pods turn yellow. 2. ➢Cut plants and dry before threshing."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires 1-2 irrigations, especially at flowering. 2. <b>Weed Control:</b> Use pre-emergence herbicides or manual weeding. 3. <b>Pest Control:</b> Protect from thrips and aphids."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (June-July planting, harvested in September-October)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢60-70 days after planting."}
    ]
},
"mothbean": {
    "title": "Moth Bean Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like RMO-225, RMO-257, and Jadia. 2. ➢Seed rate: 8-10 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Provides nitrogen. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Strengthens root development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves grain filling. 4. ➢Apply farmyard manure (5-8 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Grows best in sandy and light-textured soils. 2. ➢Spacing: 30 x 10 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when pods turn brown and dry. 2. ➢Cut plants and sun-dry before threshing."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires minimal irrigation, mostly rainfed. 2. <b>Weed Control:</b> Weeding at 20-25 days after sowing. 3. <b>Pest Control:</b> Protect from leafhoppers and pod borers."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (June-July planting, harvested in September-October)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢70-80 days after planting."}
    ]
},
"pigeonpeas": {
    "title": "Pigeon Pea (Tur/Arhar) Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like ICPL-87119, Pusa-9, and UPAS-120. 2. ➢Seed rate: 12-15 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Improves vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root growth. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Improves grain development. 4. ➢Apply farmyard manure (10-15 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Prefers well-drained loamy soil with pH 6.5-7.5. 2. ➢Spacing: 75 x 30 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when pods turn brown and dry. 2. ➢Cut plants and sun-dry before threshing."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires irrigation at flowering and pod filling. 2. <b>Weed Control:</b> Use pre-emergence herbicides or manual weeding. 3. <b>Pest Control:</b> Protect from pod borers and aphids."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Kharif season</b> (June-July planting, harvested in December-January)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢150-180 days after planting."}
    ]
},
"kidneybeans": {
    "title": "Kidney Bean (Rajma) Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use varieties like VL Rajma-125, HUR-15, and PDR-14. 2. ➢Seed rate: 75-100 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Provides nitrogen. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Strengthens root growth. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances grain development. 4. ➢Apply farmyard manure (10-15 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Prefers cool climate and loamy soil. 2. ➢Spacing: 45 x 10 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when pods turn yellow and dry. 2. ➢Cut plants and sun-dry before threshing."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires irrigation at flowering and pod filling stages. 2. <b>Weed Control:</b> Use mechanical weeding or herbicides. 3. <b>Pest Control:</b> Protect from aphids and bean flies."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Rabi season</b> (October-November planting, harvested in February-March)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢100-120 days after planting."}
    ]
},
"chickpea": {
    "title": "Chickpea (Gram) Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use varieties like Pusa-256, BG-372, and JG-11. 2. ➢Seed rate: 60-80 kg per hectare."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Needed in minimal quantity. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Improves root and pod development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances grain quality. 4. ➢Apply farmyard manure (8-10 tons/ha) before planting."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Prefers sandy loam soil with good drainage. 2. ➢Spacing: 30 x 10 cm between plants."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when pods turn yellow and dry. 2. ➢Use sickles or threshing machines."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires minimal irrigation, only at pod formation stage. 2. <b>Weed Control:</b> Use pre-emergence herbicides or manual weeding. 3. <b>Pest Control:</b> Protect from pod borers and aphids."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Rabi season</b> (October-November planting, harvested in February-March)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢120-130 days after planting."}
    ]
},
"coffee": {
    "title": "Coffee Cultivation",
    "details": [
        {"heading": "🌾 Seeds Required", "content": "1. ➢Use high-yielding varieties like Thompson Seedless, Bangalore Blue, Anab-e-Shahi, and Sonaka. 2. ➢Propagation is done using stem cuttings or grafting rather than seeds."},
        {"heading": "💊 Fertilizer Requirements", "content": "1. ➢Urea (46% N) – Supports vegetative growth. 2. ➢Single Super Phosphate (SSP) (16% P₂O₅) – Improves root and fruit development. 3. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances fruit size and quality. 4. ➢Muriate of Potash (MOP) (60% K₂O) – Enhances fruit size and quality.."},
        {"heading": "🌱 Proper Plantation", "content": "1. ➢Grows best in well-drained, shaded areas with loamy soil. 2. ➢Spacing: 2.5 x 2.5 m for Arabica and 3 x 3 m for Robusta."},
        {"heading": "👨‍🌾 Proper Methodology to Harvest", "content": "1. ➢Harvest when coffee cherries turn bright red. 2. ➢Pick manually for better quality. 3. ➢Process using dry or wet methods."},
        {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires irrigation during dry spells. 2. <b>Weed Control:</b> Use mulching and manual weeding. 3. <b>Pest Control:</b> Monitor for coffee berry borer and white stem borer."},
        {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting season</b> (June-August for Arabica, May-June for Robusta)."},
        {"heading": "🚜 When to Cut the Crop", "content": "1. ➢Cherries mature in 6-8 months after flowering."}
    ]
},  

    "grapes": {
        "title": "Grape Cultivation",
        "details": [
            {"heading": "🌾 Propagation Method", "content": "1. ➞Propagation through cuttings or grafting. 2. ➞Use hardwood cuttings or tissue culture plants for better yield."},
            {"heading": "💊 Fertilizer Requirements", "content": "1. ➞Urea (46% N) – Supports vegetative growth. 2. ➞Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root and fruit development. 3. ➞Muriate of Potash (MOP) (60% K₂O) – Improves fruit size and sweetness. 4. ➞Apply farmyard manure (10-15 tons/ha) before planting."},
            {"heading": "🌱 Proper Plantation", "content": "1. ➞Plant in well-drained sandy loam soil with good sunlight. 2. ➞Spacing: 3 x 3 m for proper canopy development. 3. ➞Use trellis or pergola system for vine training."},
            {"heading": "🧑‍🌾 Proper Methodology to Harvest", "content": "1. ➞Harvest when grapes reach full color and desired sugar content (120-150 days after pruning). 2. ➞Handpick carefully to avoid damage. 3. ➞Store at 0-2°C to extend shelf life."},
            {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Regular irrigation, especially during fruit set and ripening. 2. <b>Weed Control:</b> Manual weeding or mulching. 3. <b>Pest Control:</b> Protect from mealybugs, thrips, and powdery mildew."},
            {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> January-February or June-July."},
            {"heading": "🚜 When to Cut the Crop", "content": "1. ➞Around 120-150 days after pruning, when sugar content is optimal."}
        ]
    },
    "banana": {
        "title": "Banana Cultivation",
        "details": [
            {"heading": "🌾 Propagation Method", "content": "1. ➞Propagation through suckers or tissue culture plants. 2. ➞Use disease-free plantlets for high yield."},
            {"heading": "💊 Fertilizer Requirements", "content": "1. ➞Urea (46% N) – Supports vegetative growth. 2. ➞Single Super Phosphate (SSP) (16% P₂O₅) – Improves root and fruit development. 3. ➞Muriate of Potash (MOP) (60% K₂O) – Enhances fruit size and quality. 4. ➞Apply farmyard manure (10-15 kg per plant) before planting."},
            {"heading": "🌱 Proper Plantation", "content": "1. ➞Plant in well-drained loamy soil with good moisture retention. 2. ➞Spacing: 1.5 x 1.5 m for optimal growth. 3. ➞Plant in pits of 50 x 50 x 50 cm filled with compost."},
            {"heading": "🧑‍🌾 Proper Methodology to Harvest", "content": "1. ➞Harvest when bananas reach full size and turn light green (9-12 months after planting). 2. ➞Cut bunches carefully using a sharp knife. 3. ➞Store at 13-15°C to extend shelf life."},
            {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Requires frequent irrigation for continuous growth. 2. <b>Weed Control:</b> Manual weeding or mulching. 3. <b>Pest Control:</b> Monitor for banana weevils and nematodes."},
            {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> June-July or October-November."},
            {"heading": "🚜 When to Cut the Crop", "content": "1. ➞Around 9-12 months after planting, when fingers are well-formed."}
        ]
    },
    "mango": {
        "title": "Mango Cultivation",
        "details": [
            {"heading": "🌾 Propagation Method", "content": "1. ➞Propagation through grafting or air-layering. 2. ➞Use high-yielding varieties like Alphonso, Kesar, or Dasheri."},
            {"heading": "💊 Fertilizer Requirements", "content": "1. ➞Urea (46% N) – Supports vegetative growth. 2. ➞Single Super Phosphate (SSP) (16% P₂O₅) – Enhances root and flower development. 3. ➞Muriate of Potash (MOP) (60% K₂O) – Improves fruit quality. 4. ➞Apply farmyard manure (25-30 kg per tree annually)."},
            {"heading": "🌱 Proper Plantation", "content": "1. ➞Plant in well-drained loamy soil with proper sunlight. 2. ➞Spacing: 8 x 8 m for good canopy growth. 3. ➞Use pits of 1m x 1m x 1m filled with compost before planting."},
            {"heading": "🧑‍🌾 Proper Methodology to Harvest", "content": "1. ➞Harvest when fruits develop full color and aroma (100-150 days after flowering). 2. ➞Handpick or use clippers to prevent damage. 3. ➞Store at 12-15°C to extend shelf life."},
            {"heading": "🌿 Proper Care", "content": "1. <b>Watering:</b> Deep irrigation required during flowering and fruiting. 2. <b>Weed Control:</b> Use organic mulching. 3. <b>Pest Control:</b> Protect from mango hoppers and fruit flies."},
            {"heading": "📅 Best Time to Grow", "content": "1. <b>Planting seasons:</b> June-July or February-March."},
            {"heading": "🚜 When to Cut the Crop", "content": "1. ➞Around 100-150 days after flowering, when fully matured."}
        ]
    }
}
    if crop.lower() in crop_details:
        crop_info = crop_details[crop.lower()]
        for detail in crop_info['details']:
            detail['content'] = re.split(r'\d+\.\s', detail['content'])[1:]  
        return render_template('crop_info.html', crop=crop_info, crop_image=f"{crop.lower()}.jpg")
    else:
        return "Crop information not available", 404

@app.route('/weather', methods=['GET', 'POST'])
@login_required
def weather():
    if request.method == 'POST':
        city = request.form['city']
        geocode_url = f'https://geocoding-api.open-meteo.com/v1/search?name={city}'
        geocode_response = requests.get(geocode_url)
        geocode_data = geocode_response.json()

        if geocode_response.status_code == 200 and 'results' in geocode_data:
            latitude = geocode_data['results'][0]['latitude']
            longitude = geocode_data['results'][0]['longitude']

            weather_url = f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode&timezone=auto&days=14'
            response = requests.get(weather_url)
            data = response.json()

            weather_codes = {
                0: "Clear sky ☀️",
                1: "Mainly clear 🌤️",
                2: "Partly cloudy ⛅",
                3: "Overcast ☁️",
                45: "Fog 🌫️",
                48: "Depositing rime fog 🌫️",
                51: "Light drizzle 🌦️",
                53: "Moderate drizzle 🌦️",
                55: "Dense drizzle 🌧️",
                56: "Light freezing drizzle 🌧️",
                57: "Dense freezing drizzle 🌧️",
                61: "Slight rain 🌧️",
                63: "Moderate rain 🌧️",
                65: "Heavy rain 🌧️",
                66: "Light freezing rain 🌧️",
                67: "Heavy freezing rain 🌧️",
                71: "Slight snow fall 🌨️",
                73: "Moderate snow fall 🌨️",
                75: "Heavy snow fall 🌨️",
                77: "Snow grains 🌨️",
                80: "Slight rain showers 🌧️",
                81: "Moderate rain showers 🌧️",
                82: "Violent rain showers 🌧️",
                85: "Slight snow showers 🌨️",
                86: "Heavy snow showers 🌨️",
                95: "Thunderstorm ⛈️",
                96: "Thunderstorm with slight hail ⛈️",
                99: "Thunderstorm with heavy hail ⛈️"
            }

            if response.status_code == 200:
                weather_data = []
                for i in range(len(data['daily']['time'])):
                    weather_data.append({
                        'date': data['daily']['time'][i],
                        'min_temp': data['daily']['temperature_2m_min'][i],
                        'max_temp': data['daily']['temperature_2m_max'][i],
                        'rainfall': data['daily']['precipitation_sum'][i],
                        'description': weather_codes.get(data['daily']['weathercode'][i], "Unknown")
                    })
                return render_template('weather_report.html', weather_data=weather_data, city=city)
            else:
                error_message = data.get('reason', 'An error occurred while fetching the weather data.')
                return render_template('weather_report.html', error_message=error_message)
        else:
            error_message = 'An error occurred while fetching the geocode data.'
            return render_template('weather_report.html', error_message=error_message)
    return render_template('weather_report.html')

if __name__ == "__main__":
    app.run(debug=True)