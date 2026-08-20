import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. สร้าง Dataset จำลองข้อมูลจากเซนเซอร์การเกษตร
np.random.seed(42)
n = 500
data = {
    'Temperature': np.random.uniform(20, 35, n),    # อุณหภูมิ (องศาเซลเซียส)
    'Humidity': np.random.uniform(40, 80, n),       # ความชื้นในอากาศ (%)
    'Soil_Moisture': np.random.uniform(10, 60, n),  # ความชื้นในดิน (%)
    'Target': np.random.choice(['ข้าว', 'ข้าวโพด', 'อ้อย'], n) # พืชที่เหมาะสม
}
df = pd.DataFrame(data)
df.to_csv('agriculture_data.csv', index=False) # เซฟเป็นไฟล์ CSV ส่งอาจารย์

# 2. แยกข้อมูล (Features และ Target)
X = df[['Temperature', 'Humidity', 'Soil_Moisture']]
y = df['Target']

# 3. สร้างและเทรนโมเดล (ใช้ Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. เซฟโมเดลเป็นไฟล์ .pkl
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 สร้างไฟล์ agriculture_data.csv และ crop_model.pkl เสร็จเรียบร้อย!")