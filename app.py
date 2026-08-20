import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML Project", layout="wide")

# ==========================================
# แถบด้านข้าง (Sidebar)
# ==========================================
try:
    st.sidebar.image("Image/Dev.png", caption="ผู้พัฒนาโมเดล") 
except:
    pass # ข้ามไปหากไม่พบรูป

st.sidebar.title("ข้อมูลผู้พัฒนา")
st.sidebar.info(
    "**รหัส:** 664245023\n\n"
    "**ชื่อ-นามสกุล:** นายพรรคพล พูลสวัสดิ์\n\n"
    "**หมู่เรียน:** 66/43"
)
st.sidebar.markdown("---")
st.sidebar.write("🔗 [คลิกเพื่อดู GitHub ของโปรเจกต์นี้](https://github.com/Pakkarpon/ML-Project)")

# ==========================================
# เนื้อหาหลัก (Main Content)
# ==========================================
st.title("🌱 ระบบแนะนำการปลูกพืชอัจฉริยะ (Smart Agriculture)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. ปัญหา & Dataset", "2. Data Preprocessing", "3. โมเดล ML", "4. การประเมินผล", "5. Streamlit App"
])

with tab1:
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.write("**ปัญหาที่ต้องการแก้ไข:** เกษตรกรในปัจจุบันต้องเผชิญกับสภาพอากาศที่เปลี่ยนแปลงบ่อย ทำให้ยากต่อการตัดสินใจเลือกชนิดพืชที่จะปลูกให้เหมาะสมกับสภาพดินและอากาศ ณ เวลานั้น ซึ่งอาจส่งผลให้ผลผลิตตกต่ำ")
    st.write("**เหตุผลที่เลือก Dataset นี้:** เป็นข้อมูลที่ได้จากการจำลองค่าเซนเซอร์พื้นฐานที่เข้าถึงง่าย (อุณหภูมิ, ความชื้นอากาศ, ความชื้นในดิน) ซึ่งสามารถนำไปประยุกต์ต่อยอดสร้างอุปกรณ์ IoT สำหรับฟาร์มอัจฉริยะ (Smart Farm) ได้จริง")
    
    st.write("**ตัวอย่างข้อมูล (Dataset):**")
    try:
        df = pd.read_csv('agriculture_data.csv')
        st.dataframe(df.head(10))
    except:
        st.warning("กำลังรอข้อมูล agriculture_data.csv")

with tab2:
    st.header("2. Data Preprocessing")
    st.write("ขั้นตอนการเตรียมข้อมูลก่อนนำเข้าโมเดล:")
    st.markdown("""
    * **Data Collection & Cleaning:** รวบรวมข้อมูลจากเซนเซอร์ ตรวจสอบและลบข้อมูลที่เป็นค่าว่าง (Missing Values) หรือข้อมูลที่มีค่าผิดปกติ (Outliers) ออก
    * **Feature Selection:** เลือกใช้ Feature ที่ส่งผลต่อการเติบโตของพืชโดยตรง ได้แก่ อุณหภูมิ (Temperature), ความชื้นอากาศ (Humidity), และความชื้นในดิน (Soil Moisture)
    * **Data Splitting:** ทำการแบ่งข้อมูลเป็น Training Set (80%) สำหรับสอนโมเดล และ Testing Set (20%) สำหรับทดสอบความแม่นยำ
    """)

with tab3:
    st.header("3. การสร้างโมเดล ML")
    st.subheader("Random Forest Classifier")
    st.write("โมเดลที่เลือกใช้คือ **Random Forest** ซึ่งเป็นอัลกอริทึมประเภท Ensemble Learning")
    st.write("ทฤษฎีการทำงานคือ โมเดลจะสร้าง Decision Tree (ต้นไม้ตัดสินใจ) ย่อยๆ จำนวนหลายๆ ต้น (ในที่นี้ใช้ 100 ต้น) เพื่อเรียนรู้เงื่อนไขของสภาพแวดล้อมต่างๆ เมื่อมีข้อมูลใหม่เข้ามา ต้นไม้แต่ละต้นจะทำการ 'โหวต' ว่าสภาพแวดล้อมนี้เหมาะกับพืชชนิดใดมากที่สุด และนำผลโหวตที่เยอะที่สุดมาเป็นคำตอบสุดท้าย ซึ่งจะช่วยลดข้อผิดพลาดและให้ความแม่นยำที่สูงกว่าการใช้ Decision Tree เพียงต้นเดียว")

with tab4:
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**ตารางเปรียบเทียบประสิทธิภาพ:**")
        results = pd.DataFrame({
            'Model': ['Random Forest', 'Decision Tree', 'KNN'],
            'Accuracy (%)': [96.5, 89.2, 84.8]
        })
        st.table(results)
    
    with col2:
        st.write("**กราฟเปรียบเทียบ Accuracy:**")
        chart_data = pd.DataFrame({'Accuracy': [96.5, 89.2, 84.8]}, index=['Random Forest', 'Decision Tree', 'KNN'])
        st.bar_chart(chart_data)

with tab5:
    st.header("5. Streamlit Application")
    st.write("ปรับค่าเซนเซอร์จำลองด้านล่าง เพื่อให้ AI แนะนำพืชที่เหมาะสม")
    
    # ฟอร์มรับค่า
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.slider("อุณหภูมิ (°C)", 10.0, 45.0, 25.0)
    with col2:
        humid = st.slider("ความชื้นในอากาศ (%)", 10.0, 100.0, 60.0)
    with col3:
        soil = st.slider("ความชื้นในดิน (%)", 0.0, 100.0, 40.0)
        
    st.write("") 
    if st.button("ประมวลผลคำแนะนำ (Predict)", type="primary"):
        try:
            # โหลดโมเดล
            model = pickle.load(open('crop_model.pkl', 'rb'))
            # ทำนายผล
            prediction = model.predict([[temp, humid, soil]])
            st.success(f"🌱 พืชที่เหมาะสมกับสภาพแวดล้อมนี้คือ: **{prediction[0]}**")
            st.balloons()
        except FileNotFoundError:
            st.error("ไม่พบไฟล์โมเดล crop_model.pkl กรุณารันไฟล์ train_model.py ก่อน")