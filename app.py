import streamlit as st
import pandas as pd
import numpy as np

# ตั้งค่าหน้าเว็บให้เป็นแนวกว้าง
st.set_page_config(page_title="ML Project (30 คะแนน)", layout="wide")

# ==========================================
# แถบด้านข้าง (Sidebar) - ข้อมูลผู้พัฒนา
# ==========================================
# ดึงรูปจากโฟลเดอร์ Image ที่คุณเตรียมไว้
try:
    st.sidebar.image("Image/Dev.png", caption="ผู้พัฒนาโมเดล") 
except FileNotFoundError:
    st.sidebar.warning("ไม่พบไฟล์รูปภาพ กรุณาตรวจสอบว่ามีโฟลเดอร์ Image และไฟล์ Dev.png อยู่ในโฟลเดอร์เดียวกันกับโค้ด")

st.sidebar.title("ข้อมูลผู้พัฒนา")
st.sidebar.info(
    "**รหัส:** 664245023\n\n"
    "**ชื่อ-นามสกุล:** นายพรรคพล พูลสวัสดิ์\n\n"
    "**หมู่เรียน:** 66/43"
)
st.sidebar.markdown("---")
# ใส่ลิงก์ GitHub ของคุณ (แก้ไขตรงนี้เมื่ออัปโหลดขึ้น GitHub เสร็จ)
st.sidebar.write("🔗 [คลิกเพื่อดู GitHub ของโปรเจกต์นี้](ใส่ลิงก์ของคุณที่นี่)")

# ==========================================
# เนื้อหาหลัก (Main Content)
# ==========================================
st.title("🚀 นำเสนอโปรเจกต์ Machine Learning (30 คะแนน)")

# แบ่งหน้าจอเป็น 5 Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. ปัญหา & Dataset", 
    "2. Data Preprocessing", 
    "3. โมเดล ML", 
    "4. การประเมินผล", 
    "5. Streamlit App"
])

with tab1:
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.write("**ปัญหาที่ต้องการแก้ไข:** (พิมพ์อธิบายที่นี่)")
    st.write("**เหตุผลที่เลือก Dataset นี้:** (พิมพ์อธิบายที่นี่)")
    
    st.write("ตัวอย่างข้อมูล (Dataset):")
    # ตัวอย่างการสร้างตารางจำลอง (แทนที่ด้วย df.head() เมื่อคุณมีข้อมูลจริง)
    df_sample = pd.DataFrame(np.random.randn(5, 3), columns=('Feature X', 'Feature Y', 'Target'))
    st.dataframe(df_sample)

with tab2:
    st.header("2. Data Preprocessing")
    st.write("ในโปรเจกต์นี้ได้มีการจัดการข้อมูลก่อนนำเข้าโมเดล ดังนี้:")
    st.markdown("""
    * **ทำความสะอาดข้อมูล (Cleaning):** ...
    * **แปลงข้อมูล (Encoding):** ...
    * **ปรับสเกลข้อมูล (Scaling):** ...
    """)

with tab3:
    st.header("3. การสร้างโมเดล ML")
    st.subheader("ทฤษฎีโมเดลที่ใช้")
    st.write("อธิบายการทำงานของโมเดลที่คุณเลือก (เช่น Random Forest, SVM) อย่างสังเขป...")

with tab4:
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**ตารางเปรียบเทียบประสิทธิภาพ:**")
        results = pd.DataFrame({
            'Model': ['Model 1', 'Model 2', 'Model 3'],
            'Accuracy': [92.5, 88.4, 81.2]
        })
        st.table(results)
    
    with col2:
        st.write("**กราฟเปรียบเทียบ Accuracy:**")
        chart_data = pd.DataFrame({'Accuracy': [92.5, 88.4, 81.2]}, index=['Model 1', 'Model 2', 'Model 3'])
        st.bar_chart(chart_data)

with tab5:
    st.header("5. Streamlit Application")
    st.write("กรอกข้อมูลด้านล่างเพื่อทดสอบการทำนายของโมเดล")
    
    # ฟอร์มรับค่าเพื่อทำนาย (จำลอง)
    c1, c2 = st.columns(2)
    with c1:
        val1 = st.slider("เลือกค่า Feature X", 0, 100, 50)
    with c2:
        val2 = st.number_input("กรอกค่า Feature Y", value=10.0)
        
    st.write("") 
    if st.button("ประมวลผล (Predict)", type="primary"):
        # ในการใช้งานจริง ให้นำโค้ดโหลดโมเดล (.pkl) มาไว้ตรงนี้
        st.success(f"🎉 ผลลัพธ์: โมเดลทำนายได้ค่าสอดคล้องกับ {val1} และ {val2}")