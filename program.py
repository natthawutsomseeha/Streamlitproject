import streamlit as st
import numpy as np
import joblib

# ตั้งค่าธีม และเพิ่มพื้นหลัง
st.set_page_config(page_title="Cardiovascular Risk Assessment", page_icon="❤️", layout="centered")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนหัวของแอป
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>Cardiovascular Disease Prediction</h1>
    <p style='text-align: center; color: #555;'>กรอกข้อมูลสุขภาพของคุณเพื่อประเมินความเสี่ยงของโรคหัวใจ</p>
    """, unsafe_allow_html=True
)

# ใช้คอลัมน์เพื่อจัดระเบียบ UI
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🕒 อายุ (ปี)", 29, 64, 40)
    gender = st.radio("🚻 เพศ", ["ชาย", "หญิง"], horizontal=True)
    height = st.number_input("📏 ส่วนสูง (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("⚖️ น้ำหนัก (kg)", min_value=30, max_value=200, value=70)

with col2:
    ap_hi = st.number_input("🩸 ความดันโลหิต Systolic", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("🩸 ความดันโลหิต Diastolic", min_value=30, max_value=200, value=80)

    # คั่นกลางเพื่อควยสมมาตร
    st.markdown("")

    cholesterol = st.selectbox("🧪 ระดับคอเลสเตอรอล", [1, 2, 3], format_func=lambda x: ["ปกติ", "สูง", "สูงมาก"][x-1])
    gluc = st.selectbox("🍬 ระดับน้ำตาลในเลือด", [1, 2, 3], format_func=lambda x: ["ปกติ", "สูง", "สูงมาก"][x-1])

# ตัวเลือก Lifestyle
st.markdown("### 🏃‍♂️ พฤติกรรมสุขภาพ")
col3, col4, col5 = st.columns(3)

with col3:
    smoke = st.checkbox("🚬 สูบบุหรี่")
with col4:
    alco = st.checkbox("🍷 ดื่มแอลกอฮอล์")
with col5:
    active = st.checkbox("💪 ออกกำลังกาย")

# แปลงข้อมูลให้เป็นตัวเลข
gender = 1 if gender == "ชาย" else 2
smoke = int(smoke)
alco = int(alco)
active = int(active)
input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

# โหลดโมเดลที่ฝึกไว้
model = joblib.load("voting_classifier.pkl")

# ทำนายผลลัพธ์
if st.button("🔍 ทำนายผลลัพธ์", use_container_width=True):
    prediction = model.predict(input_data)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ คุณมีความเสี่ยงต่อโรคหัวใจ โปรดปรึกษาแพทย์")
    else:
        st.success("✅ คุณมีสุขภาพดี ไม่มีความเสี่ยงต่อโรคหัวใจ")
