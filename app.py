import streamlit as st
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image

# تنظیم صفحه
st.set_page_config(
    page_title="تشخیص گیاه",
    page_icon="🌿",
    layout="centered"
)

# عنوان
st.title("🌿 سیستم تشخیص نوع گیاه با هوش مصنوعی")
st.markdown("---")

# بارگذاری مدل و کلاس‌ها
@st.cache_resource
def load_my_model():
    return load_model('plant_cnn_model.h5')

@st.cache_data
def load_classes():
    with open('classes.json', 'r') as f:
        return json.load(f)

try:
    model = load_my_model()
    classes = load_classes()
    st.success("✅ مدل با موفقیت بارگذاری شد")
except Exception as e:
    st.error(f"❌ خطا در بارگذاری مدل: {e}")
    st.info("لطفاً ابتدا فایل train_model.py را اجرا کنید")
    st.stop()

# بخش آپلود عکس
st.subheader("📤 آپلود عکس گیاه")
uploaded_file = st.file_uploader(
    "عکس برگ گیاه را انتخاب کنید",
    type=["jpg", "jpeg", "png"]
)

# یا استفاده از دوربین
st.subheader("📸 یا گرفتن عکس با دوربین")
camera_image = st.camera_input("از برگ گیاه عکس بگیرید")

# انتخاب منبع تصویر
image_to_predict = None
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption="عکس آپلود شده", use_column_width=True)
elif camera_image is not None:
    image_to_predict = Image.open(camera_image)
    st.image(image_to_predict, caption="عکس گرفته شده", use_column_width=True)

# دکمه پیش‌بینی
if image_to_predict is not None:
    if st.button("🔍 تشخیص گیاه", type="primary"):
        with st.spinner("در حال تحلیل عکس..."):
            # پیش‌پردازش
            img = np.array(image_to_predict)
            
            # تغییر اندازه
            img = cv2.resize(img, (128, 128))
            
            # تبدیل به RGB (اگر RGBA بود)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # نرمال‌سازی
            img = img / 255.0
            
            # تغییر شکل برای ورودی مدل
            img = np.expand_dims(img, axis=0)
            
            # پیش‌بینی
            prediction = model.predict(img)
            predicted_idx = np.argmax(prediction[0])
            predicted_class = classes[predicted_idx]
            confidence = prediction[0][predicted_idx] * 100
            
            # نمایش نتیجه
            st.markdown("---")
            st.subheader("🌱 نتیجه تشخیص:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("نوع گیاه", predicted_class)
            with col2:
                st.metric("درصد اطمینان", f"{confidence:.2f}%")
            
            # نوار پیشرفت اعتماد به نفس
            st.progress(int(confidence))
            
            # نمایش تمام احتمالات
            with st.expander("📊 مشاهده جزئیات بیشتر"):
                for i, (cls, prob) in enumerate(zip(classes, prediction[0])):
                    st.write(f"{cls}: {prob*100:.2f}%")
                    st.progress(int(prob*100))

# راهنما
with st.sidebar:
    st.header("📖 راهنما")
    st.markdown("""
    1. عکس برگ گیاه را آپلود کنید
    2. یا با دوربین از برگ عکس بگیرید
    3. روی دکمه تشخیص کلیک کنید
    4. نتیجه را مشاهده کنید
    
    **توجه:**
    - عکس باید از برگ گیاه باشد
    - کیفیت عکس روی دقت تأثیر دارد
    - پس‌زمینه ساده بهتر جواب می‌دهد
    """)
    
    st.markdown("---")
    st.caption("پروژه تشخیص گیاه با هوش مصنوعی | CNN Model")