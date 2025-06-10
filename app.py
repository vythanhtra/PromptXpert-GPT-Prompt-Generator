# app.py – GPT Prompt Generator V5: Đa tầng + UX nâng cấp + Song ngữ
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# Dataset mẫu (có thể mở rộng theo dữ liệu doanh nghiệp)
data = pd.DataFrame([
    {"industry": "Giáo dục", "objective": "Cân nhắc", "state": "Cân nhắc", "framework": "CARE"},
    {"industry": "Startup", "objective": "Gọi vốn", "state": "Ra quyết định", "framework": "AIDA"},
    {"industry": "Tài chính", "objective": "Tạo niềm tin", "state": "Mất niềm tin", "framework": "Before–After–Bridge"},
    {"industry": "Marketing", "objective": "Tăng chuyển đổi", "state": "Ra quyết định", "framework": "PAS"},
    {"industry": "Công nghệ", "objective": "Trình bày lợi ích", "state": "Cân nhắc", "framework": "FAB"},
    {"industry": "Bất động sản", "objective": "Gây chú ý", "state": "Nhận biết", "framework": "AIDA"},
])

# Vector hóa dữ liệu
text_data = data["industry"] + " " + data["objective"] + " " + data["state"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)

knn = NearestNeighbors(n_neighbors=1).fit(X)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
data["cluster"] = kmeans.labels_

def suggest_cluster_framework(industry, objective, customer_state):
    query = vectorizer.transform([f"{industry} {objective} {customer_state}"])
    cluster_id = kmeans.predict(query)[0]
    similar = data[data["cluster"] == cluster_id]["framework"].unique()
    return list(similar)

def select_framework(industry, objective, customer_state):
    query = vectorizer.transform([f"{industry} {objective} {customer_state}"])
    dist, idx = knn.kneighbors(query)
    return data.iloc[idx[0][0]]["framework"]

def generate_prompts(industry, objective, customer_state, format_type, lang_option):
    framework = select_framework(industry, objective, customer_state)
    related = suggest_cluster_framework(industry, objective, customer_state)
    related_str = ", ".join(related)

    # Full version (long-form content)
    vi_full = f"""Bạn là chuyên gia trong lĩnh vực {industry}.
Hãy viết nội dung dạng {format_type} theo mô hình {framework} dành cho khách hàng đang ở trạng thái "{customer_state}".
Nội dung cần đạt mục tiêu: {objective}. Hãy sử dụng phong cách chuyên nghiệp, dễ hiểu và tạo được ấn tượng rõ ràng.
Gợi ý thêm: Bạn có thể cân nhắc các framework tương tự theo nhóm hành vi: {related_str}"""

    en_full = f"""You are an expert in the {industry} field.
Write a {format_type}-style content using the {framework} framework for customers in the "{customer_state}" stage.
The goal of the content is: {objective}. Use a professional, easy-to-understand tone that creates a clear impression.
Additionally, consider similar behavioral frameworks: {related_str}"""

    # Short version (ads)
    vi_short = f"{format_type} ngắn theo mô hình {framework}, dành cho ngành {industry}, khách hàng ở trạng thái {customer_state}, mục tiêu: {objective}."
    en_short = f"Short {format_type} using {framework}, for {industry} industry, customer state: {customer_state}, goal: {objective}."

    return {
        "vi_full": vi_full.strip(),
        "en_full": en_full.strip(),
        "vi_short": vi_short.strip(),
        "en_short": en_short.strip()
    }

# Giao diện người dùng
st.set_page_config(page_title="GPT Prompt Generator V5", layout="centered")
st.title("🎯 GPT Prompt Generator – Đa tầng & Song ngữ")

col1, col2 = st.columns(2)
with col1:
    industry = st.selectbox("🔧 Ngành nghề / Industry:", data["industry"].unique())
    objective = st.selectbox("🎯 Mục tiêu / Goal:", data["objective"].unique())
with col2:
    customer_state = st.selectbox("🧠 Trạng thái khách hàng / Customer State:", data["state"].unique())
    format_type = st.text_input("📄 Định dạng nội dung / Format:", "Bài viết / Article")

lang_option = st.radio("🌐 Ngôn ngữ đầu ra / Output Language:", ["Chỉ Tiếng Việt", "Chỉ English", "Song ngữ / Bilingual"])

if st.button("🚀 Tạo Prompt / Generate Prompt"):
    result = generate_prompts(industry, objective, customer_state, format_type, lang_option)
    st.subheader("📄 Prompt đề xuất:")
    if lang_option == "Chỉ Tiếng Việt":
        st.code(result["vi_full"])
        st.text("📌 Bản rút gọn:")
        st.code(result["vi_short"])
    elif lang_option == "Chỉ English":
        st.code(result["en_full"])
        st.text("📌 Short version:")
        st.code(result["en_short"])
    else:
        st.markdown("**🇻🇳 Vietnamese:**")
        st.code(result["vi_full"])
        st.markdown("**🇺🇸 English:**")
        st.code(result["en_full"])
        st.text("📌 Bản rút gọn / Short version:")
        st.code(result["vi_short"] + "\n" + result["en_short"])
