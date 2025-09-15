# 🎯 PromptXpert – GPT Prompt Generator theo hành vi khách hàng

AI Web App giúp bạn tạo prompt GPT chất lượng cao chỉ với 1 click – tùy chỉnh theo:
- Hành vi khách hàng (Nhận biết / Cân nhắc / Ra quyết định)
- Mục tiêu truyền thông (Tăng chuyển đổi, Tạo niềm tin, Gọi vốn...)
- Giao diện chọn ngôn ngữ: 🇻🇳 Tiếng Việt / 🇺🇸 English
- Sinh prompt đa tầng: bản đầy đủ, bản ads rút gọn, và PDF báo cáo

## 🚀 Tính năng nổi bật
- ✅ Chọn framework phù hợp với ngữ cảnh (AIDA, CARE, PAS…)
- ✅ Gợi ý thêm các framework tương tự (AI clustering)
- ✅ Xuất bản song ngữ: Việt & Anh
- ✅ Xuất file PDF chia sẻ cho investor / team
- ✅ Dễ dàng mở rộng thành hệ thống SAAS nội bộ

## 📦 Cài đặt local
```bash
pip install -r requirements.txt
streamlit run app.py
```

Sau khi chạy lệnh trên, truy cập `http://localhost:8501` trên trình duyệt để sử dụng ứng dụng.

Dataset trong `app.py` đã được mở rộng với các ngành PPP quy mô quốc tế (hạ tầng, năng lượng, viễn thông...), giúp doanh nghiệp lớn dễ dàng tùy biến thêm theo nhu cầu.

## 🏗️ PPP Financial Model Dashboard
Ứng dụng web tương tác mô phỏng dòng tiền và các chỉ số NPV, IRR, DSCR cho dự án PPP.

### Chạy mô hình
```bash
streamlit run ppp_app.py
```
Sau khi chạy, mở trình duyệt tới `http://localhost:8501` để xem dashboard và xuất báo cáo Excel.
