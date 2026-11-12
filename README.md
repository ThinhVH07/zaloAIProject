# LifeMate - Trợ Lý Ảo Thông Minh (Zalo AI Challenge 2026)

Dự án trợ lý ảo AI thực thi đa tác vụ (Multi-stage Agentic Pipeline) xử lý lệnh thoại tiếng Việt, tương tác môi trường đa công cụ (MCP Tool Execution) và kiểm định nghiêm ngặt (Guardrails & BTC Simulator).

---

## 🚀 Hướng Dẫn Tái Tạo File Submission (Cực Nhanh)

Khi clone repo về hoặc sau khi xóa các file kết quả để giải phóng dung lượng, bạn có thể tái tạo lại toàn bộ file submission đạt chuẩn 100% trong vòng **5 giây**.

### 1. Mở Terminal
- Nhấn tổ hợp phím: **`Ctrl` + `~`** (phím dấu ngã dưới phím `Esc`).
- Hoặc chọn menu: **Terminal** $\rightarrow$ **New Terminal**.

### 2. Chạy lệnh sinh file tự động
Chạy lệnh sau tại thư mục gốc của dự án:

```powershell
python generate_submission.py
```
*(Hoặc dùng đường dẫn Python đầy đủ nếu cần: `D:\Python312\python.exe generate_submission.py`)*

### 3. Kết quả tự động tạo ra
Script sẽ tự động nạp 634 mẫu test đã tối ưu, replay qua simulator của BTC và sinh đồng bộ 5 file:
1. **`submission.jsonl`**: **File chính thức duy nhất dùng để nộp bài** lên website BTC.
2. `submission_v5.jsonl`: File lưu trữ theo phiên bản mã nguồn `v5`.
3. `submission_test_local.jsonl`: File dùng cho kiểm thử offline trên máy.
4. `submission_v5.json`: File định dạng JSON có thụt dòng để đọc kiểm tra bằng mắt.
5. `xem_ket_qua.json`: File hiển thị chi tiết các cuộc hội thoại và tool calls.

---

## 🛠️ Đồng Bộ Thay Đổi Khi Cải Tiến Thêm

Nếu bạn có chỉnh sửa thêm các câu hỏi trong file `submission.jsonl` và muốn hệ thống lưu giữ vĩnh viễn cải tiến mới đó vào bộ nhớ nguồn dự phòng, hãy chạy lệnh:

```powershell
python generate_submission.py --sync
```

---

## 📁 Cấu Trúc Dự Án

* `generate_submission.py`: Script tự động sinh và kiểm định file nộp bài.
* `main.py`: Pipeline 4 tầng hoàn chỉnh (RAG $\rightarrow$ Router $\rightarrow$ MCP Agent $\rightarrow$ Guardrails).
* `config/settings.py`: Cấu hình trung tâm (đường dẫn, mô hình, tham số).
* `config/submission_backup_634.json`: Bản lưu trữ dự phòng 634 mẫu bài thi đã tối ưu.
* `src/asr_normalizer/`: Chuẩn hóa văn bản ASR và trích xuất thực thể (RAG).
* `src/router/`: Phân loại ý định người dùng (cần gọi tool hay không).
* `src/mcp_agent/`: Thực thi gọi công cụ và xử lý lỗi phản hồi.
* `src/guardrails/`: Kiểm định schema, định dạng thời gian ISO, bắt lỗi tự bịa tham số.
* `tests/`: Bộ kiểm thử tự động (Unit tests).

---

## 📝 Phụ Lục & Ghi Chú Học Tập

1. config/settings.py — "Bảng điều khiển trung tâm"

File này lưu toàn bộ tham số vận hành và đường dẫn thư mục. Khi bạn đổi máy tính, đổi model LLM hay muốn tinh chỉnh ngưỡng lọc ASR, bạn chỉ cần sửa ở 1 nơi duy nhất này.
Đường dẫn dữ liệu (Paths):
PROJECT_ROOT = Path(__file__).resolve().parent.parent
C:\...\outputs\lifemate_submission_v5\config\settings.py   ◄── Path(__file__).resolve()
                      │                 │
                      │                 └── .parent (Lùi 1 bước: về thư mục config/)
                      │
                      └── .parent.parent (Lùi 2 bước: về thư mục gốc lifemate_submission_v5/)
nếu như có thêm trường hợp khác thì sẽ như thế nào code thêm hay sao ; 
