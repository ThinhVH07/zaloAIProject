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