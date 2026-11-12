# LifeMate v5

V5 chép lại yêu cầu cuối bằng PhoWhisper-large và giữ bản chép lịch sử từ Whisper large-v3-turbo. Qwen3-4B chạy local tạo lời gọi công cụ. Bộ thực thi kiểm tra ID đã quan sát, tính mốc thời gian và số dư, hỏi lại khi gặp liên hệ mơ hồ và phát lại lời gọi bằng simulator.

| Chỉ số nội bộ trên 309 mẫu | V4 | V5 chạy riêng | V5 có bộ chọn, kiểm tra chéo |
|---|---:|---:|---:|
| Khớp toàn chuỗi lời gọi | 14,89% | 15,86% | 19,42% |
| F1 công cụ và tham số | 35,49% | 40,17% | 38,46% |
| Khớp trạng thái cuối | 34,63% | 37,54% | 41,42% |
| Quyết định hỏi lại đúng | 87,06% | 84,14% | 83,50% |

PhoWhisper giảm WER yêu cầu cuối từ 32,84% xuống 25,84% trên 309 bản ghi âm validation. Trên 73 đoạn dev, WER giảm từ 35,24% xuống 27,31%. Qwen3.5-4B được thử trên cùng 72 mẫu với bản đối chứng nhưng khớp trạng thái cuối thấp hơn v4, nên pipeline chọn Qwen3-4B.

Bộ chọn học từ kết quả v4 và v5 trên train validation. Việc đo ở cột cuối dùng năm phần, giữ các mẫu cùng nhóm trong một phần. Mỗi quyết định đánh giá do mô hình học trên bốn phần còn lại tạo ra. Bộ chọn cuối được huấn luyện trên toàn bộ 309 mẫu để áp dụng cho public test. Nó dùng thống kê lời gọi và kết quả thực thi, không có nhãn public. Cây chọn cuối giữ v4 khi v5 đề xuất nhiều hơn một thao tác ghi và tổng số lời gọi tăng.

Điểm public của v4 là 17,86. Các tỷ lệ trong bảng đo trên dữ liệu train giữ lại. Điểm v5 do portal xác định khi nộp. Quyết định hỏi lại trên toàn tập vẫn là điểm yếu và v5 chưa huấn luyện LoRA.

## File public

`submission_v5.jsonl` có đủ 634 mẫu. Bộ chọn lấy 588 kết quả từ v5 và 46 kết quả từ v4. Trình kiểm tra chính thức báo 0 lỗi schema và toàn bộ hash khớp khi phát lại 1.294 lời gọi. Hai mẫu chạm giới hạn vòng trong lượt suy luận v5 trước bước chọn.

SHA-256 `B385A21D8F08731F48B2ABED9E431378BC2F1FF4F383C3B5D90C135795D84BB2`.

## Tái lập kết quả

- `run.py` chạy Qwen qua localhost và lưu checkpoint từng mẫu.
- `asr_handoff.py` ghép bản chép mới với ngữ cảnh có sẵn.
- `refine.py --normalize` kiểm tra và phát lại các thao tác của agent.
- `selector.py calibrate` tạo cây và đánh giá kiểm tra chéo. `selector.py public --export` áp dụng cây, đối chiếu replay và chạy trình kiểm tra chính thức.
- Hai nguồn trọng số là [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) và [PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large), với bản chuyển đổi CT2 đã ghim checksum.
