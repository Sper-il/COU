# Dự báo nhiệt độ Seq2Seq Jena — Huấn luyện trên Google Colab

> Notebook cho dự án `jena-seq2seq` (đề bài: Sequence-to-Sequence Weather Forecasting trên Jena Climate).

## Nội dung thực hiện

1. **Cài đặt** môi trường + clone repo + tải dữ liệu Jena.
2. **Quan sát dữ liệu**: kiểm tra chia tập, đường cơ sở `persistence` và `seasonal-naive`.
3. **Sweep 15 epoch**: thử nhiều cấu hình siêu tham số để tìm cấu hình tốt nhất.
4. **Train 30 epoch** bằng cấu hình tốt nhất tìm được.
5. **Quan sát quá trình train**: biểu đồ loss/MAE theo epoch, học tốc độ học, teacher forcing, độ trễ suy luận.
6. **Đánh giá** trên tập kiểm thử 2016: MAE/RMSE tại tầm 1/24/72/144, so sánh các kiến trúc.

## Yêu cầu

- Chạy trên **GPU** (Colab T4 hoặc cao hơn) để huấn luyện 3 kiến trúc xong trong thời gian hợp lý. Nếu không có GPU, notebook vẫn chạy nhưng lâu hơn nhiều.
- Dữ liệu Jena được tải tự động từ Kaggle hoặc link trực tiếp bên dưới.
