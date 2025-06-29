# Dự đoán Hủy đặt phòng Khách sạn (Hotel Booking Cancellation Prediction)

Dự án này là một pipeline Machine Learning hoàn chỉnh nhằm mục đích dự đoán liệu một lượt đặt phòng khách sạn có bị hủy hay không, dựa trên các thông tin chi tiết của lượt đặt phòng đó. Dự án bao gồm các giai đoạn từ thu thập dữ liệu, tiền xử lý, huấn luyện mô hình, tinh chỉnh siêu tham số, và cuối cùng là triển khai mô hình thông qua một ứng dụng web Flask.

## 📋 Mục lục

- [Luồng hoạt động (Workflow)](#-luồng-hoạt-động-workflow)
- [Các tính năng chính](#-các-tính-năng-chính)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Hướng dẫn cài đặt và Chạy dự án](#-hướng-dẫn-cài-đặt-và-chạy-dự-án)
  - [Bước 1: Thiết lập môi trường](#bước-1-thiết-lập-môi-trường)
  - [Bước 2: Cấu hình dự án](#bước-2-cấu-hình-dự-án)
  - [Bước 3: Chạy Pipeline Huấn luyện](#bước-3-chạy-pipeline-huấn-luyện)
  - [Bước 4: Chạy Ứng dụng Web để Dự đoán](#bước-4-chạy-ứng-dụng-web-để-dự-đoán)
- [Triển khai CI/CD](#-triển-khai-cicd)


## 🚀 Luồng hoạt động (Workflow)

Dự án được xây dựng theo một pipeline gồm các bước tuần tự, đảm bảo tính module hóa và dễ dàng quản lý:

1.  **Nạp Dữ liệu (Data Ingestion)**: Tự động tải dữ liệu gốc từ Google Cloud Storage (GCS), sau đó chia thành hai tập `train.csv` và `test.csv`.
2.  **Tiền xử lý Dữ liệu (Data Processing)**:
    - Làm sạch dữ liệu: Loại bỏ các cột không cần thiết và các bản ghi trùng lặp.
    - Mã hóa (Encoding): Chuyển đổi các đặc trưng dạng chữ (categorical) sang dạng số bằng `LabelEncoder`.
    - Xử lý độ xiên (Skewness Handling): Áp dụng phép biến đổi logarit để chuẩn hóa phân phối của các đặc trưng số.
    - Cân bằng Dữ liệu (Data Balancing): Sử dụng kỹ thuật SMOTE trên tập huấn luyện để giải quyết vấn đề mất cân bằng giữa các lớp (hủy và không hủy).
    - Lựa chọn Đặc trưng (Feature Selection): Sử dụng mô hình `RandomForestClassifier` để đánh giá và chọn ra các đặc trưng quan trọng nhất.
3.  **Huấn luyện Mô hình (Model Training)**:
    - Tinh chỉnh Siêu tham số: Sử dụng `RandomizedSearchCV` để tự động tìm kiếm bộ siêu tham số tốt nhất cho mô hình `RandomForestClassifier`.
    - Huấn luyện mô hình cuối cùng với bộ tham số tốt nhất trên toàn bộ tập huấn luyện đã xử lý.
    - Lưu lại mô hình đã được huấn luyện.
4.  **Dự đoán (Prediction)**:
    - Một ứng dụng web được xây dựng bằng Flask cung cấp giao diện người dùng để nhập thông tin đặt phòng.
    - Dữ liệu đầu vào được xử lý và đưa vào mô hình đã được huấn luyện để đưa ra dự đoán "Canceled" (Hủy) hoặc "Not Canceled" (Không hủy).

## ✨ Các tính năng chính

- **Pipeline End-to-End**: Toàn bộ quy trình từ dữ liệu thô đến mô hình sẵn sàng dự đoán được tự động hóa.
- **Cấu hình linh hoạt**: Các tham số quan trọng (đường dẫn, tên bucket, tham số mô hình) được quản lý tập trung trong file `config.yaml`.
- **Ghi log chi tiết**: Tích hợp hệ thống logging để theo dõi và gỡ lỗi mọi bước trong pipeline.
- **Xử lý Exception tùy chỉnh**: Xây dựng các lớp Exception riêng để xử lý lỗi một cách rõ ràng.
- **Giao diện Web**: Cung cấp một giao diện đơn giản để người dùng cuối có thể tương tác và nhận dự đoán.
- **Sẵn sàng cho CI/CD**: Cấu trúc dự án và các tài liệu đi kèm đã được chuẩn bị cho việc triển khai liên tục.

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.8+
- **Thư viện chính**:
  - **Phân tích & Xử lý dữ liệu**: Pandas, NumPy
  - **Học máy**: Scikit-learn, Imbalanced-learn (imblearn)
  - **Web Framework**: Flask
  - **Cloud**: Google Cloud Storage (GCS)
- **Đóng gói**: `setuptools`

## 📂 Cấu trúc dự án

```
.
├── config/                 # Chứa các file cấu hình (YAML, Python)
├── pipeline/               # Định nghĩa các pipeline (huấn luyện, dự đoán)
├── src/                    # Mã nguồn chính của dự án
│   ├── components/         # Các thành phần của pipeline (ingestion, processing, training)
│   ├── logger.py           # Thiết lập logging
│   └── custom_exception.py # Định nghĩa exception tùy chỉnh
├── utils/                  # Chứa các hàm tiện ích chung
├── templates/              # Chứa file HTML cho ứng dụng Flask
├── static/                 # Chứa file CSS
├── notebook/               # Chứa Jupyter Notebook cho việc khám phá dữ liệu
├── application.py          # File để chạy ứng dụng Flask
├── requirements.txt        # Danh sách các thư viện cần thiết
├── setup.py                # File cài đặt dự án như một package
└── README.md               # Tài liệu mô tả dự án
```

## ⚙️ Hướng dẫn cài đặt và Chạy dự án

### Bước 1: Thiết lập môi trường

1.  **Clone repository về máy của bạn:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```
2.  **Tạo và kích hoạt môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Trên Windows: venv\Scripts\activate
    ```
3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -e .
    ```

### Bước 2: Cấu hình dự án

Trước khi chạy, bạn cần thiết lập thông tin về Google Cloud Storage của mình.

1.  Mở file `config/config.yaml`.
2.  Chỉnh sửa các giá trị trong mục `data_ingestion`:
    ```yaml
    data_ingestion:
      bucket_name: your-gcs-bucket-name  # <-- THAY THẾ bằng tên bucket của bạn
      bucket_file_name: Hotel Reservations.csv # Tên file trong bucket
      train_ratio: 0.8
    ```
3.  Đảm bảo rằng file `Hotel Reservations.csv` đã được tải lên bucket GCS của bạn.
4.  Đồng thời, hãy đảm bảo rằng môi trường của bạn đã được xác thực với Google Cloud.

### Bước 3: Chạy Pipeline Huấn luyện

Để thực hiện toàn bộ quá trình từ tải dữ liệu, xử lý, đến huấn luyện và lưu mô hình, hãy chạy lệnh sau từ thư mục gốc của dự án:

```bash
python pipeline/training_pipeline.py
```

Quá trình này có thể mất vài phút. Sau khi hoàn tất, các file dữ liệu đã xử lý sẽ nằm trong thư mục `artifact/processed` và mô hình đã huấn luyện (`model.pkl`) sẽ nằm trong thư mục `artifact/model`.

### Bước 4: Chạy Ứng dụng Web để Dự đoán

Sau khi pipeline huấn luyện đã chạy thành công, bạn có thể khởi động ứng dụng web để thực hiện các dự đoán theo thời gian thực.

1.  **Chạy ứng dụng Flask:**
    ```bash
    python application.py
    ```
2.  **Truy cập ứng dụng:**
    Mở trình duyệt web và truy cập địa chỉ: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Bạn sẽ thấy một giao diện web nơi bạn có thể nhập các thông tin của một lượt đặt phòng và nhấn "Predict" để xem kết quả dự đoán.

## 🔄 Triển khai CI/CD

Dự án này đã được cấu trúc để có thể tích hợp vào một quy trình CI/CD. Các tài liệu trong thư mục `CI-CD Deployment Materials` cung cấp hướng dẫn các bước để triển khai tự động bằng Jenkins.
