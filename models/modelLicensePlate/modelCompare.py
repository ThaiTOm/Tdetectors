import torch
import time
import numpy as np

# --- CÀI ĐẶT ---
# 1. Đường dẫn đến file trọng số .pt hoặc .pth của bạn
#    Đây là file bạn dùng TRƯỚC KHI chuyển đổi sang ONNX.
#    Thường nó nằm trong thư mục runs/detect/train/weights/best.pt
PYTORCH_MODEL_PATH = '/home/geso/Tdetectors/models/modelLicensePlate/runs/detect/train/weights/best.pt' # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY

# 2. Kích thước input của mô hình (giống như trong file .xml của bạn)
#    Từ log của bạn: [1, 3, 640, 640]
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 3. Số lần chạy suy luận để đo lường (càng nhiều càng chính xác)
NUM_INFERENCES = 200

# --- CHUẨN BỊ MÔ HÌNH ---
# Quan trọng: Đảm bảo bạn đang ở trong môi trường conda/virtualenv đã cài PyTorch
# (KHÔNG phải môi trường openvino nếu chúng tách biệt)

# Tải mô hình từ Ultralytics YOLOv8 (đây là cách phổ biến)
# Nếu bạn dùng YOLOv5 hoặc mô hình khác, cách tải sẽ khác đi
# from models.common import DetectMultiBackend # YOLOv5
# model = DetectMultiBackend(PYTORCH_MODEL_PATH, device=torch.device('cpu'))

# Cách đơn giản nhất là dùng hub của torch
print(f"Loading model from {PYTORCH_MODEL_PATH}...")
device = torch.device('cuda:0')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=PYTORCH_MODEL_PATH) # <-- Hoặc dùng yolov8 nếu là yolov8
from ultralytics import YOLO

model = YOLO(PYTORCH_MODEL_PATH)  # Nếu bạn dùng YOLOv8
# Hoặc nếu bạn có code mô hình:
# model = YourModelClass()
# model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device)['model'])

model.to(device)
model.eval() # Rất quan trọng: Chuyển sang chế độ suy luận
print("Model loaded successfully on CPU and set to evaluation mode.")

# Tạo một input giả ngẫu nhiên (giống như benchmark_app đã làm)
# Input có dạng [batch, channels, height, width]
dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH).to(device)
print(f"Created a dummy input tensor of shape: {dummy_input.shape}")

# --- CHẠY WARM-UP ---
# Chạy suy luận vài lần đầu tiên để "làm nóng" hệ thống.
# Các lần chạy đầu tiên thường chậm hơn do khởi tạo.
print("Running warm-up inferences...")
for _ in range(20):
    with torch.no_grad(): # Không tính gradient để tăng tốc
        _ = model(dummy_input)

# --- BẮT ĐẦU ĐO LƯỜNG ---
print(f"\nStarting benchmark for {NUM_INFERENCES} inferences...")
start_time = time.time()

for _ in range(NUM_INFERENCES):
    with torch.no_grad():
        _ = model(dummy_input)

end_time = time.time()
print("Benchmark finished.")

# --- TÍNH TOÁN VÀ IN KẾT QUẢ ---
total_duration = end_time - start_time
average_latency_ms = (total_duration / NUM_INFERENCES) * 1000
throughput_fps = NUM_INFERENCES / total_duration

print("\n--- PyTorch Original Model Performance (on CPU) ---")
print(f"Total time for {NUM_INFERENCES} inferences: {total_duration:.2f} seconds")
print(f"Average Latency: {average_latency_ms:.2f} ms")
print(f"Throughput: {throughput_fps:.2f} FPS")
print("--------------------------------------------------")