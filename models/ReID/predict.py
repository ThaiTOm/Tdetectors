import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import shutil
from processor import get_model  # Giả sử get_model nằm trong processor.py
import yaml
import imagehash  # Thêm thư viện này


# --- Bước 0: Loại bỏ ảnh trùng lặp ở mức pixel/hash ---

def initial_deduplicate_images(source_dir, target_dir, hash_size=8, hamming_distance_threshold=5):
    """
    Loại bỏ các ảnh trùng lặp hoặc rất giống nhau dựa trên perceptual hash (dhash).
    Các ảnh duy nhất sẽ được sao chép vào target_dir.

    Args:
        source_dir (str): Thư mục chứa tất cả các ảnh gốc.
        target_dir (str): Thư mục đích để lưu trữ các ảnh đã loại bỏ trùng lặp.
        hash_size (int): Kích thước của hash (ví dụ: 8 cho hash 8x8).
        hamming_distance_threshold (int): Ngưỡng khoảng cách Hamming.
                                           Nếu khoảng cách giữa hai hash nhỏ hơn hoặc bằng ngưỡng này,
                                           chúng được coi là ảnh trùng lặp.
                                           0: chỉ chấp nhận ảnh giống hệt.
                                           5-10: tốt cho việc tìm ảnh "gần giống".
    Returns:
        list: Danh sách các đường dẫn đến ảnh đã được loại bỏ trùng lặp trong target_dir.
    """
    if os.path.exists(target_dir):
        print(f"Thư mục '{target_dir}' đã tồn tại. Đang xóa và tạo lại cho quá trình deduplication ban đầu.")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    unique_hashes = []  # Lưu trữ các tuple (hash_object, đường_dẫn_ảnh_gốc)
    deduplicated_paths = []

    # Lấy tất cả các file ảnh phổ biến
    all_files = glob.glob(os.path.join(source_dir, '*'))
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

    if not image_files:
        print(
            f"Không tìm thấy ảnh nào trong '{source_dir}' với các định dạng phổ biến. Vui lòng kiểm tra đường dẫn và định dạng ảnh.")
        return []

    print(f"\n--- BƯỚC 0: Loại bỏ ảnh trùng lặp ở mức pixel/hash ---")
    print(f"Đang xử lý {len(image_files)} ảnh từ '{source_dir}'...")

    for img_path in tqdm(image_files, desc="Initial Deduplication"):
        try:
            img = Image.open(img_path)
            # Chuyển đổi sang RGB để tránh lỗi với ảnh grayscale hoặc ảnh có kênh alpha
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Tính toán dhash cho ảnh hiện tại
            current_hash = imagehash.dhash(img, hash_size=hash_size)

            is_duplicate = False
            # So sánh hash hiện tại với tất cả các hash duy nhất đã tìm thấy
            for existing_hash, _ in unique_hashes:
                # Tính khoảng cách Hamming giữa hai hash
                # Nếu khoảng cách <= ngưỡng, thì coi là trùng lặp
                if (current_hash - existing_hash) <= hamming_distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Nếu không phải là trùng lặp, thêm hash vào danh sách duy nhất
                unique_hashes.append((current_hash, img_path))
                # Sao chép ảnh vào thư mục đích
                dest_path = os.path.join(target_dir, os.path.basename(img_path))
                shutil.copy(img_path, dest_path)
                deduplicated_paths.append(dest_path)

        except Exception as e:
            print(f"Cảnh báo: Lỗi khi xử lý ảnh '{img_path}' trong quá trình deduplicate ban đầu. Bỏ qua. Lỗi: {e}")
            continue

    print(
        f"Quá trình deduplication ban đầu hoàn tất. Đã giữ lại {len(deduplicated_paths)} ảnh duy nhất trong '{target_dir}'.")
    print(f"Số ảnh gốc: {len(image_files)}, Số ảnh bị loại bỏ: {len(image_files) - len(deduplicated_paths)}")
    return deduplicated_paths


# --- Bước 1: Thiết lập (Các hàm này giữ nguyên) ---

def load_trained_model(model_path, class_num, device):
    """
    Hàm tải mô hình đã được huấn luyện.
    Bạn cần điều chỉnh các tham số của MBR_model cho khớp với mô hình đã lưu.
    """
    try:
        with open("./config/config_duythai.yaml", "r") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file cấu hình './config/config_duythai.yaml'.")
        print("Vui lòng đảm bảo đường dẫn đến file cấu hình là chính xác.")
        exit()

    model = get_model(data, device)

    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path} and set to evaluation mode.")
    return model


def get_image_transforms():
    """
    Định nghĩa phép biến đổi cho ảnh đầu vào.
    Phải giống với 'teste_transform' trong lúc huấn luyện.
    """
    n_mean = [0.485, 0.456, 0.406]
    n_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(n_mean, n_std),
    ])


# --- Bước 2: Trích xuất Đặc trưng (Các hàm này giữ nguyên) ---

@torch.no_grad()
def extract_features(model, image_paths, transform, device, batch_size=32):
    """
    Trích xuất vector đặc trưng cuối cùng cho một danh sách ảnh.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("Không có ảnh nào để trích xuất đặc trưng.")
        return torch.tensor([])

    all_features = []
    processed_image_paths = []  # Để lưu các đường dẫn ảnh đã xử lý thành công

    model.eval()

    print(f"\n--- BƯỚC 1: Trích xuất Đặc trưng bằng mô hình Re-ID ---")
    for i in tqdm(range(0, num_images, batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        current_batch_processed_paths = []  # Paths for current successful batch

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                current_batch_processed_paths.append(img_path)
            except Exception as e:
                print(f"Cảnh báo: Không thể tải hoặc xử lý ảnh '{img_path}'. Bỏ qua. Lỗi: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        _, _, ffs, _ = model(batch_tensor, cam=None, view=None)

        if not ffs:
            print(f"Cảnh báo: Mô hình không trả về đặc trưng cho batch bắt đầu từ {batch_paths[0]}.")
            continue

        batch_features = torch.cat(ffs, dim=1)
        batch_features = nn.functional.normalize(batch_features, p=2, dim=1)

        all_features.append(batch_features.cpu())
        processed_image_paths.extend(current_batch_processed_paths)

    if not all_features:
        print("Không có đặc trưng nào được trích xuất thành công.")
        return torch.tensor([]), []

    return torch.cat(all_features, dim=0), processed_image_paths


# --- Bước 3: Phân cụm (Các hàm này giữ nguyên) ---

def cluster_images(features, image_paths, distance_threshold=1.0):
    """
    Phân cụm các ảnh dựa trên vector đặc trưng.
    Sử dụng Agglomerative Clustering (Phân cụm phân cấp tích tụ).
    """
    if features.shape[0] == 0:
        print("Không có đặc trưng để phân cụm. Vui lòng kiểm tra lại bước trích xuất đặc trưng.")
        return {}, []

    print(f"\n--- BƯỚC 2: Phân cụm ảnh bằng đặc trưng Re-ID ---")
    print(f"Đang phân cụm {len(image_paths)} ảnh với ngưỡng khoảng cách: {distance_threshold}")

    features_np = features.numpy()

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='euclidean',
        linkage='average',
        distance_threshold=distance_threshold
    )

    labels = clustering.fit_predict(features_np)

    grouped_images = {}
    for img_path, label in zip(image_paths, labels):
        if label not in grouped_images:
            grouped_images[label] = []
        grouped_images[label].append(img_path)

    print(f"Tìm thấy {len(grouped_images)} nhóm đối tượng/xe duy nhất.")
    return grouped_images, labels


# --- Bước 4: Chọn ảnh đại diện và Sao chép ảnh đã loại bỏ trùng lặp (Giữ nguyên) ---

def select_representative_images(grouped_images):
    """
    Chọn một ảnh đại diện từ mỗi nhóm đã phân cụm.
    Mặc định chọn ảnh đầu tiên trong danh sách của mỗi nhóm.
    """
    representative_paths = []
    print("\nĐang chọn một ảnh đại diện cho mỗi nhóm...")
    for label, paths in grouped_images.items():
        if paths:
            representative_paths.append(paths[0])

    print(f"Đã chọn {len(representative_paths)} ảnh đại diện.")
    return representative_paths


def copy_reid_classified_images(representative_paths, output_dir="reid_classified_images"):
    """
    Sao chép các ảnh đại diện từ Re-ID clustering vào một thư mục mới.
    Đây là kết quả cuối cùng sau khi deduplicate hai lớp.
    """
    if os.path.exists(output_dir):
        print(f"Thư mục '{output_dir}' đã tồn tại. Đang xóa và tạo lại.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nĐang sao chép {len(representative_paths)} ảnh đã được phân loại bởi Re-ID tới '{output_dir}'...")
    for img_path in tqdm(representative_paths, desc="Copying Re-ID Classified"):
        shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
    print(f"Phân loại Re-ID hoàn tất! Ảnh duy nhất được lưu tại '{output_dir}'")


def save_clustered_images_for_inspection(grouped_images, output_dir="grouped_images_for_inspection"):
    """
    Lưu các ảnh đã được phân nhóm bởi Re-ID vào các thư mục riêng.
    Hữu ích để kiểm tra kết quả phân cụm Re-ID.
    """
    if os.path.exists(output_dir):
        print(f"Thư mục '{output_dir}' đã tồn tại. Đang xóa và tạo lại.")
        shutil.rmtree(output_dir)

    print(f"\nĐang lưu các nhóm ảnh (để kiểm tra) tới '{output_dir}'...")
    for label, paths in tqdm(grouped_images.items(), desc="Saving Re-ID Groups"):
        group_dir = os.path.join(output_dir, f"group_{label}")
        os.makedirs(group_dir, exist_ok=True)
        for img_path in paths:
            shutil.copy(img_path, os.path.join(group_dir, os.path.basename(img_path)))

    print("Hoàn thành việc lưu các nhóm ảnh!")


# --- Hàm main để thực thi ---
if __name__ == "__main__":
    # --- CÁC THAM SỐ CẦN THAY ĐỔI ---
    RAW_IMAGE_DIR = "/home/geso/Tdetectors/models/ReID/test"  # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY (Thư mục gốc chứa tất cả ảnh)

    # Thư mục tạm thời cho ảnh sau khi loại bỏ trùng lặp pixel
    DEDUPLICATED_PIXEL_LEVEL_DIR = "deduplicated_pixel_level"

    # Tham số cho deduplicate pixel-level (Bước 0)
    # HASH_SIZE: Kích thước hash. Lớn hơn -> chi tiết hơn, ít va chạm hơn, nhưng chậm hơn. 8 là phổ biến.
    HASH_SIZE = 8
    # HAMMING_THRESHOLD: Ngưỡng khoảng cách Hamming.
    # 0: chỉ trùng khớp chính xác (MD5 sẽ làm việc tương tự).
    # 1-5: trùng khớp rất gần, chấp nhận thay đổi nhỏ (nén, resize 1-2px).
    # 6-15: trùng khớp "gần giống", chấp nhận biến đổi đáng kể hơn.
    # Thử nghiệm với giá trị này để phù hợp với dữ liệu của bạn.
    HAMMING_THRESHOLD = 5

    # Tham số cho mô hình Re-ID và phân cụm (Bước 1 & 2)
    MODEL_PATH = "/home/geso/Tdetectors/models/ReID/logs/VRIC/Baseline/22/last.pt"  # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY
    CLASS_NUM = 49725  # Số lượng lớp khi huấn luyện (ví dụ Veri-776 là 776)

    # Ngưỡng khoảng cách cho phân cụm Re-ID (sau khi đã trích xuất đặc trưng).
    # Cần tinh chỉnh. Giá trị nhỏ hơn -> ít cụm, mỗi cụm chặt chẽ hơn.
    # Giá trị lớn hơn -> nhiều cụm, mỗi cụm lỏng lẻo hơn.
    DISTANCE_THRESHOLD_REID = 0.4  # <<< THAY ĐỔI VÀ TINH CHỈNH NGƯỠNG NÀY

    # Thư mục đầu ra cuối cùng sau khi phân loại bằng Re-ID
    REID_CLASSIFIED_OUTPUT_DIR = "reid_classified_unique_images"
    # Thư mục để kiểm tra các nhóm ảnh Re-ID (tùy chọn)
    GROUPED_OUTPUT_DIR_FOR_INSPECTION = "reid_grouped_images_for_inspection"

    # Kiểm tra xem các đường dẫn có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found at '{MODEL_PATH}'")
        exit()
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"Error: Raw image directory not found at '{RAW_IMAGE_DIR}'")
        exit()

    # --- BẮT ĐẦU QUY TRÌNH ---
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    # 0. Loại bỏ ảnh trùng lặp ở mức pixel/hash trước
    deduplicated_pixel_image_paths = initial_deduplicate_images(
        RAW_IMAGE_DIR,
        DEDUPLICATED_PIXEL_LEVEL_DIR,
        hash_size=HASH_SIZE,
        hamming_distance_threshold=HAMMING_THRESHOLD
    )

    if not deduplicated_pixel_image_paths:
        print("Không có ảnh nào sau khi loại bỏ trùng lặp ở mức pixel. Dừng xử lý.")
        exit()

    # 1. Tải mô hình và các phép biến đổi
    model = load_trained_model(MODEL_PATH, CLASS_NUM, device)
    transform = get_image_transforms()

    # 2. Trích xuất đặc trưng từ các ảnh đã được deduplicate ở mức pixel
    # Truyền `deduplicated_pixel_image_paths` làm đầu vào cho Re-ID
    features, reid_processed_paths = extract_features(model, deduplicated_pixel_image_paths, transform, device)

    # Kiểm tra nếu không có đặc trưng nào được trích xuất
    if features.shape[0] == 0:
        print("Không thể tiếp tục vì không có đặc trưng ảnh nào được trích xuất từ các ảnh đã qua deduplicate pixel.")
        exit()

    # 3. Phân cụm các ảnh dựa trên đặc trưng Re-ID
    # Đảm bảo image_paths được truyền vào cluster_images khớp với features
    grouped_images_reid, _ = cluster_images(features, reid_processed_paths, distance_threshold=DISTANCE_THRESHOLD_REID)

    # 4. Chọn ảnh đại diện từ mỗi cụm Re-ID (bước loại bỏ ảnh trùng lặp Re-ID)
    representative_images_reid = select_representative_images(grouped_images_reid)

    # 5. Sao chép các ảnh đã được phân loại bởi Re-ID vào một thư mục mới (kết quả cuối cùng)
    copy_reid_classified_images(representative_images_reid, REID_CLASSIFIED_OUTPUT_DIR)

    # (Tùy chọn) 6. Lưu tất cả các nhóm ảnh Re-ID vào các thư mục riêng để kiểm tra trực quan
    save_clustered_images_for_inspection(grouped_images_reid, output_dir=GROUPED_OUTPUT_DIR_FOR_INSPECTION)

    # In ra tóm tắt cuối cùng
    print(f"\n--- TÓM TẮT QUÁ TRÌNH ---")
    print(f"Tổng số ảnh ban đầu trong '{RAW_IMAGE_DIR}': {len(glob.glob(os.path.join(RAW_IMAGE_DIR, '*.*')))}")
    print(
        f"Số ảnh sau khi loại bỏ trùng lặp pixel/hash: {len(deduplicated_pixel_image_paths)} (lưu tại '{DEDUPLICATED_PIXEL_LEVEL_DIR}')")
    print(f"Số ảnh được trích xuất đặc trưng bởi Re-ID: {len(reid_processed_paths)}")
    print(f"Số nhóm đối tượng/xe duy nhất được tìm thấy bởi Re-ID: {len(grouped_images_reid)}")
    print(
        f"Tổng số ảnh duy nhất cuối cùng (sau Re-ID): {len(representative_images_reid)} (lưu tại '{REID_CLASSIFIED_OUTPUT_DIR}')")
    print(f"Kiểm tra các nhóm Re-ID tại: '{GROUPED_OUTPUT_DIR_FOR_INSPECTION}'")