import pandas as pd
import os
import asyncio
import aiohttp
import aiofiles
import time
from datetime import datetime, timedelta
from haversine import haversine, Unit

# --- 1. CẤU HÌNH TRUNG TÂM ---
CONFIG = {
    "EXCEL_FILE": 'DANH_SACH_CAMERA_DAY_DU_FINAL.xlsx',
    "SHEET_NAME": 'Sheet1',
    "BASE_OUTPUT_FOLDER": 'captured_images',
    "RUN_DURATION_HOURS": 48,
    "DOWNLOAD_INTERVAL_SECONDS": 5,
    "CAMERA_URL_TEMPLATE": "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id={}&w=500&h=300",

    # --- CẤU HÌNH MỚI CHO VIỆC TÌM KIẾM CAMERA GẦN NHẤT ---
    # Thay đổi tọa độ này thành vị trí trung tâm bạn muốn theo dõi
    # Ví dụ: Tọa độ Dinh Độc Lập
    "REFERENCE_POINT": (10.7770, 106.6953),  # (Vĩ độ, Kinh độ)
    "NEAREST_CAMERA_COUNT": 200
}

# --- HEADERS ĐỂ GIẢ MẠO TRÌNH DUYỆT (CHỐNG LỖI 403) ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Referer': 'https://giaothong.hochiminhcity.gov.vn/',
    'Connection': 'keep-alive',
    'Cookie': ".VDMS=E00768546E36A98DC4139B71C010A05606F712E73F6CA25FB810DAFBBEA9EE537C2A192858C37BBD845B7AE822AE2D4F7BC8D7103752651D264D0022CF1251564DF72B3F463E24B18F1B792DF1ED178B2525AA60402EA8409CFFE730F482634F4E1879D7AEDCB066868409E83AAD1A0739575A36; _ga=GA1.3.968099383.1752635290; _gid=GA1.3.923925777.1752807067; _ga_JCXT8BPG4E=GS2.3.s1752843096$o5$g0$t1752843096$j60$l0$h0"
}


# --- HÀM MỚI: TÌM CÁC CAMERA GẦN NHẤT ---
def get_nearest_cameras(df, ref_point, count):
    """
    Tính khoảng cách từ mỗi camera đến điểm tham chiếu, sắp xếp và trả về
    danh sách ID của 'count' camera gần nhất.
    """
    try:
        # Hàm để áp dụng cho mỗi dòng trong DataFrame
        def calculate_distance(row):
            camera_point = (row['Vĩ độ'], row['Kinh độ'])
            return haversine(ref_point, camera_point, unit=Unit.KILOMETERS)

        # Áp dụng hàm tính khoảng cách để tạo một cột mới
        df['distance_km'] = df.apply(calculate_distance, axis=1)

        # Sắp xếp DataFrame dựa trên khoảng cách, từ nhỏ đến lớn
        df_sorted = df.sort_values(by='distance_km')

        # Lấy 'count' camera gần nhất
        nearest_df = df_sorted.head(count)

        # In ra thông tin hữu ích
        if not nearest_df.empty:
            max_distance = nearest_df['distance_km'].max()
            print(f"Đã xác định được {len(nearest_df)} camera gần nhất.")
            print(f"Camera xa nhất trong nhóm này cách điểm tham chiếu {max_distance:.2f} km.")

        # Trả về danh sách các ID
        return nearest_df['CamId (Metadata)'].astype(str).tolist()

    except KeyError as e:
        print(f"[LỖI CHÍ MẠNG] Không tìm thấy cột cần thiết trong file Excel: {e}.")
        print("Vui lòng đảm bảo file Excel có các cột 'Vĩ độ', 'Kinh độ', và 'CamId (Metadata)'.")
        return None
    except Exception as e:
        print(f"[LỖI CHÍ MẠNG] Xảy ra lỗi khi xử lý dữ liệu camera: {e}")
        return None


# --- CÁC HÀM TẢI ẢNH (Giữ nguyên) ---
async def download_and_save_image(session, cam_id, batch_folder):
    url = CONFIG["CAMERA_URL_TEMPLATE"].format(cam_id) + f"&t={time.time()}"
    output_path = os.path.join(batch_folder, f"{cam_id}.jpg")
    try:
        async with session.get(url, ssl=False, timeout=30, headers=HEADERS) as response:
            if response.status == 200:
                async with aiofiles.open(output_path, mode='wb') as f:
                    await f.write(await response.read())
            else:
                print(f"  [Lỗi] Status {response.status} cho camera {cam_id}")
    except Exception as e:
        print(f"  [Lỗi] Ngoại lệ khi tải camera {cam_id}: {e}")


# --- THAY ĐỔI: Hàm `run_download_batch` giờ nhận danh sách ID camera để tải ---
async def run_download_batch(camera_ids_to_download):
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    batch_folder = os.path.join(CONFIG["BASE_OUTPUT_FOLDER"], timestamp_str)
    os.makedirs(batch_folder, exist_ok=True)

    print(f"\n--- Bắt đầu chu kỳ tải lúc: {timestamp_str} ---")
    print(f"Đang lưu ảnh cho {len(camera_ids_to_download)} camera vào thư mục: {batch_folder}")

    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_and_save_image(session, cam_id, batch_folder) for cam_id in camera_ids_to_download]
        await asyncio.gather(*tasks)

    print(f"--- Hoàn thành chu kỳ lúc: {datetime.now().strftime('%H:%M:%S')} ---")


# --- THAY ĐỔI: Vòng lặp chính sẽ chuẩn bị dữ liệu trước khi chạy ---
async def main():
    print("=" * 50)
    print("CHƯƠNG TRÌNH TẢI ẢNH CAMERA LIÊN TỤC")

    # BƯỚC CHUẨN BỊ DỮ LIỆU
    print("\n--- Bước 1: Chuẩn bị danh sách camera ---")
    try:
        full_df = pd.read_excel(CONFIG["EXCEL_FILE"], sheet_name=CONFIG["SHEET_NAME"])
        print(f"Đã đọc thành công {len(full_df)} camera từ file Excel.")
    except Exception as e:
        print(f"[LỖI CHÍ MẠNG] Không thể đọc file Excel '{CONFIG['EXCEL_FILE']}'. Lỗi: {e}")
        return

    # Lấy danh sách 200 camera gần nhất
    ref_point = CONFIG["REFERENCE_POINT"]
    count = CONFIG["NEAREST_CAMERA_COUNT"]
    print(f"Đang tìm {count} camera gần nhất với điểm tham chiếu: (Lat: {ref_point[0]}, Lon: {ref_point[1]})")

    # Lấy danh sách ID camera cần theo dõi
    target_camera_ids = get_nearest_cameras(full_df, ref_point, count)

    if not target_camera_ids:
        print("Không thể xác định danh sách camera. Chương trình sẽ dừng lại.")
        return

    # BƯỚC CHẠY VÒNG LẶP
    print("\n--- Bước 2: Bắt đầu vòng lặp tải ảnh ---")
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=CONFIG["RUN_DURATION_HOURS"])
    interval_seconds = CONFIG["DOWNLOAD_INTERVAL_SECONDS"]

    print(f"Cấu trúc lưu trữ: captured_images / [NGAY_GIO] / [ID_CAMERA].jpg")
    print(f"Bắt đầu lúc: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dự kiến kết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tần suất tải: Mỗi {interval_seconds} GIÂY")
    print("Nhấn Ctrl+C để dừng chương trình bất cứ lúc nào.")
    print("=" * 50)

    try:
        while datetime.now() < end_time:
            loop_start_time = time.time()
            # Truyền danh sách camera đã được lọc vào hàm tải
            await run_download_batch(target_camera_ids)

            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            sleep_time = max(0, interval_seconds - elapsed_time)

            if datetime.now() >= end_time:
                break

            print(
                f"Đã hoàn tất chu kỳ trong {elapsed_time:.2f} giây. Chờ {sleep_time:.2f} giây cho chu kỳ tiếp theo...")
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        print("\nChương trình đã được dừng bởi người dùng (Ctrl+C).")
    finally:
        print("\n--- Chương trình đã kết thúc. ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nĐã nhận lệnh dừng. Đang tắt chương trình...")