import os
from datetime import datetime

# --- CONFIGURATION ---
# This should match the BASE_OUTPUT_FOLDER from your capture script.
BASE_FOLDER = 'captured_images'
TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M-%S'


def format_timedelta(td):
    """Formats a timedelta object into a human-readable string."""
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:  # Show seconds if it's the only unit
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return ', '.join(parts)


def summarize_capture_folder(base_path):
    """
    Analyzes the capture folder to provide a summary of the collected data.
    """
    print("=" * 60)
    print("--- DATA CAPTURE SUMMARY REPORT ---")
    print(f"Analyzing folder: '{base_path}'")
    print("=" * 60)

    if not os.path.exists(base_path):
        print(f"[ERROR] The directory '{base_path}' does not exist.")
        print("Please make sure you have run the capture script first.")
        return

    all_timestamps = []
    total_images = 0
    total_capture_cycles = 0
    camera_ids = set()

    # List all subdirectories which represent capture cycles
    try:
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    except Exception as e:
        print(f"[ERROR] Could not read subdirectories in '{base_path}': {e}")
        return

    for folder_name in subfolders:
        try:
            # Parse the folder name to get the timestamp
            ts = datetime.strptime(folder_name, TIMESTAMP_FORMAT)
            all_timestamps.append(ts)
            total_capture_cycles += 1

            # Count images and collect camera IDs within this cycle's folder
            current_cycle_path = os.path.join(base_path, folder_name)
            images_in_cycle = [f for f in os.listdir(current_cycle_path) if f.lower().endswith('.jpg')]

            total_images += len(images_in_cycle)

            # Extract camera IDs from filenames (e.g., "1234.jpg" -> "1234")
            for img_file in images_in_cycle:
                cam_id = os.path.splitext(img_file)[0]
                camera_ids.add(cam_id)

        except ValueError:
            # This folder name doesn't match our timestamp format, so we skip it.
            print(f"[Warning] Skipping non-timestamp folder: '{folder_name}'")
            continue
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while processing '{folder_name}': {e}")

    # --- Generate and Print the Summary ---
    if not all_timestamps:
        print("\n[RESULT] No valid capture data found in the directory.")
        print("The folder might be empty or contain incorrectly named subfolders.")
        return

    all_timestamps.sort()
    first_capture_time = all_timestamps[0]
    last_capture_time = all_timestamps[-1]
    total_duration = last_capture_time - first_capture_time

    print("\n--- SUMMARY ---")
    print(f"Total Capture Cycles (Folders): {total_capture_cycles}")
    print(f"Total Images Captured:          {total_images}")
    print(f"Number of Unique Cameras Seen:  {len(camera_ids)}")

    if total_capture_cycles > 0:
        avg_images_per_cycle = total_images / total_capture_cycles
        print(f"Average Images per Cycle:       {avg_images_per_cycle:.2f}")

    print("\n--- TIME FRAME ---")
    print(f"First Capture Batch: {first_capture_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Last Capture Batch:  {last_capture_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Recording Span: {format_timedelta(total_duration)}")
    print("-" * 60)


if __name__ == "__main__":
    summarize_capture_folder(BASE_FOLDER)