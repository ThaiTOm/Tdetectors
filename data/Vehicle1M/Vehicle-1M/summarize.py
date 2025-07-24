import os

def summarize_train_list(file_path):
    """
    Summarizes the content of a train_list.txt file, counting:
    - Total number of images
    - Total number of unique IDs in the second column
    - Total number of unique IDs in the third column

    Args:
        file_path (str): The path to the train_list.txt file.

    Returns:
        dict: A dictionary containing the summary:
              - 'total_images': int
              - 'total_unique_id_col2': int
              - 'total_unique_id_col3': int
              - 'error': str (if an error occurred)
    """
    total_images = 0
    unique_ids_col2 = set()
    unique_ids_col3 = set()
    errors = []

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                total_images += 1
                parts = line.strip().split()

                if len(parts) >= 3:
                    try:
                        id_col2 = int(parts[1])
                        unique_ids_col2.add(id_col2)
                    except ValueError:
                        errors.append(f"Warning: Could not parse ID in second column for line {line_num}: '{parts[1]}'")
                    try:
                        id_col3 = int(parts[2])
                        unique_ids_col3.add(id_col3)
                    except ValueError:
                        errors.append(f"Warning: Could not parse ID in third column for line {line_num}: '{parts[2]}'")
                else:
                    errors.append(f"Warning: Line {line_num} does not have at least 3 columns: '{line.strip()}'")

    except FileNotFoundError:
        return {"error": f"Error: File not found at '{file_path}'"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

    summary = {
        "total_images": total_images,
        "total_unique_id_col2": len(unique_ids_col2),
        "total_unique_id_col3": len(unique_ids_col3),
    }
    if errors:
        summary["warnings"] = errors
    return summary

# Example usage based on your provided context:
if __name__ == "__main__":
    # Simulate the file path
    # In a real scenario, you would replace this with the actual path
    # (TDetectors) geso@ai01svr:~/Tdetectors/data/Vehicle1M/Vehicle-1M/train-test-split$
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Get current script directory
    # Adjust this path if your script is not in the 'train-test-split' directory
    # For demonstration, let's create a dummy file
    dummy_file_path = os.path.join(current_dir, "train-test-split/train_list.txt")

    print(f"Analyzing file: {dummy_file_path}")
    summary = summarize_train_list(dummy_file_path)

    if "error" in summary:
        print(summary["error"])
    else:
        print("\n--- Summary ---")
        print(f"Total Images: {summary['total_images']}")
        print(f"Total Unique IDs (second column): {summary['total_unique_id_col2']}")
        print(f"Total Unique IDs (third column): {summary['total_unique_id_col3']}")
        if "warnings" in summary:
            print("\n--- Warnings ---")
            for warning in summary["warnings"]:
                print(warning)
