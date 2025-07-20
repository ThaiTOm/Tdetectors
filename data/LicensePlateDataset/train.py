import os
import sys
from omegaconf import OmegaConf
from roboflow import Roboflow
from tqdm import tqdm
# Determine the parent directory
parent_dir = os.getenv("ROOT_PATH", r"/home/geso/Tdetectors")

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
# Load the global configuration file
from data import *

if __name__ == '__main__':
    """
    This block parses command-line arguments and calls the main training function.
    This allows you to run the script with different configurations without changing the code.

    Example Usage:

    1. Basic Training:
       python train_yolo.py --data /path/to/your/data.yaml

    2. Training with more options and exporting the model:
       python train_yolo.py --model yolov8m.pt --data /path/to/data.yaml --epochs 150 --batch 8 --name yolov8m_custom_run --export-onnx

    3. Resuming training from a checkpoint:
       python train_yolo.py --model /path/to/runs/train/exp/weights/last.pt --data /path/to/data.yaml
    """
    parser = argparse.ArgumentParser(description="YOLOv12 Training Script")

    # Required arguments
    parser.add_argument('--data', type=str, help='Path to the dataset .yaml file',
                        default=os.path.join(parent_dir, "data/LicensePlateDataset/License-Plate-1/data.yaml"))

    # Optional arguments with default values
    parser.add_argument('--model', type=str, default='yolo12n.pt',
                        help='Starting model path (e.g., yolov8n.pt) or path to a checkpoint for resuming')

    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training (adjust based on VRAM)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training (e.g., 640 for 640x640)')
    parser.add_argument('--project', type=str, default='runs/train', help='Directory to save training runs')
    parser.add_argument('--name', type=str, default='exp', help='Name for the specific training run folder')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export the best model to ONNX format after training')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main training function
    train_model(args)