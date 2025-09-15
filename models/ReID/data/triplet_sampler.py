import collections
import copy
import random
from pathlib import Path
from typing import Dict, List

import cv2
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import os
import torchvision
from PIL import Image # <--- ADD THIS IMPORT
import torchvision.io
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET  # Import the XML parsing module


def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids 

        
class CustomDataSet4VERIWILD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = self.data_info.iloc[idx, 2]
        
        view_id = 0 #self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view :
            return img, vid, camid, view_id
        else:
            return img, vid, camid, 0





class CustomDataSet4VERIWILDv2(Dataset):
    """VeriWild 2.0 dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = 0 #self.data_info.iloc[idx, 2]
        view_id = 0 # = self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view:
            return img, vid, camid, view_id
        else:
            return img, vid, camid



class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in range(len(self.data_source.data_info)):
            pid = self.data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

        
class CustomDataSet4Market1501(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])  
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].split('s')[0].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid     

       
 


class CustomDataSet4Veri776(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])     
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
                # print(pid, id, 'debug')
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0 






class CustomDataSet4Veri776_withviewpont(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, viewpoint_train, viewpoint_test, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.viewpoint_train = pd.read_csv(viewpoint_train, sep=' ', header = None)
        self.viewpoint_test = pd.read_csv(viewpoint_test, sep=' ', header = None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        self.view = []
        conta_missing_images = 0
        if is_train == True:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line]
                if self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line].shape[0] ==0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1]) 
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line]
                if self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line].shape[0] == 0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform
        print('Missed viewpoint for ', conta_missing_images, ' images!')
    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))-1
        viewid = np.int64(self.view[idx])


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, viewid     

class CustomDataSet4VehicleID_Random(Dataset):
    def __init__(self, lines, root_dir, is_train=True, mode=None, transform=None, teste=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = []
        self.names = []
        self.labels = []
        self.teste = teste
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        ### no camera information
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.teste:
            return img, vid, camid, 0
        else:
            return img, vid, camid



class CustomDataSet4VehicleID(Dataset):
    def __init__(self, image_list, root_dir, is_train=True, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0



# import random # Only needed if you plan to use random.shuffle, which was commented out

class Dataset4VehicleID(Dataset):
    def __init__(self, image_list, root_dir, is_train=True, mode=None, transform=None):
        """
        Args:
            image_list (string): Path to the XML file with annotations (e.g., 'train_label.xml').
            root_dir (string): Directory with all the images.
            is_train (bool): True for training mode, False for aiCity22/inference mode.
            mode (string, optional): 'g' for gallery in aiCity22 mode, otherwise query.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.names = []
        self.labels = []
        self.cams = []

        # --- XML Parsing ---
        try:
            # 1. Open the XML file with the specified encoding 'gb2312'
            #    and read its entire content into a string.
            with open(image_list, 'r', encoding='gb2312') as f:
                xml_string = f.read()

            # 2. Parse the XML from the string content using ET.fromstring()
            #    This bypasses the underlying C parser's direct multi-byte encoding limitation
            root = ET.fromstring(xml_string)

            for item_elem in root.findall('.//Item'):
                image_name_with_ext = item_elem.get('imageName')
                vehicle_id = item_elem.get('vehicleID')
                camera_id = item_elem.get('cameraID')

                if image_name_with_ext is None or vehicle_id is None or camera_id is None:
                    print(f"Warning: Skipping item with missing attributes: {item_elem.attrib}")
                    continue

                self.names.append(image_name_with_ext.replace('.jpg', ''))
                self.labels.append(vehicle_id)
                self.cams.append(camera_id)
        except ET.ParseError as e:
            print(f"Error parsing XML file {image_list}: {e}")
            self.names = []
            self.labels = []
            self.cams = []
        except FileNotFoundError:
            print(f"Error: XML file not found at {image_list}")
            self.names = []
            self.labels = []
            self.cams = []
        except UnicodeDecodeError as e:
            print(f"Error decoding XML file {image_list} with gb2312 encoding. Please check file encoding: {e}")
            self.names = []
            self.labels = []
            self.cams = []
        # --- End XML Parsing ---

        if is_train:
            unique_labels = sorted(list(set(self.labels)))
            print(f"Number of unique vehicle IDs (classes) for training: {len(unique_labels)}")

            label_to_pid = {label: pid for pid, label in enumerate(unique_labels)}
            self.labels = [label_to_pid[label] for label in self.labels]
        else:
            print(f"Dataload Test mode: {mode}")

            temp_names = []
            temp_labels = []
            temp_cams = []

            vid_container = set()

            for i in range(len(self.names)):
                current_name = self.names[i]
                current_vid = self.labels[i]
                current_cam = self.cams[i]

                if mode == 'g':
                    if current_vid not in vid_container:
                        vid_container.add(current_vid)
                        temp_names.append(current_name)
                        temp_labels.append(current_vid)
                        temp_cams.append(current_cam)
                else:
                    if current_vid in vid_container:
                        temp_names.append(current_name)
                        temp_labels.append(current_vid)
                        temp_cams.append(current_cam)
                    else:
                        vid_container.add(current_vid)

            self.names = temp_names
            self.labels = temp_labels
            self.cams = temp_cams
            
        self.data_info = self.names

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_full_path = os.path.join(self.root_dir, self.names[idx] + ".jpg")

        image = torchvision.io.read_image(img_name_full_path)

        vid = np.int64(self.labels[idx])

        camid_str = self.cams[idx]
        camid = np.int64(camid_str.replace('c', ''))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor)) / 255.0)
        else:
            img = (image.type(torch.FloatTensor)) / 255.0

        return img, vid, camid, 0



# Note: The xml.etree.ElementTree import is no longer needed.

class VRICDataset(Dataset):
    """
    Custom PyTorch Dataset for the Vehicle Re-Identification in Context (VRIC) dataset.
    This class reads annotation files in the format: '[Image_name] [ID label] [Cam Label]'
    """

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            image_list (string): Path to the annotation file (e.g., 'vric_train.txt').
            root_dir (string): Directory with all the corresponding images (e.g., 'train_images/').
            is_train (bool): True for training mode, False for aiCity22/inference mode.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        self.names = []
        self.labels = []
        self.cams = []

        # --- Text File Parsing for VRIC Dataset ---
        try:
            with open(image_list, 'r') as f:
                for line in f:
                    # Skip empty lines
                    if not line.strip():
                        continue

                    # The format is: [Image_name] [ID label] [Cam Label]
                    # Example line: 'image_0001.jpg 1 2'
                    image_name, vehicle_id, camera_id = line.strip().split()

                    self.names.append(image_name)
                    self.labels.append(vehicle_id)
                    self.cams.append(camera_id)

        except FileNotFoundError:
            print(f"Error: Annotation file not found at {image_list}")
            # Ensure lists are empty if file not found
            self.names, self.labels, self.cams = [], [], []
        except Exception as e:
            print(f"An error occurred while reading {image_list}: {e}")
            self.names, self.labels, self.cams = [], [], []
        # --- End Text File Parsing ---

        if self.is_train:
            # For training, map string labels to integer process IDs (PIDs)
            # This is standard practice for classification tasks.
            unique_labels = sorted(list(set(self.labels)))
            print(f"Number of unique vehicle IDs (classes) for training: {len(unique_labels)}")

            # Create a mapping from the original string ID to a new 0-based integer ID
            self.label_to_pid = {label: pid for pid, label in enumerate(unique_labels)}
            self.pids = [self.label_to_pid[label] for label in self.labels]
        else:
            # For testing (probe/gallery), we use the original labels directly
            # for evaluation (mAP, CMC), so no re-mapping is needed.
            print(f"Loading aiCity22 data from: {os.path.basename(image_list)}")
            self.pids = self.labels  # Keep original IDs

        self.data_info = self.names

    def get_class(self, idx):
        """Returns the process ID (PID) for a given index."""
        return self.pids[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct the full path to the image
        img_name = self.names[idx]
        img_name_full_path = os.path.join(self.root_dir, img_name)

        try:
            image = torchvision.io.read_image(img_name_full_path)
        except Exception as e:
            print(f"Error loading image: {img_name_full_path}. {e}")
            # Return a placeholder tensor if image is corrupt or missing
            return torch.zeros(3, 256, 128), -1, -1, 0

        # Get the process ID (PID)
        pid = np.int64(self.pids[idx])

        # Get the camera ID
        camid_str = self.cams[idx]
        camid = np.int64(camid_str)  # Cam IDs in VRIC are just numbers

        if self.transform:
            # Convert image to float tensor in range [0, 1] before transform
            img = self.transform((image.type(torch.FloatTensor)) / 255.0)
        else:
            img = (image.type(torch.FloatTensor)) / 255.0

        # The last value '0' is a placeholder for tracklet ID, which is not used here.
        return img, pid, camid, 0



# Ensure all necessary imports are present
# from PIL import Image # Potentially needed if torchvision.io fails
class CombinedVehicleDataset(Dataset):
    """
    A combined PyTorch Dataset for the Thai VehicleID, VRIC, Vehicle-1M,
    and the AICity22 Challenge (train split) datasets.

    This class loads data from all four sources, processes the AICity22 videos
    to extract and save vehicle crops, ensures vehicle IDs are unique across
    datasets by prefixing them, and creates a unified set of process IDs (PIDs)
    for training.
    """

    def __init__(self,
                 # Paths for the first three datasets
                 vehicleID_xml_path="data/VehicleID/VeRi/train_label.xml",
                 vehicleID_root_dir="data/VehicleID/VeRi/image_train",
                 vric_txt_path="data/VRIC/vric_train.txt",
                 vric_root_dir="data/VRIC/train_images",
                 vehicle1M_txt_path="data/Vehicle-1M/train-aiCity22-split/train_list.txt",
                 vehicle1M_root_dir="data/Vehicle-1M/image",
                 # Paths for the new AICity22 dataset
                 aicity_root_path="data/aiCity22",
                 aicity_crop_dir:str="data/aiCity22/crops",
                 aicity_split: str = 'train',
                 is_train=True,
                 transform=None):
        """
        Args:
            vehicleID_xml_path (str, optional): Path to the Thai dataset's XML file.
            vehicleID_root_dir (str, optional): Root directory for the Thai dataset's images.
            vric_txt_path (str, optional): Path to the VRIC dataset's text annotation file.
            vric_root_dir (str, optional): Root directory for the VRIC dataset's images.
            vehicle1M_txt_path (str, optional): Path to the Vehicle-1M dataset's text file.
            vehicle1M_root_dir (str, optional): Root directory for the Vehicle-1M dataset's images.
            aicity_root_path (str, optional): Path to the root of the AICity22 dataset.
            aicity_crop_dir (str, optional): Directory to save/load extracted vehicle crops from AICity22.
            is_train (bool): Set to True for training mode. Creates PIDs for classification.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.is_train = is_train

        # Load data from each source
        vehicleID_data = self._load_vehicleID_data(vehicleID_xml_path, vehicleID_root_dir)
        vric_data = self._load_vric_data(vric_txt_path, vric_root_dir)
        vehicle1M_data = self._load_vehicle1M_data(vehicle1M_txt_path, vehicle1M_root_dir)
        aicity_data = self._load_aicity_data(aicity_root_path, aicity_crop_dir, aicity_split)

        # Combine the data
        all_image_paths = (vehicleID_data['paths'] +
                           vric_data['paths'] +
                           vehicle1M_data['paths'] +
                           aicity_data['paths'])

        all_labels = ([f"vehicleid_{l}" for l in vehicleID_data['labels']] +
                      [f"vric_{l}" for l in vric_data['labels']] +
                      [f"vehicle1m_{l}" for l in vehicle1M_data['labels']] +
                      [f"aicity_{l}" for l in aicity_data['labels']])

        all_cams = (vehicleID_data['cams'] +
                    vric_data['cams'] +
                    vehicle1M_data['cams'] +
                    aicity_data['cams'])

        print("-" * 40)
        print(f"Loaded {len(vehicleID_data['paths'])} images from VehicleID.")
        print(f"Loaded {len(vric_data['paths'])} images from VRIC.")
        print(f"Loaded {len(vehicle1M_data['paths'])} images from Vehicle-1M.")
        print(f"Loaded {len(aicity_data['paths'])} images from AICity22.")
        print(f"Total images in combined dataset: {len(all_image_paths)}")
        print("-" * 40)

        self.image_paths = all_image_paths
        self.cams = all_cams

        if self.is_train:
            unique_labels = sorted(list(set(all_labels)))
            self.label_to_pid = {label: pid for pid, label in enumerate(unique_labels)}
            self.pids = [self.label_to_pid[label] for label in all_labels]
            num_classes = len(unique_labels)
            print(f"Total number of unique vehicle IDs (classes) for training: {num_classes}")
        else:
            self.pids = all_labels

        self.data_info = self.image_paths

    def _load_vehicleID_data(self, xml_path, root_dir):
        """Helper to load data from the Thai VehicleID XML file."""
        paths, labels, cams = [], [], []
        if not xml_path or not root_dir: return {'paths': [], 'labels': [], 'cams': []}
        try:
            with open(xml_path, 'r', encoding='gb2312') as f:
                root = ET.fromstring(f.read())
            for item in root.findall('.//Item'):
                paths.append(os.path.join(root_dir, item.get('imageName')))
                labels.append(item.get('vehicleID'))
                cams.append(int(item.get('cameraID').replace('c', '')))
        except Exception as e:
            print(f"Error loading VehicleID dataset from {xml_path}: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_vric_data(self, txt_path, root_dir):
        """Helper to load data from the VRIC text file."""
        paths, labels, cams = [], [], []
        if not txt_path or not root_dir: return {'paths': [], 'labels': [], 'cams': []}
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    name, v_id, c_id = line.strip().split()
                    paths.append(os.path.join(root_dir, name))
                    labels.append(v_id)
                    cams.append(int(c_id))
        except Exception as e:
            print(f"Error loading VRIC dataset from {txt_path}: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_vehicle1M_data(self, txt_path, root_dir):
        """Helper to load data from the Vehicle-1M text file."""
        paths, labels, cams = [], [], []
        if not txt_path or not root_dir: return {'paths': [], 'labels': [], 'cams': []}
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    img_path, v_id, c_id = parts[0], parts[1], parts[2]
                    full_path = os.path.join(root_dir, img_path)
                    if os.path.exists(full_path):
                        paths.append(full_path)
                        labels.append(v_id)
                        cams.append(int(c_id))
        except Exception as e:
            print(f"Error loading Vehicle-1M dataset from {txt_path}: {e}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    def _load_aicity_data(self, root_dir: str, crop_dir: str, split: str) -> Dict:
        """
        Helper to load data from the AICity22 Challenge dataset.
        It extracts vehicle crops from video files and saves them to disk for caching.
        """
        root_path = Path(root_dir)
        crop_path = Path(crop_dir)
        paths, labels, cams = [], [], []

        # FAST PATH: Load pre-extracted crops if they exist
        if crop_path.exists() and any(crop_path.iterdir()):
            print(f"INFO: Loading pre-extracted AICity crops from cache: {crop_path}")
            for img_file in sorted(list(crop_path.glob("*.jpg"))):
                try:
                    # Filename format: c<cam_id>_v<vehicle_id>_f<frame_num>.jpg
                    parts = img_file.stem.split('_')
                    cam_id_str = parts[0]
                    vehicle_id_str = parts[1]

                    # Ensure the filename matches the expected split by checking the camera ID
                    # (This is a simplified check, assuming camera IDs don't overlap between splits)
                    paths.append(str(img_file))
                    labels.append(vehicle_id_str.replace('v', ''))
                    cams.append(int(cam_id_str.replace('c', '')))
                except (ValueError, IndexError):
                    print(f"WARNING: Skipping malformed filename in cache: {img_file.name}")

            if paths:
                return {'paths': paths, 'labels': labels, 'cams': cams}

        # SLOW PATH: Extract crops from videos for the first time
        print(f"INFO: Cache directory '{crop_path}' is empty or not found.")
        print("      Proceeding with SLOW PATH: extracting vehicle crops from videos.")
        print("      This is a one-time process and may take a very long time.")
        crop_path.mkdir(parents=True, exist_ok=True)
        print(f"      Created cache directory at: {crop_path.resolve()}")

        vehicle_data = self._group_aicity_vehicle_ids(root_path, split=split)
        total_sightings = sum(len(s) for s in vehicle_data.values())
        processed_count = 0
        video_caps = {}  # Cache open video captures to avoid re-opening files

        try:
            for vehicle_id, sightings in vehicle_data.items():
                for sighting in sightings:
                    video_file = sighting['video_path']
                    frame_num, bbox = sighting['frame_num'], sighting['bbox']

                    try:
                        cam_name = Path(video_file).parent.name
                        cam_id = int(cam_name.replace('c', ''))
                    except (ValueError, IndexError):
                        continue

                    if video_file not in video_caps:
                        cap = cv2.VideoCapture(video_file)
                        if not cap.isOpened(): continue
                        video_caps[video_file] = cap

                    cap = video_caps[video_file]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                    ret, frame = cap.read()

                    if ret:
                        x, y, w, h = bbox
                        # Ensure bounding box coordinates are valid
                        if w <= 0 or h <= 0: continue
                        crop = frame[max(0, y):y + h, max(0, x):x + w]
                        if crop.size == 0: continue

                        # Create a unique filename for the crop
                        filename = f"c{cam_id:03d}_v{vehicle_id:05d}_f{frame_num:06d}.jpg"
                        save_path = crop_path / filename
                        cv2.imwrite(str(save_path), crop)

                        paths.append(str(save_path))
                        labels.append(str(vehicle_id))
                        cams.append(cam_id)

                    processed_count += 1
                    if processed_count % 2000 == 0:
                        print(f"      ... processed {processed_count} / {total_sightings} sightings.")
        finally:
            # Important: release all video captures
            for cap in video_caps.values():
                cap.release()

        print(f"INFO: AICity crop extraction complete. Saved {len(paths)} images to {crop_path}")
        return {'paths': paths, 'labels': labels, 'cams': cams}

    @staticmethod
    def _group_aicity_vehicle_ids(dataset_path: Path, split: str) -> Dict[int, List[Dict]]:
        """Parses the AIC22 dataset to group all vehicle sightings by their unique ID for a given split."""
        list_cam_path = dataset_path / 'list_cam.txt'
        if not list_cam_path.exists():
            raise FileNotFoundError(f"Required file 'list_cam.txt' not found in {dataset_path}")

        cam_paths = []
        with open(list_cam_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(f'./{split}/'):
                    cam_paths.append(dataset_path / line.lstrip('./'))

        if not cam_paths:
            print(f"WARNING: No camera paths found for split '{split}' in {list_cam_path}.")
            return {}

        vehicle_data = collections.defaultdict(list)
        for cam_path in cam_paths:
            gt_path = cam_path / 'gt/gt.txt'
            video_path = cam_path / 'vdo.avi'
            if not gt_path.exists():
                # print(f"DEBUG: gt.txt not found for {cam_path}, skipping.")
                continue
            if not video_path.exists():
                # print(f"DEBUG: vdo.avi not found for {cam_path}, skipping.")
                continue

            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    try:
                        frame, v_id, x, y, w, h = map(int, parts[:6])
                        if v_id == -1: continue  # Ignore placeholder IDs
                        vehicle_data[v_id].append({
                            'video_path': str(video_path),
                            'frame_num': frame,
                            'bbox': [x, y, w, h]
                        })
                    except ValueError:
                        continue  # Skip malformed lines

        print(f"INFO: Found {len(vehicle_data)} unique vehicle IDs in '{split}' split from GT files.")
        return dict(vehicle_data)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pid = self.pids[idx]
        camid = self.cams[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_path}. {e}. Returning a placeholder.")
            # Return a placeholder tensor if image is corrupt
            placeholder_tensor = torch.zeros(3, 256, 128)
            if self.transform:
                placeholder_tensor = self.transform(placeholder_tensor)
            return placeholder_tensor, -1, -1, 0

        if self.transform:
            image = self.transform(image)

        pid_val = np.int64(pid) if self.is_train else pid
        camid_val = np.int64(camid)

        # The '0' is often a placeholder for tracklet ID, not used here
        return image, pid_val, camid_val, 0