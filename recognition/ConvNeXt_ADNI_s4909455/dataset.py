import torch
import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class ADNIDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, validation_split=0.1):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.validation_split = validation_split
        
        metadata_path = os.path.join(data_dir, 'meta_data_with_label.json')
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {} # Proceed without metadata if file not found
        
        # Load file paths and labels
        self.data = self.load_data_paths()

    def load_data_paths(self):
        """Load file paths and labels"""
        all_data = []
        
        if self.split in ['train', 'validation']:
            train_path = os.path.join(self.data_dir, 'AD_NC', 'train')
            all_data.extend(self.get_paths_from_folder(train_path, 'AD', label=1))
            all_data.extend(self.get_paths_from_folder(train_path, 'NC', label=0))
            
            if len(all_data) > 0:
                train_data, val_data = train_test_split(
                    all_data, 
                    test_size=self.validation_split, 
                    random_state=42,
                    stratify=[item[1] for item in all_data]
                )
                return train_data if self.split == 'train' else val_data
            else:
                return [] # Return empty list if no data
        
        elif self.split == 'test':
            test_path = os.path.join(self.data_dir, 'AD_NC', 'test')
            all_data.extend(self.get_paths_from_folder(test_path, 'AD', label=1))
            all_data.extend(self.get_paths_from_folder(test_path, 'NC', label=0))
        
        return all_data

    def get_paths_from_folder(self, base_path, class_name, label):
        """Get list of (image_path, label) tuples"""
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            return []
        
        data = []
        for image_name in os.listdir(class_path):
            if image_name.endswith('.jpeg'):
                subject_id = image_name.split('_')[0]
                
                # Load if metadata is missing OR if subject is in metadata
                if not self.metadata or subject_id in self.metadata:
                    image_path = os.path.join(class_path, image_name)
                    data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    DATA_ROOT = "ADNI" 
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data directory not found at {DATA_ROOT}")
    else:
        print("Testing Dataset Initialization...")
        train_dataset = ADNIDataset(DATA_ROOT, split='train')
        val_dataset = ADNIDataset(DATA_ROOT, split='validation')
        test_dataset = ADNIDataset(DATA_ROOT, split='test')
        
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Validation dataset length: {len(val_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")