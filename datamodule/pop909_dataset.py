import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pretty_midi
from schema.dataset_schema import Pop909DatasetSchema, Pop909DataModuleSchema


class Pop909Dataset(Dataset):
    def __init__(self, config: Pop909DatasetSchema):
        self.root_dir = config.root_dir
        self.transform = config.transform
        self.midi_files = []

        # Collect all MIDI files
        for i in range(1, 910):  # POP909 has 909 songs, numbered 001 to 909
            midi_path = os.path.join(self.root_dir, f"{i:03d}", f"{i:03d}.mid")
            if os.path.exists(midi_path):
                self.midi_files.append(midi_path)
            else:
                # For debugging, if some files are missing
                print(f"Warning: MIDI file not found at {midi_path}")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            if self.transform:
                midi_data = self.transform(midi_data)
            return midi_data
        except Exception as e:
            print(f"Error loading MIDI file {midi_path}: {e}")
            # Return None or handle the error appropriately
            return None


class Pop909DataModule(pl.LightningDataModule):
    def __init__(self, config: Pop909DataModuleSchema):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_dataset_config = config.train_dataset_config
        self.val_dataset_config = config.val_dataset_config
        self.test_dataset_config = config.test_dataset_config
        self.predict_dataset_config = config.predict_dataset_config

    def setup(self, stage=None):
        if stage == 'fit' :
            # Setup for training and validation datasets
            if self.train_dataset_config:
                self.train_dataset = Pop909Dataset(self.train_dataset_config)
            if self.val_dataset_config:
                self.val_dataset = Pop909Dataset(self.val_dataset_config)
        elif stage == 'test':
            # Setup for test dataset
            if self.test_dataset_config:
                self.test_dataset = Pop909Dataset(self.test_dataset_config)
        elif stage == 'predict':
            # Setup for prediction dataset
            if self.predict_dataset_config:
                self.predict_dataset = Pop909Dataset(self.predict_dataset_config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        # For prediction, you might want to use the full dataset or a specific subset
        # For now, returning the test_dataloader for demonstration
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
