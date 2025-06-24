import pytest
import os
import json
from unittest.mock import MagicMock, patch, mock_open
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Adjust the import path based on your project structure
from datamodules.remi_json_datamodule import Pop909Dataset, Pop909DataModule
from schema.dataset_schema import Pop909DatasetSchema, Pop909DataModuleSchema

# Mock the schema classes by inheriting directly from them
# Pydantic models handle their own initialization, so custom __init__ methods are not needed here.
class MockPop909DatasetSchema(Pop909DatasetSchema):
    pass

class MockPop909DataModuleSchema(Pop909DataModuleSchema):
    pass

@pytest.fixture
def mock_dataset_config():
    return MockPop909DatasetSchema(mel_dir="mock_mel_root", acc_dir="mock_acc_root")

@pytest.fixture
def mock_datamodule_config():
    return MockPop909DataModuleSchema(
        batch_size=2,
        num_workers=0,
        train_dataset_config=MockPop909DatasetSchema(mel_dir="mock_mel_root", acc_dir="mock_acc_root", transform=None),
        val_dataset_config=MockPop909DatasetSchema(mel_dir="mock_mel_root", acc_dir="mock_acc_root", transform=None),
        test_dataset_config=MockPop909DatasetSchema(mel_dir="mock_mel_root", acc_dir="mock_acc_root", transform=None),
        predict_dataset_config=MockPop909DatasetSchema(mel_dir="mock_mel_root", acc_dir="mock_acc_root", transform=None)
    )

# Test Pop909Dataset
@patch('os.path.exists', side_effect=lambda path: "001.json" in path or "002.json" in path)
def test_pop909_dataset_init(mock_exists, mock_dataset_config):
    dataset = Pop909Dataset(mock_dataset_config)
    assert len(dataset.midi_file_pairs) == 2 # Assuming 001.json and 002.json exist
    assert ("mock_mel_root/001.json", "mock_acc_root/001.json") in dataset.midi_file_pairs
    assert ("mock_mel_root/002.json", "mock_acc_root/002.json") in dataset.midi_file_pairs

@patch('os.path.exists', side_effect=lambda path: "001.json" in path or "002.json" in path)
def test_pop909_dataset_len(mock_exists, mock_dataset_config):
    dataset = Pop909Dataset(mock_dataset_config)
    assert len(dataset) == 2

@patch('os.path.exists', side_effect=lambda path: "001.json" in path)
@patch('json.load', side_effect=lambda f: {"tokens": [1, 2, 3]})
def test_pop909_dataset_getitem(mock_exists, mock_file_open, mock_json_load, mock_dataset_config):
    dataset = Pop909Dataset(mock_dataset_config)
    mel_data, acc_data = dataset[0]
    assert mel_data == {"tokens": [1, 2, 3]}
    assert acc_data == {"tokens": [1, 2, 3]}
    mock_file_open.assert_any_call("mock_mel_root/001.json", 'r')
    mock_file_open.assert_any_call("mock_acc_root/001.json", 'r')
    assert mock_json_load.call_count == 2

@patch('os.path.exists', side_effect=lambda path: False) # No files exist
def test_pop909_dataset_empty(mock_exists, mock_dataset_config):
    dataset = Pop909Dataset(mock_dataset_config)
    assert len(dataset) == 0
    assert dataset.midi_file_pairs == []

@patch('os.path.exists', side_effect=lambda path: "001.json" in path)
@patch('json.load', side_effect=Exception("Corrupted JSON"))
def test_pop909_dataset_getitem_error_handling(mock_exists, mock_file_open, mock_json_load, mock_dataset_config, capsys):
    dataset = Pop909Dataset(mock_dataset_config)
    mel_data, acc_data = dataset[0]
    assert mel_data is None
    assert acc_data is None
    captured = capsys.readouterr()
    assert "Error loading JSON files" in captured.out
    assert "Corrupted JSON" in captured.out

# Test Pop909DataModule
def test_pop909_datamodule_init(mock_datamodule_config):
    datamodule = Pop909DataModule(mock_datamodule_config)
    assert datamodule.batch_size == 2
    assert datamodule.num_workers == 0
    assert isinstance(datamodule.train_dataset_config, MockPop909DatasetSchema)

@patch('datamodule.pop909_dataset.Pop909Dataset', autospec=True)
def test_pop909_datamodule_setup_fit(mock_pop909_dataset, mock_datamodule_config):
    datamodule = Pop909DataModule(mock_datamodule_config)
    datamodule.setup(stage="fit")
    mock_pop909_dataset.assert_any_call(mock_datamodule_config.train_dataset_config)
    mock_pop909_dataset.assert_any_call(mock_datamodule_config.val_dataset_config)
    assert hasattr(datamodule, 'train_dataset')
    assert hasattr(datamodule, 'val_dataset')

@patch('datamodule.pop909_dataset.Pop909Dataset', autospec=True)
def test_pop909_datamodule_setup_test(mock_pop909_dataset, mock_datamodule_config):
    datamodule = Pop909DataModule(mock_datamodule_config)
    datamodule.setup(stage="test")
    mock_pop909_dataset.assert_called_once_with(mock_datamodule_config.test_dataset_config)
    assert hasattr(datamodule, 'test_dataset')

@patch('datamodule.pop909_dataset.Pop909Dataset', autospec=True)
def test_pop909_datamodule_setup_predict(mock_pop909_dataset, mock_datamodule_config):
    datamodule = Pop909DataModule(mock_datamodule_config)
    datamodule.setup(stage="predict")
    mock_pop909_dataset.assert_called_once_with(mock_datamodule_config.predict_dataset_config)
    assert hasattr(datamodule, 'predict_dataset')

@patch('datamodule.pop909_dataset.Pop909Dataset', autospec=True)
def test_pop909_datamodule_dataloaders(mock_pop909_dataset, mock_datamodule_config):
    datamodule = Pop909DataModule(mock_datamodule_config)
    
    # Mock the datasets to be available for dataloader creation
    datamodule.train_dataset = MagicMock()
    datamodule.val_dataset = MagicMock()
    datamodule.test_dataset = MagicMock()
    datamodule.predict_dataset = MagicMock()

    train_loader = datamodule.train_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 2
    assert train_loader.dataset == datamodule.train_dataset

    val_loader = datamodule.val_dataloader()
    assert isinstance(val_loader, DataLoader)
    assert val_loader.batch_size == 2
    assert val_loader.dataset == datamodule.val_dataset

    test_loader = datamodule.test_dataloader()
    assert isinstance(test_loader, DataLoader)
    assert test_loader.batch_size == 2
    assert test_loader.dataset == datamodule.test_dataset

    predict_loader = datamodule.predict_dataloader()
    assert isinstance(predict_loader, DataLoader)
    assert predict_loader.batch_size == 2
    assert predict_loader.dataset == datamodule.test_dataset # predict_dataloader uses test_dataset in the original code
