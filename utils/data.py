import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Helper function to unpickle a file
def unpickle(file, logger=None, distributed=False):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    if logger:
        if not distributed or dist.get_rank() == 0:
            logger.info(f"Unpickled file: {file}")
    return data_dict

def is_channel_last(tensor):
    if len(tensor.shape) != 4:
        raise ValueError("Expected a 4D tensor (batch, height, width, channels) or (batch, channels, height, width)")
    
    # Assume that the number of channels is always smaller than height and width
    if tensor.shape[-1] < tensor.shape[1] and tensor.shape[-1] < tensor.shape[2]:
        return True
    elif tensor.shape[1] < tensor.shape[2] and tensor.shape[1] < tensor.shape[3]:
        return False
    else:
        raise ValueError("Unable to determine channel order definitively")
  
def preprocess_data(data):
    data = data.reshape(-1, 3, 32, 32)
    if not is_channel_last(data):
        return data.transpose(0, 2, 3, 1)  # Change to (N, 32, 32, 3) for easier processing
    return data

def get_training_augmentation_pipeline():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),     
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

def normalize_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = transforms.ToTensor()(tensor)
    
    return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])(tensor)

def generate_augmentation_plot(train_data, transform, save_dir, name="augmentation_examples"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Original vs Augmented Images", fontsize=16)

    for i in range(3):
        # Randomly select an image
        idx = np.random.randint(0, len(train_data))
        img = train_data[idx]
        
        # Original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        # Augmented image
        aug_img = transform(img.copy())
        aug_img_np = np.array(aug_img).transpose(1, 2, 0)
        axes[1, i].imshow(aug_img_np)
        axes[1, i].set_title(f"Augmented {i+1}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, normalize=None, transforms=None, augmentation_ratio=0):
        self.data = data
        self.labels = labels
        self.normalize = normalize
        self.transforms = transforms
        self.augmentation_ratio = augmentation_ratio

    def __len__(self):
        return len(self.data) * (1 + self.augmentation_ratio)

    def __getitem__(self, index):
        original_length = len(self.data)

        # If index belongs to original dataset, return the original image
        if index < original_length:
            image = self.data[index]
            label = self.labels[index]
            if self.normalize:
                image = self.normalize(image)
            return image, label

        # For augmented data, apply augmentations
        aug_index = index % original_length
        image = self.data[aug_index]
        label = self.labels[aug_index]
        aug_img = self.transforms(image) if self.transforms else image
        if self.normalize:
            aug_img = self.normalize(aug_img)
        
        return aug_img, label
    
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
# def get_contrastive_transforms():
#     simclr_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),     # TODO: modify the hard-coded size
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#     ])

#     return simclr_transforms

class CIFAR10ContrastiveDataset(Dataset):
    def __init__(self, data, labels, normalize=None, transforms=None, augmentation_ratio=0):
        self.data = data
        self.labels = labels
        self.normalize = normalize
        self.transforms = transforms
        self.augmentation_ratio = augmentation_ratio

    def __len__(self):
        return len(self.data) * (1 + self.augmentation_ratio)

    def __getitem__(self, index):
        original_length = len(self.data)

        # If index belongs to original dataset, return the original image
        if index < original_length:
            image = self.data[index]
            label = self.labels[index]
            if self.normalize:
                image = self.normalize(image)
            images = [self.transforms(image), self.transforms(image)]
            return images, label

        # For augmented data, apply augmentations
        aug_index = index % original_length
        image = self.data[aug_index]
        label = self.labels[aug_index]
        if self.normalize:
            image = self.normalize(image)
        images = [self.transforms(image), self.transforms(image)]
        
        return images, label
    

# ~~~~~~~~~~ CIFAR-10-C HELPERS ~~~~~~~~~~

def load_cifar10c_data_single(folder, corruption_type=None, preprocess=True, severity=1, logger=None):
    """
    Load CIFAR-10-C data for either a specific corruption or all corruptions of a given severity
    
    Args:
        folder (str): Path to CIFAR-10-C data
        corruption_type (str, optional): Specific corruption type. If None, loads all corruptions
        preprocess (bool): Whether to preprocess the data
        severity (int): Severity level (1-5)
        logger: Logger object for logging information
    
    Returns:
        tuple: (data, labels)
    """
    if corruption_type is not None:
        # Single corruption case - original behavior
        data = np.load(f"{folder}/{corruption_type}.npy")
        data = data[(severity-1)*10000:severity*10000]
        
        labels = np.load(f"{folder}/labels.npy")
        labels = labels[(severity-1)*10000:severity*10000]
        
        if logger and dist.get_rank() == 0:
            logger.info(f"Loaded CIFAR-10-C data for corruption: {corruption_type}, severity: {severity}")
    else:
        # All corruptions for given severity case
        corruptions = [
            'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
            'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
            'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
            'snow', 'spatter', 'speckle_noise', 'zoom_blur'
        ]
        
        all_data = []
        all_labels = []
        base_labels = np.load(f"{folder}/labels.npy")
        base_labels = base_labels[(severity-1)*10000:severity*10000]
        
        for corruption in corruptions:
            data = np.load(f"{folder}/{corruption}.npy")
            data = data[(severity-1)*10000:severity*10000]
            all_data.append(data)
            all_labels.append(base_labels)
            
            if logger and dist.get_rank() == 0:
                logger.info(f"Loaded corruption: {corruption} with severity {severity}")
        
        data = np.concatenate(all_data)
        labels = np.concatenate(all_labels)
        
    data, labels = shuffle(data, labels, random_state=0)
    
    return data, labels

def load_cifar10c_data_full(folder, preprocess=True, logger=None, distributed=False):
    corruptions = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
        'jpeg_compression', 'labels', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
        'snow', 'spatter', 'speckle_noise', 'zoom_blur'
    ]
    
    all_data = []
    all_labels = []
    all_corruption_types = []
    all_severities = []
    
    labels = np.load(f"{folder}/labels.npy")
    
    for corruption in corruptions:
        if corruption != 'labels':  # Skip the labels file since we already loaded it
            data = np.load(f"{folder}/{corruption}.npy")
            
            for severity in range(1, 6):
                start_idx = (severity - 1) * 10000
                end_idx = severity * 10000
                severity_data = data[start_idx:end_idx]
                severity_labels = labels[start_idx:end_idx]
                
                all_data.append(severity_data)
                all_labels.append(severity_labels)
                all_corruption_types.extend([corruption] * 10000)
                all_severities.extend([severity] * 10000)
    
    # Concatenate all data along the first dimension
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle all arrays together
    indices = np.random.RandomState(0).permutation(len(all_data))
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    all_corruption_types = np.array(all_corruption_types)[indices]
    all_severities = np.array(all_severities)[indices]
    
    if logger:
        if not distributed or dist.get_rank() == 0:
            final_label_order = sorted(set(all_labels))
            logger.info(f"CIFAR-10-C label order: {final_label_order}")
            logger.info(f"Loaded full CIFAR-10-C dataset from folder: {folder}")
            logger.info(f"Total number of samples: {len(all_data)}")
    
    return all_data, all_labels, all_corruption_types, all_severities

def create_cifar10c_dataloaders_single(folder, corruption_type=None, severity=1, preprocess=True, batch_size=64, test_split=0.1, logger=None, contrastive=False):
    """
    Create CIFAR-10-C dataloaders for either a specific corruption or all corruptions of a given severity
    
    Args:
        folder (str): Path to CIFAR-10-C data
        corruption_type (str, optional): Specific corruption type. If None, loads all corruptions
        severity (int): Severity level (1-5)
        preprocess (bool): Whether to preprocess the data
        batch_size (int): Batch size for dataloaders
        test_split (float): Proportion of data to use for testing
        logger: Logger object for logging information
        contrastive (bool): Whether to use contrastive learning datasets
    
    Returns:
        tuple: (train_loader, train_sampler, train_dataset, test_loader, test_dataset)
    """
    data, labels = load_cifar10c_data_single(folder, corruption_type, preprocess, severity, logger)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_split, random_state=42)

    if contrastive:
        train_dataset = CIFAR10ContrastiveDataset(train_data, train_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
        test_dataset = CIFAR10ContrastiveDataset(test_data, test_data, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
    else:
        train_dataset = CIFAR10Dataset(train_data, train_labels, normalize=normalize_tensor)
        test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)
    
    # Initialize distributed environment
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders with distributed samplers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    if logger and rank == 0:
        if corruption_type:
            logger.info(f"Created CIFAR-10-C dataloaders for corruption: {corruption_type}, severity: {severity}")
        else:
            logger.info(f"Created CIFAR-10-C dataloaders for all corruptions with severity: {severity}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, train_sampler, train_dataset, test_loader, test_dataset

def create_cifar10c_dataloaders_full(folder, preprocess=True, batch_size=64, test_split=0.1, logger=None, contrastive=False):
    distributed = torch.distributed.is_initialized()
    data, labels, corruption_types, severities = load_cifar10c_data_full(folder, preprocess, logger, distributed)
    
    # Split the data
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_split, random_state=42
    )
    
    if contrastive:
        train_dataset = CIFAR10ContrastiveDataset(
            train_data, train_labels, 
            normalize=normalize_tensor, 
            transforms=get_training_augmentation_pipeline()
        )
        test_dataset = CIFAR10ContrastiveDataset(
            test_data, test_labels, 
            normalize=normalize_tensor, 
            transforms=get_training_augmentation_pipeline()
        )
    else:
        train_dataset = CIFAR10Dataset(train_data, train_labels, normalize=normalize_tensor)
        test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)
    
    # Initialize distributed environment
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders with distributed samplers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    if logger and rank == 0:
        logger.info(f"Created full CIFAR-10-C dataloaders from folder: {folder}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, train_sampler, train_dataset, test_loader, test_dataset

# ~~~~~~~~~~ CIFAR-10 HELPERS ~~~~~~~~~

def load_cifar10_data(folder, preprocess=True, to_float=False, logger=None, distributed=False):
    data = []
    labels = []
    
    # Load training data
    for i in range(1, 6):
        batch = unpickle(f"{folder}/data_batch_{i}", logger, distributed=distributed)
        data.append(batch[b'data'])
        labels.append(batch[b'labels'])

    # Load test data
    test_batch = unpickle(f"{folder}/test_batch", logger, distributed=distributed)
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Concatenate training data
    train_data = np.concatenate(data)
    train_labels = np.concatenate(labels)
            
    train_data, train_labels = shuffle(train_data, train_labels, random_state=0)

    # Convert to float if required
    if to_float:
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)

    # Preprocess if required
    if preprocess:
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

    # Log overall label order
    if logger:
        if not distributed or dist.get_rank() == 0:
            all_labels = np.concatenate([train_labels, test_labels])
            final_label_order = sorted(set(map(int, all_labels)))  # Convert to regular Python ints
            logger.info(f"CIFAR-10 label order: {final_label_order}")
            logger.info(f"Loaded CIFAR-10 data from folder: {folder}")
            
    return train_data, train_labels, test_data, test_labels

def create_cifar10_dataloaders(folder, batch_size=64, preprocess=True, val_split=0.1, augmentation_ratio=0, logger=None, save_dir=None, contrastive=False):
    distributed = torch.distributed.is_initialized()
    
    train_data, train_labels, test_data, test_labels = load_cifar10_data(folder, preprocess, logger=logger, distributed=distributed)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_split, random_state=42)
        
    if save_dir:
        if not distributed or torch.distributed.get_rank() == 0: 
            generate_augmentation_plot(train_data, get_training_augmentation_pipeline(), save_dir)

    if contrastive:
        train_dataset = CIFAR10ContrastiveDataset(train_data, train_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline(), augmentation_ratio=augmentation_ratio)
        val_dataset = CIFAR10ContrastiveDataset(val_data, val_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
        test_dataset = CIFAR10ContrastiveDataset(test_data, test_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
    else:     
        train_dataset = CIFAR10Dataset(train_data, train_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline(), augmentation_ratio=augmentation_ratio)
        val_dataset = CIFAR10Dataset(val_data, val_labels, normalize=normalize_tensor)
        test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)

    # Initialize distributed environment
    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders with distributed samplers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    if logger:
        if not distributed or torch.distributed.get_rank() == 0:
            logger.info(f"Created CIFAR-10 dataloaders from folder: {folder}")
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Validation dataset size: {len(val_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # return train_loader, train_sampler, train_dataset, val_loader, val_sampler, val_dataset, test_loader, test_sampler, test_dataset
    return train_loader, train_sampler, train_dataset, val_loader, val_dataset, test_loader, test_dataset

# ~~~~~~~~~~ CIFAR-10.1 HELPERS ~~~~~~~~~

def load_cifar101_data(folder, preprocess=True, to_float=False, logger=None, distributed=False):
    # Load CIFAR-10.1 test data and labels from .npy files
    test_data = np.load(f"{folder}/cifar10.1_data.npy")
    test_labels = np.load(f"{folder}/cifar10.1_labels.npy")

    # Convert to float if required
    if to_float:
        test_data = test_data.astype(np.float32)

    # Preprocess if required
    if preprocess:
        test_data = preprocess_data(test_data)

    # Log overall label order for CIFAR-10.1
    if logger:
        if not distributed or dist.get_rank() == 0:
            final_label_order = sorted(set(map(int, test_labels)))  # Convert to regular Python ints
            logger.info(f"CIFAR-10.1 label order: {final_label_order}")
            logger.info(f"Loaded CIFAR-10.1 data from folder: {folder}")

    return test_data, test_labels

def create_cifar101_dataloaders(folder, batch_size=64, preprocess=True, logger=None):
    distributed = torch.distributed.is_initialized()
    
    # Load CIFAR-10.1 data if requested
    test_data, test_labels = load_cifar101_data(folder, preprocess, logger=logger, distributed=distributed)
    test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)
    contrastive_dataset = CIFAR10ContrastiveDataset(test_data, test_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())

    # Initialize distributed environment
    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    contrastive_sampler = DistributedSampler(contrastive_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    contrastive_loader = DataLoader(contrastive_dataset, batch_size=batch_size, sampler=contrastive_sampler, num_workers=4, pin_memory=True)

    if logger:
        if not distributed or torch.distributed.get_rank() == 0:
            logger.info(f"Created CIFAR-101 dataloaders from folder: {folder}")
            logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return test_loader, test_sampler, test_dataset, contrastive_loader, contrastive_sampler, contrastive_dataset

# ~~~~~~~~~~ CIFAR-100 HELPERS ~~~~~~~~~~

def load_cifar100_data(folder, preprocess=True, to_float=False, logger=None, distributed=False):
    """
    Load CIFAR-100 data from the specified folder.
    """
    # Load training data
    train_batch = unpickle(f"{folder}/train", logger, distributed=distributed)
    train_data = train_batch[b'data']
    train_labels = train_batch[b'fine_labels']  # Use fine-grained labels

    # Load test data
    test_batch = unpickle(f"{folder}/test", logger, distributed=distributed)
    test_data = test_batch[b'data']
    test_labels = test_batch[b'fine_labels']  # Use fine-grained labels

    train_data, train_labels = shuffle(train_data, train_labels, random_state=0)

    # Convert to float if required
    if to_float:
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)

    # Preprocess if required
    if preprocess:
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

    # Log overall label order
    if logger:
        if not distributed or dist.get_rank() == 0:
            all_labels = np.concatenate([train_labels, test_labels])
            final_label_order = sorted(set(map(int, all_labels)))
            logger.info(f"CIFAR-100 label order: {final_label_order}")
            logger.info(f"Loaded CIFAR-100 data from folder: {folder}")
            
    return train_data, train_labels, test_data, test_labels

def create_cifar100_dataloaders(folder, batch_size=64, preprocess=True, val_split=0.1, augmentation_ratio=0, logger=None, save_dir=None, contrastive=False):
    """
    Create CIFAR-100 dataloaders with optional validation split and data augmentation.
    """
    distributed = torch.distributed.is_initialized()
    
    train_data, train_labels, test_data, test_labels = load_cifar100_data(folder, preprocess, logger=logger, distributed=distributed)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_split, random_state=42)
        
    if save_dir:
        if not distributed or torch.distributed.get_rank() == 0: 
            generate_augmentation_plot(train_data, get_training_augmentation_pipeline(), save_dir, "cifar100_augmentation_examples")

    if contrastive:
        train_dataset = CIFAR10ContrastiveDataset(train_data, train_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline(), augmentation_ratio=augmentation_ratio)
        val_dataset = CIFAR10ContrastiveDataset(val_data, val_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
        test_dataset = CIFAR10ContrastiveDataset(test_data, test_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
    else:     
        train_dataset = CIFAR10Dataset(train_data, train_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline(), augmentation_ratio=augmentation_ratio)
        val_dataset = CIFAR10Dataset(val_data, val_labels, normalize=normalize_tensor)
        test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)

    # Initialize distributed environment
    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders with distributed samplers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    if logger:
        if not distributed or torch.distributed.get_rank() == 0:
            logger.info(f"Created CIFAR-100 dataloaders from folder: {folder}")
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Validation dataset size: {len(val_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, train_sampler, train_dataset, val_loader, val_dataset, test_loader, test_dataset

# ~~~~~~~~~~ CIFAR-100-C HELPERS ~~~~~~~~~~

def load_cifar100c_data_single(folder, corruption_type=None, preprocess=True, severity=1, logger=None):
    """
    Load CIFAR-100-C data for either a specific corruption or all corruptions of a given severity.
    Similar to CIFAR-10-C but with 100 classes.
    """
    if corruption_type is not None:
        # Single corruption case
        data = np.load(f"{folder}/{corruption_type}.npy")
        data = data[(severity-1)*10000:severity*10000]
        
        labels = np.load(f"{folder}/labels.npy")
        labels = labels[(severity-1)*10000:severity*10000]
        
        if logger and dist.get_rank() == 0:
            logger.info(f"Loaded CIFAR-100-C data for corruption: {corruption_type}, severity: {severity}")
    else:
        # All corruptions for given severity case
        corruptions = [
            'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
            'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
            'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
            'snow', 'spatter', 'speckle_noise', 'zoom_blur'
        ]
        
        all_data = []
        all_labels = []
        base_labels = np.load(f"{folder}/labels.npy")
        base_labels = base_labels[(severity-1)*10000:severity*10000]
        
        for corruption in corruptions:
            data = np.load(f"{folder}/{corruption}.npy")
            data = data[(severity-1)*10000:severity*10000]
            all_data.append(data)
            all_labels.append(base_labels)
            
            if logger and dist.get_rank() == 0:
                logger.info(f"Loaded corruption: {corruption} with severity {severity}")
        
        data = np.concatenate(all_data)
        labels = np.concatenate(all_labels)
        
    data, labels = shuffle(data, labels, random_state=0)
    
    return data, labels

def load_cifar100c_data_full(folder, preprocess=True, logger=None, distributed=False):
    """
    Load the complete CIFAR-100-C dataset including all corruptions and severities.
    """
    corruptions = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
        'jpeg_compression', 'labels', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
        'snow', 'spatter', 'speckle_noise', 'zoom_blur'
    ]
    
    all_data = []
    all_labels = []
    all_corruption_types = []
    all_severities = []
    
    labels = np.load(f"{folder}/labels.npy")
    
    for corruption in corruptions:
        if corruption != 'labels':
            data = np.load(f"{folder}/{corruption}.npy")
            
            for severity in range(1, 6):
                start_idx = (severity - 1) * 10000
                end_idx = severity * 10000
                severity_data = data[start_idx:end_idx]
                severity_labels = labels[start_idx:end_idx]
                
                all_data.append(severity_data)
                all_labels.append(severity_labels)
                all_corruption_types.extend([corruption] * 10000)
                all_severities.extend([severity] * 10000)
    
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    indices = np.random.RandomState(0).permutation(len(all_data))
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    all_corruption_types = np.array(all_corruption_types)[indices]
    all_severities = np.array(all_severities)[indices]
    
    if logger:
        if not distributed or dist.get_rank() == 0:
            final_label_order = sorted(set(all_labels))
            logger.info(f"CIFAR-100-C label order: {final_label_order}")
            logger.info(f"Loaded full CIFAR-100-C dataset from folder: {folder}")
            logger.info(f"Total number of samples: {len(all_data)}")
    
    return all_data, all_labels, all_corruption_types, all_severities

def create_cifar100c_dataloaders_single(folder, corruption_type=None, severity=1, preprocess=True, batch_size=64, test_split=0.1, logger=None, contrastive=False):
    """
    Create CIFAR-100-C dataloaders for either a specific corruption or all corruptions of a given severity.
    """
    data, labels = load_cifar100c_data_single(folder, corruption_type, preprocess, severity, logger)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_split, random_state=42)

    if contrastive:
        train_dataset = CIFAR10ContrastiveDataset(train_data, train_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
        test_dataset = CIFAR10ContrastiveDataset(test_data, test_labels, normalize=normalize_tensor, transforms=get_training_augmentation_pipeline())
    else:
        train_dataset = CIFAR10Dataset(train_data, train_labels, normalize=normalize_tensor)
        test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)
    
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    if logger and rank == 0:
        if corruption_type:
            logger.info(f"Created CIFAR-100-C dataloaders for corruption: {corruption_type}, severity: {severity}")
        else:
            logger.info(f"Created CIFAR-100-C dataloaders for all corruptions with severity: {severity}")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, train_sampler, train_dataset, test_loader, test_dataset

def create_cifar100c_dataloaders_full(folder, preprocess=True, batch_size=64, test_split=0.1, logger=None, contrastive=False):
    """
    Create dataloaders for the complete CIFAR-100-C dataset.
    """
    distributed = torch.distributed.is_initialized()
    data, labels, corruption_types, severities = load_cifar100c_data_full(folder, preprocess, logger, distributed)
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_split, random_state=42
    )
    
    if contrastive:
        train_dataset = CIFAR10ContrastiveDataset(
            train_data, train_labels, 
            normalize=normalize_tensor, 
            transforms=get_training_augmentation_pipeline()
        )
        test_dataset = CIFAR10ContrastiveDataset(
            test_data, test_labels, 
            normalize=normalize_tensor, 
            transforms=get_training_augmentation_pipeline()
        )
    else:
        train_dataset = CIFAR10Dataset(train_data, train_labels, normalize=normalize_tensor)
        test_dataset = CIFAR10Dataset(test_data, test_labels, normalize=normalize_tensor)
    
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    if logger and rank == 0:
        logger.info(f"Created full CIFAR-100-C dataloaders")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, train_sampler, train_dataset, test_loader, test_dataset
