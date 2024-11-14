import tent
import torch
from models.resnet_reconstruction import ResNet18Reconstruct
from utils.data import create_cifar10c_dataloaders_single
from tqdm import tqdm
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

all_corruptions = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

def setup_directories():
    """Create directories for saving results if they don't exist."""
    os.makedirs('convergence_plots', exist_ok=True)
    os.makedirs('convergence_data', exist_ok=True)

def print_trainable_layers(model):
    """Print which layers of the model have trainable parameters."""
    print("\nVerifying trainable layers:")
    print("-" * 50)
    trainable_params = 0
    total_params = 0
    
    max_name_length = max(len(name) for name, _ in model.named_modules())
    
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            params_in_layer = sum(p.numel() for p in module.parameters() if p.requires_grad)
            trainable_params += params_in_layer
            print(f"✓ {name:<{max_name_length}} | Trainable parameters: {params_in_layer:,}")
        total_params += sum(p.numel() for p in module.parameters())
    
    print("-" * 50)
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print("-" * 50)

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on the entire test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100 * correct / total

def plot_convergence(convergence_data, corruption, save_path=None):
    """Plot convergence curves for different adaptation modes."""
    plt.figure(figsize=(10, 6))
    
    styles = {
        'projection': ('blue', '-'),
        'affine': ('red', '--'),
        'both': ('green', ':')
    }
    
    for mode, data in convergence_data.items():
        color, linestyle = styles[mode]
        steps = np.arange(len(data))
        plt.plot(steps, data, label=f'{mode} adaptation', 
                color=color, linestyle=linestyle, linewidth=2)
    
    plt.title(f'Adaptation Convergence on {corruption} Corruption')
    plt.xlabel('Steps (batches)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TENT adaptation with different parameter configurations')
    parser.add_argument('--modes', nargs='+', choices=['projection', 'affine', 'both'], 
                       default=['projection', 'affine'], help='Which parameters to adapt')
    args = parser.parse_args()
    
    setup_directories()
    
    mode_map = {
        'projection': tent.AdaptMode.PROJECTION_ONLY,
        'affine': tent.AdaptMode.AFFINE_ONLY,
        'both': tent.AdaptMode.BOTH
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_convergence_data = defaultdict(lambda: defaultdict(list))
    
    # Load the state dict once
    state_dict = torch.load('runs/resnet18_cifar10_base_4/final_model/mp_rank_00_model_states.pt')["module"]
    
    for corruption in all_corruptions:
        print(f"\n{'='*50}\nAnalyzing convergence on corruption: {corruption}\n{'='*50}")
        
        # Load data once for this corruption
        _, _, _, test_loader, _ = create_cifar10c_dataloaders_single(
            "./datasets/CIFAR-10-C",
            corruption_type=corruption,
            severity=5,
            preprocess=True,
            batch_size=32
        )
        
        # Create a separate loader for adaptation
        adapt_loader, _, _, eval_loader, _ = create_cifar10c_dataloaders_single(
            "./datasets/CIFAR-10-C",
            corruption_type=corruption,
            severity=5,
            preprocess=True,
            batch_size=32
        )
        
        # Initialize fresh base model for this corruption
        base_model = ResNet18Reconstruct().to(device)
        base_model.load_state_dict(state_dict, strict=False)
        base_accuracy = evaluate_model(base_model, eval_loader, device)
        print(f"Base model accuracy on {corruption}: {base_accuracy:.2f}%")
        
        convergence_data = {mode: [] for mode in args.modes}
        
        for mode in args.modes:
            print(f"\nTesting {mode} adaptation")
            
            # Initialize fresh model for each mode
            model = ResNet18Reconstruct().to(device)
            model.load_state_dict(state_dict, strict=False)
            
            # Configure TENT
            model = tent.configure_model_for_adaptation(model, mode_map[mode])
            print_trainable_layers(model)
            params, _ = tent.collect_params(model, mode_map[mode])
            optimizer = torch.optim.Adam(params, lr=0.00025, weight_decay=0.0001)
            tented_model = tent.Tent(model, optimizer)
            
            # Initial evaluation
            initial_accuracy = evaluate_model(tented_model.model, eval_loader, device)
            convergence_data[mode].append(initial_accuracy)
            
            # Adaptation loop with tracking
            for batch_idx, batch in enumerate(tqdm(adapt_loader, desc=f"Tracking {mode} convergence")):
                # Adapt on batch
                inputs, _ = batch
                inputs = inputs.to(device)
                _ = tented_model(inputs)
                
                # Evaluate on full test set
                accuracy = evaluate_model(tented_model.model, eval_loader, device)
                convergence_data[mode].append(accuracy)
                all_convergence_data[corruption][mode] = convergence_data[mode]
        
        # Plot and save convergence curves
        plot_convergence(
            convergence_data, 
            corruption,
            save_path=f'convergence_plots/{corruption}_convergence.png'
        )
        
        # Print convergence statistics
        print("\nConvergence Analysis:")
        print(f"Base accuracy:    {base_accuracy:.2f}%")
        for mode in args.modes:
            data = convergence_data[mode]
            print(f"\n{mode.capitalize()} adaptation:")
            print(f"Initial accuracy:  {data[0]:.2f}%")
            print(f"Final accuracy:    {data[-1]:.2f}%")
            print(f"Max accuracy:      {max(data):.2f}%")
            print(f"Min accuracy:      {min(data):.2f}%")
            print(f"Improvement:       {data[-1] - data[0]:+.2f}%")
            
            # Calculate steps to 90% of max improvement
            max_improvement = max(data) - data[0]
            if max_improvement > 0:
                threshold = data[0] + 0.9 * max_improvement
                steps_to_90 = next((i for i, acc in enumerate(data) if acc >= threshold), len(data))
                print(f"Steps to 90% of max improvement: {steps_to_90}")
    
    # Save the raw convergence data
    np.save('convergence_data/all_convergence_data.npy', dict(all_convergence_data))
    
    # Create average convergence plot
    plt.figure(figsize=(10, 6))
    for mode in args.modes:
        all_curves = [data[mode] for data in all_convergence_data.values()]
        min_length = min(len(curve) for curve in all_curves)
        truncated_curves = [curve[:min_length] for curve in all_curves]
        avg_curve = np.mean(truncated_curves, axis=0)
        std_curve = np.std(truncated_curves, axis=0)
        
        steps = np.arange(len(avg_curve))
        plt.plot(steps, avg_curve, label=f'{mode} adaptation', **styles[mode])
        plt.fill_between(steps, 
                        avg_curve - std_curve, 
                        avg_curve + std_curve, 
                        alpha=0.2, 
                        color=styles[mode][0])
    
    plt.title('Average Adaptation Convergence Across All Corruptions')
    plt.xlabel('Steps (batches)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('convergence_plots/average_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()