import tent
import torch
from models.resnet_reconstruction import ResNet18Reconstruct
from utils.data import create_cifar10c_dataloaders_single
from tqdm import tqdm
from collections import defaultdict
import argparse

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

def adapt_and_evaluate(model, test_loader, mode, device='cuda'):
    """Run adaptation and return accuracy."""
    total = 0
    correct = 0
    for inputs, labels in tqdm(test_loader, desc=f"Adapting ({mode})"):
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100 * correct / total

all_corruptions = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TENT adaptation with different parameter configurations')
    parser.add_argument('--mode', type=str, choices=['projection', 'affine', 'both', 'sequential'], 
                       default='both', help='Which parameters to adapt')
    args = parser.parse_args()
    
    mode_map = {
        'projection': tent.AdaptMode.PROJECTION_ONLY,
        'affine': tent.AdaptMode.AFFINE_ONLY,
        'both': tent.AdaptMode.BOTH
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {'per_corruption': defaultdict(lambda: {'before': 0.0, 'affine': 0.0, 'projection': 0.0, 'after': 0.0}),
              'global': {'before': [], 'affine': [], 'projection': [], 'after': []}}
    
    # Initialize models
    print(f"\nInitializing models with adaptation mode: {args.mode}")
    base_model = ResNet18Reconstruct().to(device)
    model = ResNet18Reconstruct().to(device)
    state_dict = torch.load('runs/resnet18_cifar10_base_4_dal/latest_model/mp_rank_00_model_states.pt')["module"]
    base_model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict, strict=False)
    
    # Evaluation loop
    for corruption in all_corruptions:
        print(f"\n{'='*50}\nTesting corruption: {corruption}\n{'='*50}")
        
        # Load data
        _, _, _, test_loader, _ = create_cifar10c_dataloaders_single(
            "./datasets/CIFAR-10-C",
            corruption_type=corruption,
            severity=5,
            preprocess=True,
            batch_size=32
        )
        
        # Get base accuracy
        base_accuracy = evaluate_model(base_model, test_loader, device)
        results['per_corruption'][corruption]['before'] = base_accuracy
        results['global']['before'].append(base_accuracy)
        print(f"Base model accuracy: {base_accuracy:.2f}%")
        
        if args.mode == 'sequential':
            # First phase: Affine adaptation
            print("\nPhase 1: Affine adaptation")
            model_affine = tent.configure_model_for_adaptation(model, tent.AdaptMode.AFFINE_ONLY)
            print_trainable_layers(model_affine)
            params_affine, _ = tent.collect_params(model_affine, tent.AdaptMode.AFFINE_ONLY)
            optimizer_affine = torch.optim.Adam(params_affine, lr=0.00025, weight_decay=0.0001)
            tented_model_affine = tent.Tent(model_affine, optimizer_affine)
            
            affine_accuracy = adapt_and_evaluate(tented_model_affine, test_loader, "affine", device)
            results['per_corruption'][corruption]['affine'] = affine_accuracy
            results['global']['affine'].append(affine_accuracy)
            print(f"After affine adaptation: {affine_accuracy:.2f}%")
            
            # Second phase: Projection adaptation (starting from affine-adapted model)
            print("\nPhase 2: Projection adaptation")
            model_proj = tent.configure_model_for_adaptation(model_affine, tent.AdaptMode.PROJECTION_ONLY)
            print_trainable_layers(model_proj)
            params_proj, _ = tent.collect_params(model_proj, tent.AdaptMode.PROJECTION_ONLY)
            optimizer_proj = torch.optim.Adam(params_proj, lr=0.00025, weight_decay=0.0001)
            tented_model_proj = tent.Tent(model_proj, optimizer_proj)
            
            final_accuracy = adapt_and_evaluate(tented_model_proj, test_loader, "projection", device)
            results['per_corruption'][corruption]['after'] = final_accuracy
            results['global']['after'].append(final_accuracy)
            
        else:
            # Regular (non-sequential) adaptation
            model = tent.configure_model_for_adaptation(model, mode_map[args.mode])
            print_trainable_layers(model)
            params, _ = tent.collect_params(model, mode_map[args.mode])
            optimizer = torch.optim.Adam(params, lr=0.00025, weight_decay=0.0001)
            tented_model = tent.Tent(model, optimizer)
            
            final_accuracy = adapt_and_evaluate(tented_model, test_loader, args.mode, device)
            results['per_corruption'][corruption]['after'] = final_accuracy
            results['global']['after'].append(final_accuracy)
        
        # Print results for this corruption
        print(f"\nResults for {corruption}:")
        print(f"Before adaptation: {base_accuracy:.2f}%")
        if args.mode == 'sequential':
            print(f"After affine:      {affine_accuracy:.2f}% (Δ: {affine_accuracy - base_accuracy:+.2f}%)")
            print(f"After projection:  {final_accuracy:.2f}% (Δ: {final_accuracy - affine_accuracy:+.2f}%)")
            print(f"Total improvement: {final_accuracy - base_accuracy:+.2f}%")
        else:
            print(f"After adaptation:  {final_accuracy:.2f}%")
            print(f"Improvement:       {final_accuracy - base_accuracy:+.2f}%")
    
    # Print final statistics
    print("\n" + "="*50 + "\nGLOBAL STATISTICS\n" + "="*50)
    global_before_avg = sum(results['global']['before']) / len(results['global']['before'])
    global_after_avg = sum(results['global']['after']) / len(results['global']['after'])
    
    print("\nPer-corruption results:")
    if args.mode == 'sequential':
        global_affine_avg = sum(results['global']['affine']) / len(results['global']['affine'])
        print(f"{'Corruption':<20} {'Before':>10} {'Affine':>10} {'Final':>10} {'Δ Total':>10}")
        print("-" * 60)
        for corruption in all_corruptions:
            before = results['per_corruption'][corruption]['before']
            affine = results['per_corruption'][corruption]['affine']
            after = results['per_corruption'][corruption]['after']
            print(f"{corruption:<20} {before:>10.2f}% {affine:>10.2f}% {after:>10.2f}% {after-before:>+10.2f}%")
        print(f"\nOverall accuracy:")
        print(f"Before adaptation:      {global_before_avg:.2f}%")
        print(f"After affine:          {global_affine_avg:.2f}% (Δ: {global_affine_avg - global_before_avg:+.2f}%)")
        print(f"After projection:      {global_after_avg:.2f}% (Δ: {global_after_avg - global_affine_avg:+.2f}%)")
        print(f"Total improvement:     {global_after_avg - global_before_avg:+.2f}%")
    else:
        print(f"{'Corruption':<20} {'Before':>10} {'After':>10} {'Δ':>10}")
        print("-" * 50)
        for corruption in all_corruptions:
            before = results['per_corruption'][corruption]['before']
            after = results['per_corruption'][corruption]['after']
            print(f"{corruption:<20} {before:>10.2f}% {after:>10.2f}% {after-before:>+10.2f}%")
        print(f"\nOverall accuracy:")
        print(f"Before adaptation:     {global_before_avg:.2f}%")
        print(f"After adaptation:      {global_after_avg:.2f}%")
        print(f"Total improvement:     {global_after_avg - global_before_avg:+.2f}%")