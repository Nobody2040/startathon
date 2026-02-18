# DESERT SEGMENTATION MODEL - PERFORMANCE REPORT
Date: 2026-02-18 19:18:51

## MODEL ARCHITECTURE
- Architecture: deeplabv3+
- Encoder: resnet50 (pretrained on ImageNet)
- Number of classes: 10
- Classes: Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky

## TRAINING CONFIGURATION
- Epochs: 30
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss

## RESULTS

### Overall Performance
- Best Validation IoU: 0.6161
- Hackathon Score (80%): 49.3/80

### Per-Class IoU

- Trees: 0.4393
- Lush Bushes: 0.5003
- Dry Grass: 0.0036
- Dry Bushes: 0.0812
- Ground Clutter: 0.0000
- Flowers: 0.5116
- Logs: 0.9790
- Rocks: 0.0000
- Landscape: 0.0000
- Sky: 0.0000

### Hardest Classes (Lowest IoU)
- Ground Clutter: 0.0000
- Rocks: 0.0000
- Landscape: 0.0000

### Easiest Classes (Highest IoU)
- Logs: 0.9790
- Flowers: 0.5116
- Lush Bushes: 0.5003

## FAILURE ANALYSIS

### Why hardest classes perform poorly:
1. **Ground Clutter** (IoU: 0.0000):
   - Small objects that are hard to detect
   - Might be confused with similar classes
   - Suggestion: Use FPN model which is better for small objects

2. **Rocks** (IoU: 0.0000):
   - Limited training examples
   - Visual similarity to other classes
   - Suggestion: Add more augmentation

3. **Landscape** (IoU: 0.0000):
   - Complex boundaries
   - Varying appearances
   - Suggestion: Try DeepLabV3+ for better boundaries

## VISUALIZATIONS
- Training curves: training_curves.png
- Per-class performance: per_class_iou.png
- Sample predictions: predictions.png

## REPRODUCTION STEPS
1. Install requirements: `pip install -r requirements.txt`
2. Run inference: See example below

```python
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Ensure truncated images are handled during inference too
import numpy as np
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load config
with open('submission/models/config.json', 'r') as f:
    model_config = json.load(f)

model_arch_name = model_config['model']
encoder_name = model_config['encoder']
num_classes_config = model_config['num_classes']

# Dynamically create and load model
if model_arch_name == 'deeplabv3+':
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None, # Set to None for inference unless you want to re-download
        in_channels=3,
        classes=num_classes_config,
    )
elif model_arch_name == 'unet':
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None, # Set to None for inference unless you want to re-download
        in_channels=3,
        classes=num_classes_config,
    )
# Add other model types (pspnet, fpn) as needed
else:
    raise ValueError("Unknown model architecture: " + model_arch_name)

model.load_state_dict(torch.load('submission/models/model.pth'))
model.eval()

# Define inference transform (should match validation transform from training)
inference_transform = A.Compose([
    A.Resize(512, 512), # Assuming 512x512 from training
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet normalization
    ToTensorV2(),
])

# Predict
image = Image.open('your_image.jpg').convert('RGB')
image_np = np.array(image) # Convert PIL Image to numpy array

# Apply transforms
transformed = inference_transform(image=image_np)
image_tensor = transformed['image'].unsqueeze(0) # Add batch dimension

# Move model to CPU if not on GPU (for general reproducibility)
model.to('cpu')

with torch.no_grad():
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1).squeeze().numpy()

# Example of how to visualize prediction
# import matplotlib.pyplot as plt
# plt.imshow(pred, cmap='tab10', vmin=0, vmax=num_classes_config-1)
# plt.title('Prediction')
# plt.axis('off')
# plt.show()
```
