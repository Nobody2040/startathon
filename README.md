1. PROJECT OVERVIEW
1.1 Problem Statement
Train a Semantic Segmentation Model using a synthetic desert dataset generated from Falcon's digital twin platform. The model must classify 10 different desert environment classes.
1.2 Classes (10 Categories)
Class ID	Class Name	Description
0	Trees	Desert trees and large vegetation
1	Lush Bushes	Green, healthy bushes
2	Dry Grass	Dried grass and vegetation
3	Dry Bushes	Dried bushes and shrubs
4	Ground Clutter	Soil, sand, small debris
5	Flowers	Desert flowers and blooming plants
6	Logs	Fallen trees and branches
7	Rocks	Stones and rock formations
8	Landscape	Background terrain
9	Sky	Sky area in images
1.3 Evaluation Criteria
Component	Points
IoU Score	80 points
Documentation Quality	20 points
TOTAL	100 points
________________________________________
2. DATASET INFORMATION
2.1 Dataset Source
The dataset is a synthetic desert environment generated from Falcon's digital twin platform.
2.2 Dataset Structure
text
dataset/
├── train/
│   ├── images/          # Training images (RGB)
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── masks/           # Training masks (class indices 0-9)
│       ├── mask_001.png
│       ├── mask_002.png
│       └── ...
└── val/
    ├── images/           # Validation images
    └── masks/            # Validation masks
2.3 Dataset Statistics
Split	Number of Images	Image Format	Mask Format
Training	2857	RGB JPG	PNG (0-9 values)
Validation	317	RGB JPG	PNG (0-9 values)
Total	3274	-	-
2.4 Mask Value Mapping
Each pixel in the mask contains a value from 0-9 corresponding to the class:
•	0 = Trees
•	1 = Lush Bushes
•	2 = Dry Grass
•	3 = Dry Bushes
•	4 = Ground Clutter
•	5 = Flowers
•	6 = Logs
•	7 = Rocks
•	8 = Landscape
•	9 = Sky

2.5 Class Distribution Analysis
Class                                Percentage	
Trees          : 4.33%	
   Lush Bushes    : 35.14%	
   Dry Grass      : 4.87%	
   Dry Bushes     : 3.24%	
   Ground Clutter : 1.81%	
   Flowers        : 33.72%	
   Logs           : 16.89%	
   Rocks          : 0.00%	
   Landscape      : 0.00%	
   Sky            : 0.00%	
3. MODEL ARCHITECTURE
Model Drive Link - https://drive.google.com/drive/folders/14_euT_nMhjU97vYgmN6vqP6mvUH7WYZz?usp=sharing
3.1 Primary Model: DeepLabV3+ with ResNet50 Encoder
We selected DeepLabV3+ architecture with a ResNet50 encoder pretrained on ImageNet for this segmentation task. DeepLabV3+ is widely regarded as one of the best architectures for semantic segmentation, especially for complex scenes with multiple objects [citation:3][citation:6].
3.2 Why DeepLabV3+?
Advantage	Explanation
✅ Atrous Convolution	Captures multi-scale context without losing resolution
✅ ASPP Module	Atrous Spatial Pyramid Pooling captures objects at multiple scales
✅ Encoder-Decoder Structure	Sharp object boundaries with rich semantic information
✅ State-of-the-Art	Top performance on multiple segmentation benchmarks
✅ Boundary Precision	Excellent at delineating object boundaries
3.3 Why ResNet50 Encoder?
Advantage	Explanation
✅ Pretrained on ImageNet	Already knows basic features (edges, textures, shapes)
✅ Deeper Architecture	50 layers capture more complex features
✅ Residual Connections	Avoids vanishing gradient problem
✅ Proven Performance	Excellent feature extraction capabilities
✅ Balance	Good trade-off between depth and computational cost
3.4 Model Configuration
python
Model Configuration:
├── Architecture: DeepLabV3+
├── Encoder: ResNet50
├── Encoder Weights: ImageNet (pretrained)
├── ASPP Rates: (6, 12, 18)
├── Input Channels: 3 (RGB)
├── Output Classes: 10
├── Total Parameters: 41,256,840
├── Trainable Parameters: 41,256,840
├── Activation Function: ReLU (hidden), Softmax (output)
└── Loss Function: CrossEntropyLoss
3.5 Architecture Diagram
text
Input Image (3xHxW)
       ↓
[ResNet50 Encoder]
  ├── Stem (7x7 conv, 64)
  ├── Layer 1 (256) - 3 blocks
  ├── Layer 2 (512) - 4 blocks
  ├── Layer 3 (1024) - 6 blocks
  └── Layer 4 (2048) - 3 blocks
       ↓
[ASPP Module]
  ├── 1x1 Convolution
  ├── 3x3 Conv rate=6
  ├── 3x3 Conv rate=12
  ├── 3x3 Conv rate=18
  └── Image Pooling
       ↓
[Decoder]
  ├── Upsample (x4)
  ├── Concatenate with low-level features
  ├── 3x3 Convolution
  └── Upsample (x4)
       ↓
[Final Convolution]
       ↓
Output Mask (10xHxW)
3.6 Alternative Architectures Tested
Model	Encoder	Parameters	IoU	Pros	Cons
DeepLabV3+	ResNet50	41.3M	0.XX	Best boundaries, context	Slower training
U-Net	ResNet34	24.4M	0.XX	Fast training	Less accurate boundaries
PSPNet	ResNet50	46.7M	0.XX	Good context	Memory intensive
FPN	ResNet34	23.5M	0.XX	Good for small objects	Complex
3.7 Key Innovations in DeepLabV3+
Atrous Convolution
Atrous convolution (dilated convolution) allows us to control the field-of-view and capture multi-scale context without increasing the number of parameters or computational cost.
text
Standard Convolution:          Atrous Convolution (rate=2):
┌───┬───┬───┐                 ┌───┬───┬───┐
│ X │   │ X │                 │ X │   │ X │
├───┼───┼───┤                 ├───┼───┼───┤
│   │   │   │                 │   │   │   │
├───┼───┼───┤                 ├───┼───┼───┤
│ X │   │ X │                 │ X │   │ X │
└───┴───┴───┘                 └───┴───┴───┘
Field-of-view: 3x3            Field-of-view: 5x5
ASPP (Atrous Spatial Pyramid Pooling)
ASPP applies multiple atrous convolutions with different rates to capture objects at different scales:
text
                  ┌─────────────────┐
                  │   Input Feature │
                  │      Map        │
                  └────────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ 1x1 Conv      │  │ 3x3 Conv      │  │ 3x3 Conv      │
│               │  │ rate=6        │  │ rate=12       │
└───────────────┘  └───────────────┘  └───────────────┘
        ↓                  ↓                  ↓
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Image Pooling │  │ 3x3 Conv      │  │   Concatenate │
│               │  │ rate=18       │  │                │
└───────────────┘  └───────────────┘  └───────┬───────┘
        ↓                  ↓                  │
        └──────────────────┼──────────────────┘
                           ↓
                  ┌─────────────────┐
                  │   1x1 Conv      │
                  │   (compress)    │
                  └─────────────────┘
4. INSTALLATION GUIDE
4.1 Prerequisites
Requirement	Version
Python	3.8 or higher
CUDA	11.0 or higher (for GPU)
PyTorch	2.0 or higher
RAM	16GB recommended
GPU Memory	8GB+ recommended
5. TRAINING PROCESS
5.1 Training Configuration
Parameter	Value
Epochs	30
Batch Size	6 (adjusted for DeepLabV3+ memory)
Learning Rate	0.0001 (lower for DeepLabV3+)
Optimizer	Adam (β1=0.9, β2=0.999)
Loss Function	CrossEntropyLoss
Learning Rate Scheduler	Polynomial Decay
Scheduler Power	0.9
Weight Decay	1e-4
5.2 Data Augmentation
To improve generalization, we applied these augmentations during training:
Augmentation	Probability	Description
Random Horizontal Flip	50%	Mirror image horizontally
Random Rotation	30%	Rotate by ±10 degrees
Random Scale	20%	Scale by 0.5-2.0
Color Jitter	30%	Adjust brightness, contrast
Normalization	Always	ImageNet mean/std
6. RESULTS & PERFORMANCE
6.1 Overall Performance
Metric	Score
Best Validation IoU	[YOUR IOU SCORE]
Hackathon Score (80%)	[YOUR SCORE]/80
Training Time	[YOUR TIME] minutes
Model Size	165 MB
Parameters	41.3 Million
6.2 Per-Class IoU Scores
Class	IoU Score
Trees          : 0.4393
   Lush Bushes    : 0.5003
   Dry Grass      : 0.0036
   Dry Bushes     : 0.0812
   Ground Clutter : 0.0000
   Flowers        : 0.5116
   Logs           : 0.9790
   Rocks          : 0.0000
   Landscape      : 0.0000
   Sky            : 0.0000
Mean IOU : 0.6161
Team Details
Team Modih
Name – Aarav Ahlawat
Email – unboxer2040@gmail.com
Github – Nobody2040
Name – Ankit Sharma
Email- as9560630638@gmail.com
Github – as9560630638-hash
Name – deeksha kaushal
GitHub – deenominator
Name – Arsh Yadav
Email – arshyadav4147@gmail.com
Institution
	
University	VIT Bhopal University
Event	Startathon Hackathon 2026
Date	February 18, 2026
Organizer	E-Cell VIT Bhopal


 ACKNOWLEDGMENTS
 Organizers
We would like to thank E-Cell VIT Bhopal for organizing this hackathon.
 Platform Providers
Special thanks to Falcon Platform for providing the synthetic desert dataset.
 Open Source Libraries
Library	Purpose
PyTorch	Deep learning framework
Segmentation Models PyTorch	Pre-trained DeepLabV3+ implementation
TorchMetrics	IoU and other metrics
Matplotlib	Visualization
NumPy	Numerical operations

