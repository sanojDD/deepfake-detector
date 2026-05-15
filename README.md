# Fake vs Real Face Detection

## Overview
This project implements a binary image classifier to distinguish between fake (AI-generated) and real human faces using transfer learning with EfficientNet-B0.

## Model Architecture
- **Base Model**: EfficientNet-B0 pre-trained on ImageNet
- **Frozen Layers**: All EfficientNet backbone layers (preserves pre-trained features)
- **Custom Classification Head**:
  - Linear(1280 → 512)
  - ReLU activation
  - Dropout(0.3)
  - Linear(512 → 2)  # Output: [fake, real]

## Preprocessing
- Resize images to 224×224 pixels
- Convert to tensor
- Normalize using ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
