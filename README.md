# Fine-tuning SAM with FFT-based Self-Attention for Nail Segmentation

This project demonstrates how to fine-tune Meta's Segment Anything Model (SAM) with a novel FFT-based self-attention mechanism for nail segmentation tasks. The implementation replaces traditional self-attention layers with Fast Fourier Transform (FFT) based attention to potentially improve computational efficiency.

## Key Features

- **FFT-based Self-Attention**: Custom implementation replacing standard multi-head attention with FFT-based attention mechanism
- **SAM Model Integration**: Utilizes Meta's SAM (ViT-B) as the backbone encoder
- **Automatic Dataset Handling**: Automated data loading and preprocessing for nail segmentation datasets
- **Cross-platform Support**: Compatible with CUDA, MPS (Apple Silicon), and CPU devices
- **Binary Segmentation**: Optimized for nail vs. background classification

## Architecture Overview

### FFT Self-Attention Mechanism
The core innovation lies in the `FFTSelfAttention` class that:
- Performs FFT on query and key projections
- Computes attention through complex multiplication in frequency domain
- Applies inverse FFT to return to spatial domain
- Maintains compatibility with original attention interface

### Modified SAM Architecture
- **Base Model**: SAM ViT-B encoder with pre-trained weights
- **Custom Head**: Added segmentation head with batch normalization and ReLU activations
- **Output**: Binary classification (background=0, nail=1)

## Requirements

```bash
pip install torch torchvision
pip install segment-anything
pip install Pillow numpy tqdm
```

## Dataset Structure

The code expects the following directory structure:
```
nail_seg/
├── trainset_nails_segmentation/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.jpg  # Binary mask
│       ├── image2.jpg
│       └── ...
```

## Usage

### Basic Training
```python
python main.py
```

### Key Parameters
- **Image Size**: 1024x1024 (SAM's native resolution)
- **Batch Size**: 1 (memory-optimized)
- **Learning Rate**: 1e-5
- **Epochs**: 5 (default)
- **Train/Val Split**: 80/20

## Model Components

### FFTSelfAttention Class
```python
class FFTSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        # Implements FFT-based attention mechanism
        # Replaces traditional scaled dot-product attention
```

### ModifiedSAM Class
```python
class ModifiedSAM(nn.Module):
    def __init__(self, sam_model, num_classes=2):
        # Adds segmentation head to SAM encoder
        # Handles feature upsampling and classification
```

## Training Process

1. **Model Initialization**: Downloads SAM ViT-B checkpoint automatically
2. **Attention Replacement**: Recursively replaces all MultiheadAttention modules with FFTSelfAttention
3. **Data Loading**: Automatically handles image-mask pair loading and preprocessing
4. **Training Loop**: Standard supervised learning with CrossEntropyLoss
5. **Evaluation**: Dice score computation for segmentation quality assessment

## Evaluation Metrics

- **Dice Score**: Primary metric for segmentation overlap assessment
- **Loss Tracking**: CrossEntropyLoss monitoring during training

## Output

The trained model will be saved as `fft_sam_nails_segmentation.pth` with final Dice score reporting.

## Technical Details

### Memory Optimization
- Batch size of 1 to handle large 1024x1024 images
- Efficient data loading with custom collate function
- MPS support for Apple Silicon optimization

### FFT Implementation
- Uses `torch.fft.rfft` for real-valued FFT computation
- Complex conjugate multiplication for frequency domain operations
- Maintains gradient flow for backpropagation

## Potential Applications

- Medical imaging segmentation
- Computer vision preprocessing
- Attention mechanism research
- Mobile-optimized segmentation models

## Future Improvements

- Multi-class segmentation support
- Advanced data augmentation
- Hyperparameter optimization
- Quantitative attention analysis
- Performance benchmarking against standard attention

## License

This project builds upon Meta's SAM model. Please refer to the original SAM license for usage terms.

## Citation

If you use this work, please cite:
```bibtex
@misc{fft_sam_nails,
  title={Fine-tuning SAM with FFT-based Self-Attention for Nail Segmentation},
  author={Your Name},
  year={2025}
}
```