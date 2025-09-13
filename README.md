# Face Emotion Recognition with Vision Transformer

This project implements a facial expression recognition pipeline using a Vision Transformer (ViT). It serves as a compact reference for leveraging modern transformer architectures in computer vision tasks such as emotion classification.

## Features
- **Vision Transformer Backbone** built with [`timm`](https://github.com/huggingface/pytorch-image-models).
- **Data Augmentation** including resize, random horizontal flip, rotation, normalization, and random erasing to improve model robustness.
- **Class Imbalance Handling** via class weights or optional focal loss.
- **Progress Monitoring** with `tqdm` progress bars during training and validation.

## Project Structure
```
.
├── train.py          # Training script
├── requirements.txt  # Python dependencies
├── Training/         # Training images organized by class
├── PublicTest/       # Validation images organized by class
├── PrivateTest/      # Test images organized by class
└── data/             # Additional data resources
```
Each dataset directory (`Training`, `PublicTest`, `PrivateTest`) contains subfolders named after emotion categories such as `anger`, `happy`, or `sad`, with image files inside.

## Installation
1. Clone the repository and install dependencies:
   ```bash
   git clone <repository-url>
   cd Face_Recognition
   pip install -r requirements.txt
   ```

## Usage
### Training
Run the training script with optional arguments:
```bash
python train.py --epochs 10 --batch-size 32 --pretrained --use-focal-loss
```
Use `python train.py -h` to see the full list of available options.

The trained model weights will be saved to `vit_fer_model.pth`.

### Dataset Format
Ensure the dataset follows the folder structure shown above. Each emotion class should have its own directory containing the corresponding images.

## License
This project is provided for educational purposes. Please ensure that you have the right to use any dataset employed for training or evaluation.

## Acknowledgements
- [timm](https://github.com/huggingface/pytorch-image-models) for Vision Transformer implementations
- PyTorch team and community for the deep learning framework

