Face emotion recognizer based on Vision Transformer implementation.

## Training

The training script now includes common data augmentation and optional
focal loss to better handle class imbalance.

```
python train.py --epochs 10 --batch-size 32 --use-focal-loss
```

Use `python train.py -h` for a full list of available options.
