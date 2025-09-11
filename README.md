Face emotion recognizer based on Vision Transformer implementation.

## Training

The training script now includes common data augmentation and optional
focal loss, label smoothing, MixUp, and CutMix to better handle class
imbalance and improve generalization. An optional class-balanced sampler
(`--balance-sampler`) can further mitigate dataset imbalance.

```
python train.py --epochs 10 --batch-size 32 --label-smoothing 0.1 --mixup-alpha 0.2
# or
python train.py --epochs 10 --batch-size 32 --cutmix-alpha 1.0
```

Use `--use-focal-loss` to enable focal loss instead of cross entropy,
`--mixup-alpha` or `--cutmix-alpha` to control augmentation strength
(set to 0 to disable; they are mutually exclusive), and `--balance-sampler`
to oversample minority classes. Run `python train.py -h` for a full list of
available options.

Training metrics are logged for TensorBoard. Specify `--log-dir` to
choose the log location (default `runs`) and launch TensorBoard via:

```
tensorboard --logdir runs
```
