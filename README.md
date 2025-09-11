Face emotion recognizer based on Vision Transformer implementation.

## Training

The training script now includes common data augmentation and optional
focal loss, label smoothing, and MixUp to better handle class imbalance
and improve generalization.

```
python train.py --epochs 10 --batch-size 32 --label-smoothing 0.1 --mixup-alpha 0.2
```

Use `--use-focal-loss` to enable focal loss instead of cross entropy and
`--mixup-alpha` to control MixUp strength (set to 0 to disable). Run
`python train.py -h` for a full list of available options.

Training metrics are logged for TensorBoard. Specify `--log-dir` to
choose the log location (default `runs`) and launch TensorBoard via:

```
tensorboard --logdir runs
```
