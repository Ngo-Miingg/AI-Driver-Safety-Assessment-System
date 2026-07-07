# ViT-DD Behavior Training

Notebook: `kaggle/vitdd_driver_behavior_kaggle_train.ipynb`

Use this for the driver-behavior branch of the MDDAD multimodal system.

## Dataset format

Prepare a Kaggle dataset using one of these layouts:

```text
train/c0/*.jpg
train/c1/*.jpg
...
valid/c0/*.jpg
test/c0/*.jpg
```

or:

```text
c0/*.jpg
c1/*.jpg
...
c9/*.jpg
```

The second layout is auto-split into train/valid inside the notebook.

## Output

After training, download:

```text
/kaggle/working/vitdd_behavior_training_results.zip
```

Copy `vitdd_behavior.pt` to:

```text
D:\Work\HocKy3\ComputerVision\MDDAD\models\vitdd_behavior.pt
```

Then open `/behavior-test` in the Flask app and test an image or video.
