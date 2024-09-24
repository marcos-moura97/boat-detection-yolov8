# boat-detection-yolov8

Use of Fractional Factorial Design (FFD) and Analysis of Variance (ANOVA) to evaluate best YOLOv8 configuration for a boat detection task

Procedure:

Generate experiments with:

```
python experiments_design.py
```

This will generate a csv with the experiments.

After that, train the models with:

```
python train_models.py
```

When it finishes all the models training, just run the analysis:

```
python analysis.py
```
