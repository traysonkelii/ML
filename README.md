# ML toolkit

## Basic command
```
python -m toolkit.manager -L baseline -A datasets/iris.arff -E training
```

## Help
```
python -m toolkit.manager --help
```

## Creating learners
See the baseline_learner.py and its BaselineLearner class for an example of the format of the learner. In particular, new learners will need to override the train() and predict() functions of the SupervisedLearner base class.
