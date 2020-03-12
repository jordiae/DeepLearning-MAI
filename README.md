# Deep Learning assignments (DL-MAI)

This is the repository containing the source code and experiments for the assignments of the Deep Learning course at the Master in Artificial Intelligence at UPC-BarcelonaTech. The code is based on PyTorch.

## Requirements

Provided Python3.7 and CUDA are already installed in the system, run:

```
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For obtaining the data of the CNNs assignment (mit67), run:

```
bash get-mit67.sh

```


## Convolutional Neural Networks

Preprocessing:

```
python src/cnn/preprocess.py

```

Creating experiment:

```
python src/cnn/create_experiment.py experiment_name [options]

```
```
options:
      '--arch', type=str, help='Architecture', default='BaseCNN'
      '--data', type=str, help='Dataset', default='256x256-split'
      '--epochs', type=int, help='Number of epochs', default=3
      '--lr', type=float, help='Learning Rate', default=0.001
      '--momentum', type=float, help='Momentum', default=0.9
      '--no-cuda', action='store_true', default=False, help='disables CUDA training'
      '--augment', action='store_true', default=True, help='enables data augmentation'
      '--optimizer', type=str, help='Optimizer', default='SGD'
      '--batch-size', type=int, help='Mini-batch size', default=2
      '--criterion', type=str, help='Criterion', default='cross-entropy'
      '--early-stop', action='store_true', help='Early stop in validation set with no patience'
```


Running experiment on cluster:

```
run_experiment.sh experiments/experiment_folder/ debug|main

```


Notes:

```
Mark src/ as sources root
BW images -> removed
malformatted image -> removed
RGBA, P -> RGB
stratified split 80, 10, 10. Try CV?
resize to 256x256 (not keeping aspect ratio), with filter. TODO: try Fully Convolutional.
256x256-split contains the preprocessed & split data.
TODO: add progress bar
TODO: add type annotations
TODO: add workers (parallelism)?
```


## Recurrent Neural Networks
