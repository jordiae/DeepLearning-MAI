# Deep Learning assignments (DL-MAI)

This is the repository containing the source code for the assignments of the Deep Learning course at the Master in Artificial Intelligence at UPC-BarcelonaTech. The code is based on PyTorch.

## Requirements

Provided Python3.7 and CUDA are already installed in the system, run:

```
bash setup.sh
```

This script activates the virtual environment, downloads the required dependencies and marks 'src' as the sources root.

For obtaining the data of the CNNs assignment (mit67), run:

```
bash get-mit67.sh

```

## Convolutional Neural Networks

### Instructions

Preprocessing:

```
python src/cnn/preprocess.py

```
In the preprocessing script, we:
    
    - Print some statistics of the dataset and the preprocessing procedure.
    - Remove a few malformatted images.
    - Remove a few BW images.
    - Convert a few PNG into JPG.
    - Perform a stratified split into train, validationn and test (80-10-10).
    - By default, we resize all images to 256x256, even if our model is input-size-agnostic.

Creating experiment:

```
python src/cnn/create_experiment.py experiment_name [options]

```
This script generates a new directory for the experiment in the experiments/ directory. For each experiment, bash, 
batch (Windows) and Slurm scripts are generated, and they must be executed from the corresponding experiment directory.
The text logs (.out, .err and .log) files, as well as the Tensorboard logs, will always be stored in the directory of
the corresponding experiment.

For directly launching the training (without any training), even if that is not recommended, run:
```
python src/cnn/train.py [options]

```

The training options are the following (note that the defaults have been optimized for performance in the validation set):
```
parser = argparse.ArgumentParser(description='Train a CNN for mit67')
parser.add_argument('--arch', type=str, help='Architecture', default='PyramidCNN')
parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.0001)
parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--no-augment', action='store_true', help='disables data augmentation')
parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
parser.add_argument('--criterion', type=str, help='Criterion', default='label-smooth')
parser.add_argument('--smooth-criterion', type=float, help='Smoothness for label-smoothing', default=0.1)
parser.add_argument('--early-stop', type=int,
                    help='Patience in early stop in validation set (-1 -> no early stop)', default=6)
parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.001)

parser.add_argument('--kernel-size', type=int, help='Kernel size', default=3)
parser.add_argument('--dropout', type=float, help='Dropout in FC layers', default=0.25)
parser.add_argument('--no-batch-norm', action='store_true', help='disables batch normalization')
parser.add_argument('--conv-layers', type=int, help='N convolutional layers in each block', default=2)
parser.add_argument('--conv-blocks', type=int, help='N convolutional blocks', default=5)
parser.add_argument('--fc-layers', type=int, help='N fully-connected layers', default=2)
parser.add_argument('--initial-channels', type=int, help='Channels out in first convolutional layer', default=16)
parser.add_argument('--no-pool', action='store_true', help='Replace pooling by stride = 2')

parser.add_argument('--autoencoder', action='store_true', help='Train autoencoder instead of classification')

```

The default options are the ones that we observed to perfom better in our experiments.

Notice that the ```--autoencoder``` option allows to unsupervisedly pre-train an autoencoder. For doing so, launch an
experiment with the desired configuration and the ```--autoencoder``` option. Then, move the generated
```checkpoint_best.pt``` to the experiment directory, change its name to ```checkpoint_last.pt``` and start the training
procedure with the same architecture options, but without the ```--autoencoder``` option. The training script will
automatically start the training from the pre-trained encoder instead of from scratch.

For evaluating a model:
```
python src/cnn/evaluate.py [options]

```
The evaluation options are the following (notice that ensembles are implemented by providing more than one checkoint):
```
parser.add_argument('--arch', type=str, help='Architecture')
parser.add_argument('--models-path', type=str, help='Path to model directory',nargs='+')
parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',  help='Checkpoint name')
parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
parser.add_argument('--subset', type=str, help='Data subset', default='test')
parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=2)
```

### Experiments highlights

We conducted as many as 150 experiments in the CTE-POWER cluster (V100 GPUs) at the Barcelona Supercomputing Center.

We used the mit67 dataset (<http://web.mit.edu/torralba/www/indoor.html>), a well-known indoor scene recognition. It is
challening because it only contains 15,620 images, and by manually inspecting the 67 (imbalanced) classes, we can
observe that they are not easily distinguishable. Notice that for this assignment we were not supposed to use transfer
learning from another dataset (eg. VGG16), which made the problem more difficult.

By evaluating the different configurations in the validation set and comparing the results, we highlight the following
conclusions:
    
    - Autoencoder pre-training did not lead to better results.
    - Since the dataset is tiny, data augmentation was key to improve our results.
    - After performing a grid search on a number of architectural hyperparameters, we found a model with 5 convolutional
    blocks, with 2 convolutional layers per block with a kernel size of 3, and 2 fully-connected layers to be the best
    performant model.
    - Our models starts with 16 channels and doubles the number of channels for each block, while the image size is
    divided by 2 with pooling. This is why we call the architecture 'PyramidCNN'. Both 32 and 8 initial channels,
    performed worse, leading to over and underfitting, respectively.
    - Using stride instead of pooling did not lead to any improvement (actually, it performed worse).
    - Batch normalization remarkably improved the results.
    - For regularizing, we found that a dropout (only in the convolutional layers) of 0.25 and a weight decay of 0.001
    were the best options. We had to add some patience to the early stopping procedure. A label smoothing of 0.1 was
    also very useful.
    - Ensembling independently trained models reulted in a gain of about 8 points in accuracy.
    - Since the images present in the dataset have global features, we tried to incorporate Non-Local blocks
    (<https://arxiv.org/abs/1711.07971>), but we did not observe any gains.

The best found configuration, corresponding to the default values in the ```train.py``` script, obtained an accuracy of  58  in the validation set. Then, we build an ensemble of 10 independently trained classifiers with the same configuration, and obtained a validation accuracy of 64. We selected this ensemble as our final model, and obtained a test accuracy of 63.

## Recurrent Neural Networks
