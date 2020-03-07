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