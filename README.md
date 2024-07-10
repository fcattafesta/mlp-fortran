# mlp-fortran
Code for the project of the Scientific Programming 2 course.

## Description
This project is a simple implementation of a Multilayer Perceptron (MLP) in Fortran. The MLP is used to learn to recognize handwritten digits from the MNIST dataset.

## Preparation
To extract the MNIST dataset, run the following command:
```
cd data && tar -xvf mnist_data.tar.gz && cd ..
```
Compile the code using the provided Makefile:
```
make
```

## Usage
To train the MLP, run the following command:
```
./bin/train [options]
```
A list of available options can be found by running:
```
./bin/train --help
```
Once the MLP has been trained, loss and accuracy plots can be generated by running:
```
./scripts/plotResults.py <metrics_file>
```
where `<metrics_file>` is the file containing the metrics of the trained MLP (e.g. `data/metrics.csv`) generated during training.

## Requirements
The code requires the `gfortran` compiler. For the plotting script, `python3` is required. To install the needed Python packages, run:
```
pip install -r requirements.txt
```
