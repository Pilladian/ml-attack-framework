# Privacy Enhancing Technologies - 2021
> Universität des Saarlandes - PETs 2021 - Semester Project

## Setup Cloned Repository
First create a virtual environment using:
```bash
virtualenv -p python3 PETS-2021/
```
Activate the virtual environment like this:
```bash
cd PETS-2021/
source bin/activate
```
Now install all requirements using:
```bash
pip install -r requirements.txt
```

## Run the Attacks
Since we already provide pre-trained target models, you can run the attacks using:
```bash
python run-attacks.py --target ./Target/target-model-utkface.pth --dataset utkface --inferred-attribute race --device cuda
```
or 
```bash
python run-attacks.py --target ./Target/target-model-cifar10.pth --dataset cifar10 --device cuda
```

## Train Target Models
You can also train your own target models using:
```bash
python train-target.py --dataset utkface --epochs 100 --device cuda
```
or
```bash
python train-target.py --dataset cifar10 --epochs 100 --device cuda
```

## Functionality of the Code

### Attribute Inference Attack
First we train our target model using `train-target.py´.
We then sample our attack dataset D in the following way:

- Load images as input and the provided attribute as labels
- Query target model on inputs to obtain the posteriors of the last hidden layers
- D now contains pairs (i,j) with i = posterior and j = attribute

We then use D to train our attack model.

### Membership Inference Attack
First we train our target model using `train-target.py´.
We then train our shadow model using the shadow dataset (shadow dataset is from the same domain like the target training set).
We then sample our attack dataset D in the following way:

- Load images from the target training set as positive samples and images that haven't been used for training the target as negative samples
- Query target model on inputs to obtain the posteriors of the prediction
- D now contains pairs (i,j) with i = posterior and j = 1 (posterior belongs to image that has been used to train the target model) or j = 0 (posterior belongs to image that has not been used to train the target model)

We then use D to train our attack model.