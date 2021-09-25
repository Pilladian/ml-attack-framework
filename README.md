# Machine Learning Framework
> Privacy Enhancing Technologies - 2021 --
> Universität des Saarlandes - PETs 2021 - Semester Project

## Participants
- Philipp Zimmermann (s8phzimm@stud.uni-saarland.de)
- Mohammed Raihan Hussain (s8mmhuss@stud.uni-saarland.de)
- HTMA Riyadh (s8htriya@stud.uni-saarland.de)

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
---

## Evaluation of Semester Project

### Phase 1: Use our own trained target models
- All 3 attacks on UTKFace dataset and UTKFace-Target model
    ```bash
    python run-attacks.py --target ./Target/target-model-utkface.pth --dataset utkface --inferred-attribute race --device cuda
    ```
    
- All 3 attacks on AT&T dataset and AT&T-Target model
    ```bash
    python run-attacks.py --target ./Target/target-model-att.pth --dataset att --inferred-attribute glasses --device cuda
    ```

- Membership Inference Attack and Model Inversion attack on CIFAR10 dataset and CIFAR10-Target model
    ```bash
    python run-attacks.py --target ./Target/target-model-cifar10.pth --dataset cifar10 --device cuda
    ```

- Membership Inference Attack and Model Inversion attack on MNIST dataset and MNIST-Target model
    ```bash
    python run-attacks.py --target ./Target/target-model-mnist.pth --dataset mnist --device cuda
    ```

### Phase 2: Use your own trained target models
Since we need to define the model class befor we can load the target model ([https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)) to perform our attacks please use our class `CNN()` located in [Target/\_\_init__.py](Target/__init__.py). You can simply use our provided script [./train-target.py](train-target.py) to train your own target models. 
- UTKFace dataset
    ```bash
    python train-target.py --dataset utkface --epochs 100 --device cuda
    ```
- CIFAR10 dataset
    ```bash
    python train-target.py --dataset cifar10 --epochs 100 --device cuda
    ```
- MNIST dataset
    ```bash
    python train-target.py --dataset mnist --epochs 100 --device cuda
    ```
- AT&T dataset
    ```bash
    python train-target.py --dataset att --epochs 100 --device cuda
    ```

**[ ! ] Please note the following**

Since the argument you provide to our program is the used dataset, we need you to follow the steps to create the correct input for our programs.

- create a new folder `dataset`
- create directories `dataset/train`, `dataset/eval` and `dataset/test`
- put samples used to **train** the target model in `dataset/train/` (target-members)
- put samples used to **evaluate** the target model in `dataset/eval/` (target-non-members)
- put samples that will be used to **train** and **evaluate** the shadow model in `dataset/test/` (shadow-members and shadow-non-members)
- rename `dataset` into one of [`CIFAR10`, `MNIST`, `UTKFace`, `ATT`] depending on your used samples 
- move the created dataset into [datasets](datasets/) replacing the current split

It is important that the samples you provide fullfil the requirements listed [here](datasets/README.md). You can now train your target models and after that run our implemented attacks against them.

---

## Functionality of the Code

### Attribute Inference Attack
We sample our attack dataset D in the following way:

- Load images as input and the provided attribute as labels
- Query target model on inputs to obtain the posteriors of the last hidden layers
- D now contains pairs (i,j) with i = posterior and j = attribute

We then use D to train our attack model.

### Membership Inference Attack
We train our shadow model using the shadow dataset (shadow dataset is from the same domain like the target training set).
We sample our attack dataset D in the following way:

- Load images from the target training set as positive samples and images that haven't been used for training the target as negative samples
- Query target model on inputs to obtain the posteriors of the prediction
- D now contains pairs (i,j) with i = posterior and j = 1 (posterior belongs to image that has been used to train the target model) or j = 0 (posterior belongs to image that has not been used to train the target model)

We then use D to train our attack model.

### Model Inversion
We use the Adam Optimizer to optimize a random vector `v` such that the prediction of the target model `f(v)` equals the given label `y`. 
We do this for each class, the dataset provides.
