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