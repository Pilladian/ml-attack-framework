# Python 3.8.5

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import attrinf
from Target import CNN
from attrinf import AttributeInferenceAttack


def attribute_inference_attack(target, dataset, attr):
    # Load Target Model  
    print("\t\t[1.1] Load Target model")
    device = 'cpu'
    model = CNN().to(device)
    model.load_state_dict(torch.load(target, map_location=device))

    # Create Attack Models Hyperparameter
    if dataset == "utkface":
        if attr == "race":
            num_classes = 5
        elif attr == "age":
            num_classes = 117
        elif attr == "gender":
            num_classes = 2
    hyperparameter = {'epochs': 200,
                      'lr': 0.01,
                      'batch_size': 64,
                      'feat_amount': 256,
                      'num_hnodes': 64,
                      'num_classes': num_classes,
                      'activation_fn': nn.Sigmoid(),
                      'loss_fn': F.cross_entropy,
                      'dropout': 0.5}

    # Create Attack Model and Sample Attack Dataset
    """
    [1] Create csv-files based on dataset, partition and inferred attribute
    [2] Load data into train_loader, eval_loader, test_loader
    [3] Structure of data in loader:
        - <file_name>,<value_for_attr>
        - E.g.: 20_0_4_20210202.jpg, 4
            - 20       : Age    : 20 years old
            - 0        : Gender : Male
            - 4        : Race   : Black
            - 20210202 : Date   : 02/02/2021
            - 4        : Label  : Race
    """
    print("\t\t[1.2] Sample Attack Dataset using Target Model")
    data_loader = attrinf.sample_attack_dataset(dataset, attr)
    attack = AttributeInferenceAttack(model,
                                      device=device,
                                      params=hyperparameter)
    attack.process_raw_data(data_loader)

    # Run Attack on test-set of Attacker Dataset
    print("\t\t[1.3] Perform Attribute Inference Attack")
    return attack.run()


def membership_inference_attack(target, dataset):
    return 0


def model_inversion_attack(target, dataset):
    return 0


if __name__ == '__main__':
    os.system('clear')
    print("\n Privacy Enhancing Technologies - Semester Project")
    
    # collect command line arguments
    parser = argparse.ArgumentParser(description='Privacy Enhancing Technologies - Semester Project')

    parser.add_argument("--target",
                        required=True,
                        help="File path to target model that should be tested")

    parser.add_argument("--dataset",
                        required=True,
                        help="Choose between [UTKFace, ...]")

    parser.add_argument("--inferred-attribute",
                        required=True,
                        help="Important for Attribute Inference Attack: Attribute that will be inferred")

    args = parser.parse_args()

    # Check for correct dataset
    if args.dataset.lower() not in ['utkface']:
        print(f'Error ocurred: No such dataset \"{args.dataset}\"')
        exit(0)

    # results
    results = { 'attribute': 0, 
                'membership': 0, 
                'modelinv': 0}

    # run attribute inference attack
    print("\n\t[1] Attribute Inference Attack")
    results['attribute'] = attribute_inference_attack(args.target, args.dataset.lower(), args.inferred_attribute) 

    # run membership inference attack
    print("\n\t[2] Membership Inference Attack")
    results['membership'] = membership_inference_attack(args.target, args.dataset) 

    # run model inversion attack
    print("\n\t[3] Model Inversion Attack")
    results['modelinv'] = model_inversion_attack(args.target, args.dataset) 

    # Output
    print(f"\n\n Attack Accuracies in %: \n \
                \n\tAttribute-Inference-Attack: \t{results['attribute']*100:0.2f} \
                \n\tMembership-Inference-Attack: \t{results['membership']*100:0.2f} \
                \n\tModel-Inversion-Attack: \t{results['modelinv']*100:0.2f}\n\n")