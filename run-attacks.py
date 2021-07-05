# Python 3.8.5

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import attrinf
from Target import CNN
from attrinf import AttributeInferenceAttack


def attribute_inference_attack(target, dataset, attr, device):
    # Load Target Model  
    print("\t\t[1.1] Load Target model")
    model = CNN().to(device)
    model.load_state_dict(torch.load(target, map_location=device))

    # Create Attack Models Hyperparameter
    hyperparameter = {'epochs': 200,
                      'lr': 0.01,
                      'batch_size': 64,
                      'feat_amount': 256,
                      'num_hnodes': 64,
                      'num_classes': attrinf.get_num_classes(dataset, attr),
                      'activation_fn': nn.Sigmoid(),
                      'loss_fn': F.cross_entropy,
                      'dropout': 0.5}

    # Create Attack Model and Sample Attack Dataset
    print("\t\t[1.2] Load selected Dataset")
    data_loader = attrinf.sample_attack_dataset(dataset, attr)
    
    # Sample Attack Dataset
    print("\t\t[1.2] Sample Attack Dataset using Target Model")
    attack = AttributeInferenceAttack(model, device=device, params=hyperparameter)
    attack.process_raw_data(data_loader)
    os.system('rm attrinf-utkface-race.csv')

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

    parser.add_argument("--device",
                        default='cpu',
                        help="Provide cuda device")

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
    results['attribute'] = attribute_inference_attack(args.target, args.dataset.lower(), args.inferred_attribute, args.device) 

    # run membership inference attack
    print("\n\t[2] Membership Inference Attack")
    results['membership'] = membership_inference_attack(args.target, args.dataset) 

    # run model inversion attack
    print("\n\t[3] Model Inversion Attack")
    results['modelinv'] = model_inversion_attack(args.target, args.dataset) 

    # Output
    print(f"\n\n Attack Accuracies: \n \
                \n\tAttribute-Inference-Attack: \t{results['attribute']:0.2f} \
                \n\tMembership-Inference-Attack: \t{results['membership']:0.2f} \
                \n\tModel-Inversion-Attack: \t{results['modelinv']:0.2f}\n\n")