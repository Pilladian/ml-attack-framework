# Python 3.8.5

import argparse
import os
import torch
import torch.nn as nn
from torch.nn import parameter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


import attrinf
import meminf
import datasets
from Target import CNN


def attribute_inference_attack(target, dataset, attr, device):
    # Load Target Model
    print(f'\t\t[1.1] Load Target Model {target}')
    target_model = CNN().to(device)
    target_model.load_state_dict(torch.load(target, map_location=device))

    # Load dataset with specific attribute as label
    print(f'\t\t[1.2] Load {dataset} dataset with {attr} as label')
    raw_loader = attrinf.get_raw_attack_dataset(dataset, attr)

    # Sample AttributeInferenceAttackDataset based on raw_loader and Target Model
    print('\t\t[1.3] Sample Attack Dataset using Target Model')
    train_attack_loader, eval_attack_loader, test_attack_loader = attrinf.get_attack_loader(target_model, raw_loader, device)

    # Create Attack Model
    print('\t\t[1.4] Create Attack Model')
    parameter = {}
    parameter['n_input_nodes'] = 100
    parameter['n_hidden_nodes'] = 32
    parameter['n_output_nodes'] = 5 if dataset == 'utkface' else 5
    parameter['lr'] = 0.001
    parameter['activation_fn'] = nn.Sigmoid()
    parameter['loss_fn'] = nn.CrossEntropyLoss()
    parameter['epochs'] = 100

    model = attrinf.MLP(parameter)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameter['lr'])

    # Train Attack Model
    attrinf.train_attack_model(model, train_attack_loader, parameter['epochs'], parameter['loss_fn'], optimizer, device)

    # Perform Attribute Inference Attack
    print(f'\t\t[1.6] Perform Attribute Inference Attack on Target Model: {attrinf.eval_attack_model(model, test_attack_loader, device):0.4f} Acc.')
    exit(0)

def membership_inference_attack(target, dataset, device):
    # get dataloader
    train_shadow_loader, test_shadow_loader, train_target_loader, test_target_loader = meminf.get_data(dataset)

    # get test data for attack model
    print("\t\t[2.1] Load Target model")
    target_model = CNN().to(device)
    target_model.load_state_dict(torch.load(target, map_location=device))
    print("\t\t[2.2] Sample Attack Test Data using Target")
    test_data, test_label = meminf.get_attack_test_data(target_model, train_target_loader, test_target_loader, device)


    # create shadown model
    print("\t\t[2.3] Create and Train Shadow model")
    model_shadow = CNN().to(device)
    # train shadow model
    model_shadow = meminf.train_shadow_model(model_shadow, device, train_shadow_loader)

    # get training data for attack model
    print("\t\t[2.4] Sample Attack Training Data using Shadown Model")
    train_data, train_label = meminf.get_attack_train_data(model_shadow, train_shadow_loader, test_shadow_loader, device)

    # create attack datasets
    train_data = StandardScaler().fit_transform(train_data)
    test_data = StandardScaler().fit_transform(test_data)

    # dataloader for attack  model
    trainset = datasets.MIADataset(train_data, train_label)
    testset = datasets.MIADataset(test_data, test_label)
    train_loader = DataLoader(trainset, batch_size=50, shuffle=True)
    test_loader = DataLoader(testset, batch_size=50, shuffle=True)    

    # attack model
    model_attack = meminf.BCNet(input_shape=train_data.shape[1]).to(device)
    print("\t\t[2.5] Train Attack Model")
    meminf.train_attack_model(model_attack, train_loader, device)

    # Evaluate attack
    print("\t\t[2.6] Perform Membership Inference Attack")
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1).detach().cpu().numpy()
            outputs = model_attack(data)
            outputs = outputs.reshape(-1).detach().cpu().numpy().round()
            
            for i in range(min(32, data.size(0))):
                label = labels[i]
                pred = outputs[i]
                if (label == pred and label != 0):
                    tp += 1
                elif (label != pred and label != 0):
                    fn += 1
                elif (label != pred and label == 0):
                    fp += 1
                if (label == pred):
                    n_correct += 1
                n_samples += 1

        acc = 100.0 * n_correct / n_samples
        
    return acc 

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
                        help="Choose between [UTKFace, CIFAR10]")

    parser.add_argument("--device",
                        default='cpu',
                        help="Provide cuda device")

    parser.add_argument("--inferred-attribute",
                        default='',
                        help="Important for Attribute Inference Attack: Attribute that will be inferred")

    args = parser.parse_args()

    # Check for correct dataset
    if args.dataset.lower() not in ['utkface', 'cifar10']:
        print(f'\n\t[!] Error ocurred: No such dataset \"{args.dataset}\"\n')
        exit(0)

    # results
    results = { 'attribute': 0, 
                'membership': 0, 
                'modelinv': 0}

    # run attribute inference attack
    if args.dataset != 'cifar10':
        if args.inferred_attribute == '':
            print(f'\n\t[!] Error ocurred: --inferred-attribute cannot be empty. Use --help / -h for printing the help page\n')
            exit(0)
        print("\n\t[1] Attribute Inference Attack")
        results['attribute'] = attribute_inference_attack(args.target, args.dataset.lower(), args.inferred_attribute, args.device) 
    else:
        print("\n\t[1] Attribute Inference Attack will not be performed, \n\t    since a dataset containing images of numbers does not have any attribute that could be inferred")

    # run membership inference attack
    print("\n\t[2] Membership Inference Attack")
    results['membership'] = membership_inference_attack(args.target, args.dataset, args.device) 

    # run model inversion attack
    print("\n\t[3] Model Inversion Attack")
    results['modelinv'] = model_inversion_attack(args.target, args.dataset) 

    # Output
    print(f"\n\n Attack Accuracies: \n \
                \n\tAttribute-Inference-Attack: \t{results['attribute']:0.2f} \
                \n\tMembership-Inference-Attack: \t{results['membership']:0.2f} \
                \n\tModel-Inversion-Attack: \t{results['modelinv']:0.2f}\n\n")