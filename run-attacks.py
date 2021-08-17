# Python 3.8.5

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import torch
import os

from Target import CNN
import attrinf
import meminf
import modinv
import datasets

import matplotlib.pyplot as plt
from PIL import Image
import numpy

def lookup(dataset):
    datasets = {'utkface': 'UTKFace',
                'cifar10': 'CIFAR10',
                'mnist': 'MNIST',
                'att': 'ATT'}
    
    return datasets[dataset]

def feature_size(dataset):
    feature_sizes = {'utkface': 2,
                     'cifar10': 10,
                     'mnist': 10,
                     'att': 40}
    
    return feature_sizes[dataset]

def attribute_inference_attack(target, dataset, attr, device):
    # Load Target Model
    print(f'\t     [1.1] Load Target Model {target}')
    target_model = CNN(feature_size(dataset)).to(device)
    target_model.load_state_dict(torch.load(target, map_location=device))

    # Load dataset with specific attribute as label
    print(f'\t     [1.2] Load ./datasets/{lookup(dataset)}/* with {attr} as label')
    raw_loader = attrinf.get_raw_attack_dataset(dataset, attr)

    # Sample AttributeInferenceAttackDataset based on raw_loader and Target Model
    print('\t     [1.3] Sample Attack Dataset using Target Model')
    train_attack_loader, eval_attack_loader, test_attack_loader = attrinf.get_attack_loader(target_model, raw_loader, device)

    # Create Attack Model
    print('\t     [1.4] Create Attack Model')
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
    attack_acc = attrinf.eval_attack_model(model, test_attack_loader, device)
    print(f'\t     [1.6] Perform Attribute Inference Attack on Target Model: {attack_acc:0.4f} Acc.')

    return attack_acc

def membership_inference_attack(target, dataset, device):
    # get test data for attack model
    print(f'\t     [2.1] Load Target Model {target}')
    target_model = CNN(feature_size(dataset)).to(device)
    target_model.load_state_dict(torch.load(target, map_location=device))
    
    # get dataloader
    print(f'\t     [2.2] Load provided dataset')
    print(f'\t\t   [2.2.1] ./datasets/{lookup(dataset)}/train/* \t Target Members')
    print(f'\t\t   [2.2.1] ./datasets/{lookup(dataset)}/eval/* \t Target Non-Members')
    print(f'\t\t   [2.2.1] ./datasets/{lookup(dataset)}/test/* \t Shadow Members and Shadow Non-Members')
    train_shadow_loader, test_shadow_loader, train_target_loader, test_target_loader = meminf.get_data(dataset)

    # create shadown model
    model_shadow = CNN(feature_size(dataset)).to(device)
    # train shadow model
    model_shadow = meminf.train_shadow_model(model_shadow, device, train_shadow_loader)

    # get training data for attack model
    print("\t     [2.4] Sample Attack Training Data using Shadown Model")
    train_data, train_label = meminf.get_attack_train_data(model_shadow, train_shadow_loader, test_shadow_loader, device)
    
    # get test data for attack model    
    print("\t     [2.5] Sample Attack Test Data using Target Model")
    test_data, test_label = meminf.get_attack_test_data(target_model, train_target_loader, test_target_loader, device)

    # create attack datasets
    test_data = StandardScaler().fit_transform(test_data)
    train_data = StandardScaler().fit_transform(train_data)

    # dataloader for attack  model
    trainset = datasets.MembershipInferenceAttackDataset(train_data, train_label)
    testset = datasets.MembershipInferenceAttackDataset(test_data, test_label)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=True)    

    # attack model
    print("\t     [2.6] Create Attack Model")
    model_attack = meminf.BCNet(input_shape=train_data.shape[1]).to(device)
    model_attack = meminf.train_attack_model(model_attack, train_loader, device)
    
    # Evaluate attack
    attack_acc = meminf.eval_attack_model(model_attack, test_loader, device)
    print(f"\t     [2.8] Perform Membership Inference Attack on Target Model: {attack_acc:0.4f} Acc.")
    
    return attack_acc  

def model_inversion_attack(target, dataset, device):
    print(f'\t     [3.1] Load Target Model {target}')
    target_model = CNN(feature_size(dataset))#.to(device)
    target_model.load_state_dict(torch.load(target, map_location=device))

    # generate random image 32x32 and empty label tensor 
    rand_imgs, labels = modinv.init(dataset)

    # demonstrate low epoch amount
    print(f'\t     [3.2] Demonstration of low optimization rounds: 200\n')
    print(f'\t\t     Label   Prediction   Probability   Avg. Pixel Difference to Original Image')
    for c, (c_img, c_label) in enumerate(zip(rand_imgs[:5], labels[:5])):
        optimizer = torch.optim.Adam([c_img], lr=0.001)
        adv_image = modinv.MIFace(c_img, c_label, optimizer, target_model, 200)

        logits = target_model(adv_image)
        pred = torch.max(logits, dim=1).indices[0].item()
        prob = torch.softmax(logits, -1).max().item()
        avg_pixel_diff = torch.sum(abs(c_img - adv_image)) / (c_img.shape[2]*c_img.shape[3])

        print(f"\t\t       {c:2.0f}        {pred:2.0f}          {prob:0.4f}          {avg_pixel_diff:0.4f}")

    #results
    results = {'wrong': 0, 'correct': 0}

    # generate one adv. sample for each class
    print(f'\n\t     [3.3] Model Inversion Attack: Generate adversarial example for each class: 2000 rounds\n')
    print(f'\t\t     Label   Prediction   Probability   Avg. Pixel Difference to Original Image')
    for c, (c_img, c_label) in enumerate(zip(rand_imgs, labels)):
        optimizer = torch.optim.Adam([c_img], lr=0.001)
        adv_image = modinv.MIFace(c_img, c_label, optimizer, target_model, 2000)

        logits = target_model(adv_image)
        pred = torch.max(logits, dim=1).indices[0].item()
        prob = torch.softmax(logits, -1).max().item()
        avg_pixel_diff = torch.sum(abs(c_img - adv_image)) / (c_img.shape[2]*c_img.shape[3])

        if c == pred:
            results['correct'] += 1
        else:
            results['wrong'] += 1

        if dataset in ['cifar10', 'mnist', 'att']:
            if c < 5:
                print(f"\t\t       {c:2.0f}        {pred:2.0f}          {prob:0.4f}          {avg_pixel_diff:0.4f}")
            elif c % 5 == 0:
                print(f"\t\t       {c:2.0f}        {pred:2.0f}          {prob:0.4f}          {avg_pixel_diff:0.4f}")                
        else:
            print(f"\t\t       {c:2.0f}        {pred:2.0f}          {prob:0.4f}          {avg_pixel_diff:0.4f}")

        # run this code to save the adversarial images
        # x = plot_img(adv_image)
        # x = adv_image.detach().cpu().numpy()[0]
        # x = numpy.reshape(x, (32, 32, 3))
        # img = Image.fromarray(x, 'RGB')
        # img = img.resize((92, 112))
        # img.save('test.png')
        # exit()

    return float(results['correct'] / (results['correct'] + results['wrong']))


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
                        help="Choose between [UTKFace, CIFAR10, MNIST, ATT]")

    parser.add_argument("--device",
                        default='cpu',
                        help="Provide cuda device")

    parser.add_argument("--inferred-attribute",
                        default='',
                        help="Important for Attribute Inference Attack: Attribute that will be inferred -> UTKFace (race, gender, age), ATT (glasses)")

    args = parser.parse_args()

    # Check for correct dataset
    if args.dataset.lower() != '' and args.dataset.lower() not in ['utkface', 'cifar10', 'mnist', 'att']:
        print(f'\n\t[!] Error ocurred: No such dataset \"{args.dataset}\"\n')
        exit(0)

    # results
    results = { 'attribute': 0, 
                'membership': 0, 
                'modelinv': 0}

    # run attribute inference attack
    if args.dataset not in ['cifar10', 'mnist']:
        if args.inferred_attribute == '':
            print(f'\n\t[!] Error ocurred: --inferred-attribute cannot be empty. Use --help / -h for printing the help page\n')
            exit(0)
        print("\n\t[1] Attribute Inference Attack")
        results['attribute'] = attribute_inference_attack(args.target, args.dataset.lower(), args.inferred_attribute, args.device) 
    else:
        print(f"\n\t[1] Attribute Inference Attack will not be performed, \n\t    since the {lookup(args.dataset)} dataset does not provide any attribute that could be interesting for us to infer")

    # run membership inference attack
    print("\n\t[2] Membership Inference Attack")
    results['membership'] = membership_inference_attack(args.target, args.dataset.lower(), args.device) 

    # run model inversion attack
    print("\n\t[3] Model Inversion Attack")
    results['modelinv'] = model_inversion_attack(args.target, args.dataset.lower(), args.device) 

    # Output
    print(f"\n\n Attack Accuracies: \n\n\t \
                    Attribute-Inference-Attack: \t{results['attribute']:0.2f}\n\t \
                    Membership-Inference-Attack: \t{results['membership']:0.2f}\n\t \
                    Model-Inversion-Attack: \t\t{results['modelinv']:0.2f} (Explanation -> see Report.md)\n\n")