# Python 3.8.5

import argparse
import os


def attribute_inference_attack(target, dataset, attr):
    return 0


def membership_inference_attack(target, dataset):
    return 0


def model_inversion_attack(target, dataset):
    return 0


if __name__ == '__main__':
    # collect command line arguments
    parser = argparse.ArgumentParser(description='Privacy Enhancing Technologies - Semester Project')

    parser.add_argument("--target",
                        required=True,
                        help="File path to target model that should be tested")

    parser.add_argument("--dataset",
                        required=True,
                        help="File path to dataset that should be used to test the target model")

    parser.add_argument("--inferred-attribute",
                        required=True,
                        help="Important for Attribute Inference Attack: Attribute that will be inferred")

    args = parser.parse_args()

    # results
    results = { 'attribute': 0, 
                'membership': 0, 
                'modelinv': 0}

    # run attribute inference attack
    results['attribute'] = attribute_inference_attack(args.target, args.dataset, args.inferred_attribute) 

    # run membership inference attack
    results['membership'] = membership_inference_attack(args.target, args.dataset) 

    # run model inversion attack
    results['modelinv'] = model_inversion_attack(args.target, args.dataset) 

    # Output
    os.system('clear')
    print(f"\n Attack Results: \n \
                \n\tAttribute-Inference-Attack: \t{results['attribute']} \
                \n\tMembership-Inference-Attack: \t{results['membership']} \
                \n\tModel-Inversion-Attack: \t{results['modelinv']}\n\n")