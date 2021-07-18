# Python 3.8.5

from Target import CNN
import torchvision.transforms as transforms
from datasets import UTKFace, CIFAR10, MNIST, ATT
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse
import os


def eval_model(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=args.device)
            y = y.to(device=args.device)

            logits = model(x)
            _, preds = torch.max(logits, dim=1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

    return (float(num_correct) / float(num_samples))


def main(args):
    # transform for images
    transform = transforms.Compose([transforms.Resize(size=(32,32)),
                              transforms.ToTensor(), 
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # datasets
    if args.dataset.lower() not in ['cifar10', 'utkface', 'mnist', 'att']:
        print(f'\n\n\t[!] Error ocurred: No such dataset \"{args.dataset}\"\n')
        exit(0)
    
    if args.dataset.lower() == 'cifar10':
        model = CNN(10).to(args.device)
        dataset_path = './datasets/CIFAR10/'
        train_dataset = CIFAR10(dataset_path, train=True, transform=transform)
        eval_dataset = CIFAR10(dataset_path, eval=True, transform=transform)
        test_dataset = CIFAR10(dataset_path, test=True, transform=transform)
   
    elif args.dataset.lower() == 'utkface':
        model = CNN(2).to(args.device)
        dataset_path = './datasets/UTKFace/'
        train_dataset = UTKFace(dataset_path, train=True, transform=transform)
        eval_dataset = UTKFace(dataset_path, eval=True, transform=transform)
        test_dataset = UTKFace(dataset_path, test=True, transform=transform)
    
    elif args.dataset.lower() == 'mnist':
        model = CNN(10).to(args.device)
        dataset_path = './datasets/MNIST/'
        train_dataset = MNIST(dataset_path, train=True, transform=transform)
        eval_dataset = MNIST(dataset_path, eval=True, transform=transform)
        test_dataset = MNIST(dataset_path, test=True, transform=transform)
    
    elif args.dataset.lower() == 'att':
        model = CNN(40).to(args.device)
        dataset_path = './datasets/ATT/'
        train_dataset = ATT(dataset_path, train=True, transform=transform)
        eval_dataset = ATT(dataset_path, eval=True, transform=transform)
        test_dataset = ATT(dataset_path, test=True, transform=transform)
    
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)
    eval_loader = DataLoader(dataset=eval_dataset, shuffle=True, batch_size=32)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)

    print(f' Training Set: {train_dataset.__len__()}')
    print(f' Eval Set: {eval_dataset.__len__()}')
    print(f' Test Set: {test_dataset.__len__()}\n')

    # hyperparameter
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = int(args.epochs)

    # train target
    model.train()
    i = 0
    for epoch in range(num_epochs):
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.type(torch.int64).to(args.device)
            i += 1
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print (f' Epoch [{epoch+1:2}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {eval_model(model, eval_loader):.4f}')

    print(' Training completed...!\n')
    torch.save(model.state_dict(), f'./Target/target-model-{args.dataset}.pth')
    print(f' Model stored at ./Target/target-model-{args.dataset}.pth')

    # test model
    print(f' Accuracy on Test Set: {eval_model(model, test_loader)}')


if __name__ == '__main__':
    os.system('clear')
    print("\n Privacy Enhancing Technologies - Semester Project - Train Target Model\n")
    
    # collect command line arguments
    parser = argparse.ArgumentParser(description='Privacy Enhancing Technologies - Semester Project')

    parser.add_argument("--dataset",
                        required=True,
                        help="[CIFAR10, UTKFace, MNIST, ATT]")
    
    parser.add_argument("--epochs",
                        default=50,
                        help="Number of training epochs")

    parser.add_argument("--device",
                        default='cpu',
                        help="Provide cuda")

    args = parser.parse_args()

    main(args)