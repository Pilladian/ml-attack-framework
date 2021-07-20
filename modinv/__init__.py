# Python 3.8.5

import torch

def init(dataset):
    classes = {'utkface': 2,
                'cifar10': 10,
                'mnist': 10,
                'att': 40}

    images = []
    labels = []

    for c in range(classes[dataset]):
        x = torch.randn((1, 3, 32, 32), requires_grad=True)
        y = torch.zeros(1, classes[dataset])
        y[0][c] = 1

        images.append(x)
        labels.append(y)

    return images, labels

def MIFace(x, y, optimizer, model, epoch=10000):
    model.eval()

    for i in range(epoch):
        optimizer.zero_grad()
        prob = torch.softmax(model(x), -1)
        loss = y * prob.log()
        loss = - loss.sum(-1).mean()
        
        loss.backward()
        optimizer.step()
            
    x = torch.tanh(x)
    return x

def inversion_pred(net, data, image, label):
    
    true_class = data.classes[torch.argmax(label).item()]
    true_label = torch.argmax(label).item()
    print("True image label: {} ({})".format( true_class, true_label))

    img_prob = torch.softmax(net(image), -1)
    predict_class = 0 #orig_set.classes[img_prob.argmax().item()]
    predict_label = 0 #img_prob.argmax().item()

    print("Predicted image label: {} ({}) ".format( predict_class, predict_label))
    print("Probability for predicted the class: ", img_prob.max().item())

    #plt.imshow(plot_img(image))