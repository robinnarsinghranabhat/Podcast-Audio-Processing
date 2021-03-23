#!/usr/bin/env python
# coding: utf-8

# ## Start Feature Extraction from the collected Dataset
from src.model import Net
import torch
import torch.nn as nn
import torch.nn.functional as F


from audio_loaders import trainloader, testloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device to train : ', device)


# defining the model
model = Net().to(device)
# defining the optimizer
optimizer = torch.optim.Adam(model.parameters())
# defining the loss function
criterion = nn.CrossEntropyLoss().to(device)
# checking if GPU is available
print(model)


# ## Training the model
def calc_accuracy(outputs, labels):
    total_examples = len(outputs)
    softmax_out = torch.nn.functional.softmax(outputs)
    predicted = torch.argmax(softmax_out, axis=1 )
    correct_pred = torch.sum(predicted == labels).to("cpu").item()
    return correct_pred / total_examples


for epoch in range(100):  # loop over the dataset multiple times

    model.train()
    running_loss = 0.0
    training_acc = []
    val_acc = []
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # labels = labels.unsqueeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        # labels = labels.type_as(outputs)
        # import pdb; pdb.set_trace()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        training_acc.append(calc_accuracy(outputs, labels))

        if i % 50 == 0:  #
            curr_training_loss = sum(training_acc) / len(training_acc)
            print(
                f"At {i+1}th iter, Epoch {epoch+1} :  Loss accumulated upto : {running_loss} || Running Train Accuracy : {curr_training_loss}"
            )
        

    model.eval()
    val_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # labels = labels.unsqueeze(1)
        output_val = model(inputs)

        # labels = labels.type_as(outputs)
        loss_val = criterion(output_val, labels)

        val_loss += loss_val.item()
        val_acc.append(calc_accuracy(output_val, labels))
        

    curr_training_loss = sum(training_acc) / len(training_acc)
    curr_val_loss = sum(val_acc) / len(val_acc)

    print(
        f"After Epoch {i+1} : Training Loss {running_loss} || Validation loss {val_loss}"
    )
    print(
        f"Training Accuracy {curr_training_loss} || Validation Accuracy {curr_val_loss}"
    )

    print(f"SAving at epoch {epoch} ")
    torch.save(model.state_dict(), "new_model")

