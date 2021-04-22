#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.audio_loaders import testloader, trainloader
# ## Start Feature Extraction from the collected Dataset
from src.model import Net
from src.settings import MODEL_LOC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device to train : ", device)


# defining the model
model = Net().to(device)
# defining the optimizer
optimizer = torch.optim.Adam(model.parameters())
# defining the loss function
criterion = nn.CrossEntropyLoss().to(device)
# checking if GPU is available
print(model)


# ## Training the model
def calculate_metrics(outputs, labels):
    total_examples = len(outputs)
    softmax_out = torch.nn.functional.softmax(outputs)
    predicted = torch.argmax(softmax_out, axis=1)

    # Training Accuracy
    correct_pred = torch.sum(predicted == labels).to("cpu").item()
    acc = correct_pred / total_examples

    # Confusion Matrix
    try:
        predicted = predicted.to("cpu").numpy()
        labels = labels.to("cpu").numpy()
        conf_mat = confusion_matrix(labels, predicted)
    except:
        import pdb; pdb.set_trace()

    return {"accuracy": acc, "confusion_matrix": conf_mat}


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

        train_metrics = calculate_metrics(outputs, labels)
        training_acc.append(train_metrics["accuracy"])

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
        val_metrics = calculate_metrics(output_val, labels)
        val_acc.append(val_metrics["accuracy"])

    curr_training_loss = sum(training_acc) / len(training_acc)
    curr_val_loss = sum(val_acc) / len(val_acc)

    print(
        f"After Epoch {i+1} : Training Loss {running_loss} || Validation loss {val_loss}"
    )
    print(
        f"Training Accuracy {curr_training_loss} || Validation Accuracy {curr_val_loss}"
    )
    print("Confusion Matrix is : \n", val_metrics["confusion_matrix"])

    print(f"Saving at epoch {epoch} ")
    torch.save(model.state_dict(), MODEL_LOC)
