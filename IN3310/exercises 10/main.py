import csv
import string
# from classes import *
import random
import torch
import torch.nn as nn




def runstuff_finetunealllayers():
    batchsize_tr = 1
    batchsize_test = 1
    maxnumepochs = 5
    numcl = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    lstm = nn.LSTM(input_size=1, hidden_size = 100, num_layers=2, dropout=0.1)

    #we want to track the loss for each epoch
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs - 1 + 1))
        print('-' * 10)
        
        model.train(True)
        losses = train_epoch(model,  dataloader_train,  losscriterion,  device , optimizer, classifications, selected_modules)
        train_loss.append(losses)
        
        if scheduler is not None:
            scheduler.step()
        
        model.train(False)
        measure, meanlosses = evaluate_acc(model, dataloader_test, losscriterion, device, classifications)
        val_loss.append(meanlosses)
        print(' perfmeasure', measure)
        
        if measure > best_measure: #higher is better or lower is better?
            bestweights= model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', best_measure, ' at epoch ', best_epoch+1)

        print()



if __name__ == "__main__":
    category_lines, all_categories = get_data()
    n_categories = len(all_categories)


    runstuff_finetunealllayers()