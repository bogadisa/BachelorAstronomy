import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, average_precision_score
from torchmetrics import AveragePrecision

def classify(root_dir):
    classification = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

    with open(root_dir+"/all.txt", "w") as f:
        for dir_ in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, dir_)):
                i = 0
                for img in os.listdir(os.path.join(root_dir,dir_)):
                    if os.path.isfile(os.path.join(os.path.join(root_dir, dir_), img)):
                        f.write(f"{img} {dir_}\n")
                    #     i += 1
                    # if i > 100:
                    #     break

    return classification

def split_txt(root_dir, function, X, y):
    i = 0
    with open(os.path.join(root_dir, function), "w") as f:
        for X_, y_ in zip(X, y):
            f.write(f"{X_} {y_}\n")
            i += 1
            if i > 100:
                break

def train_epoch(model,  trainloader,  losscriterion, device, optimizer, classifications, selected_modules):

    model.train()
 
    losses = list()
    matrix = torch.zeros(len(classifications), len(classifications))
    for batch_idx, data in enumerate(trainloader):

        # print(data["image"], "\n", data["label"])
        inputs  = data['image'].to(device)
        labels  = data['label'].to(device)
        
        # for nam, mod in model.named_modules():
        #     if nam in selected_modules:
        #         mod.batchindex = batch_idx

        optimizer.zero_grad()

        output = model(inputs)
        cpuout = output.to('cpu')
        loss = losscriterion(output, labels)

        _, preds = torch.max(cpuout, 1)
        labels = labels.float()
  
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx%100==0:
          print('current mean of losses ',np.mean(losses))

        #for small sample sizes, certain classes do not appear in neither preds or labels
        #this handles that exeption
        try:
            temp_matrix = confusion_matrix(labels.data, preds)
            matrix += temp_matrix
        except ValueError:
            temp_data = torch.clone(labels.data)
            temp_preds = torch.clone(preds)
            for i in range(matrix.shape[0]):
                temp_data = torch.tensor(temp_data.tolist()+ [i])
                temp_preds = torch.tensor(temp_preds.tolist() + [i])
            temp_matrix = confusion_matrix(temp_data, temp_preds)
            temp_matrix -= np.identity(matrix.shape[0], dtype=int)

            matrix += temp_matrix

    print(f"Accuracy per class:", "(training)")
    for i, acc in enumerate(matrix.diagonal()/matrix.sum(axis=1)):
        print(f"    {classifications[i]}, {acc}")
    print()

    return np.mean(losses)

def evaluate_acc(model, dataloader, losscriterion, device, classifications, testing = False):

    model.eval()

    losses = []
    curcount = 0
    accuracy = 0
    if testing:
        all_outputs = []
        all_filenames = []
        all_labels = []


    matrix = torch.zeros(len(classifications), len(classifications))
    with torch.no_grad():
        for ctr, data in enumerate(dataloader):
            
            inputs = data['image'].to(device)        
            outputs = model(inputs)
            labels = data['label']
            cpuout = outputs.to('cpu')

            loss = losscriterion(cpuout, labels)
            losses.append(loss.item())

            _, preds = torch.max(cpuout, 1)
            labels = labels.float()

            corrects = torch.sum(preds == labels.data) / float(labels.shape[0] )

            accuracy = accuracy*( curcount/ float(curcount+ labels.shape[0]) ) + corrects.float()* ( labels.shape[0]/ float(curcount+ labels.shape[0]) )
            curcount+= labels.shape[0]

            #for small sample sizes, certain classes do not appear in neither preds or labels
            #this handles that exeption
            try:
                temp_matrix = confusion_matrix(labels.data, preds)
                matrix += temp_matrix
            except ValueError:
                temp_data = torch.clone(labels.data)
                temp_preds = torch.clone(preds)
                for i in range(matrix.shape[0]):
                    temp_data = torch.tensor(temp_data.tolist()+ [i])
                    temp_preds = torch.tensor(temp_preds.tolist() + [i])
                temp_matrix = confusion_matrix(temp_data, temp_preds)
                temp_matrix -= np.identity(matrix.shape[0], dtype=int)

                matrix += temp_matrix

            if testing:
                all_outputs.append(cpuout)
                all_filenames.append(data['filename'])
                all_labels += data["label"].tolist()

    

        print(f"Accuracy per class:", "(validation)" if not(testing) else "(testing)")
        for i, acc in enumerate(matrix.diagonal()/matrix.sum(axis=1)):
            print(f"    {classifications[i]}, {acc}")
        print()

        if testing:
            all_outputs = torch.tensor(torch.cat(all_outputs))
            all_outputs_array = torch.clone(all_outputs).numpy()
            flat_filenames = [filename.split("\\")[-1] for sublist in all_filenames for filename in sublist]
            for i in range(3):
                sorted_outputs = np.sort(all_outputs_array[:, i])
                args_sorted = np.argsort(all_outputs_array[:, i]).tolist()
                sorted_filenames = [flat_filenames[idx] for idx in args_sorted]
                top10 = sorted_filenames[:10]
                bottom10 = sorted_filenames[-10:]
                # sorted_filenames = all_filenames[args_sorted] 
                print("Class", classifications[i], ":")
                print("    Top 10:")
                print("    ", top10)
                print("    Bottom 10:")
                print("    ", bottom10)

            all_labels = torch.tensor(all_labels)
            average_precision = AveragePrecision(task="multiclass", num_classes=len(classifications), average=None)
            average_precision_classes = average_precision(all_outputs, all_labels)
            print("Average precision: ")
            for classification, AP in zip(classifications, average_precision_classes):
                print("    ", classification, ": ", AP.item())

    return accuracy.item() , np.mean(losses)

def hook_relu(nam, feature_map_tracker):
    def fn(m, i, o):
        if len(feature_map_tracker[nam]) < 200:
            feature_map_tracker[nam].append(o)
    return fn

def train_model_nocv_sizes(dataloader_train, dataloader_test ,  model ,  losscriterion, optimizer, scheduler, num_epochs, device, classifications):
    feature_map_tracker = {}
    i = 1
    selected_modules = []
    hook_handles = []
    for nam, mod in model.named_modules():
        if nam == f"layer{i}.1.relu":
            selected_modules.append([nam, mod])
            feature_map_tracker[nam] = []
            handle = mod.register_forward_hook(hook_relu(nam, feature_map_tracker))
            hook_handles.append(handle)
            i += 1
        elif nam == "layer4.0.relu":
            selected_modules.append([nam, mod])
            feature_map_tracker[nam] = []
            handle = mod.register_forward_hook(hook_relu(nam, feature_map_tracker))
            hook_handles.append(handle)

    best_measure = 0
    best_epoch =-1

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

    feature_stats = []
    for nam in feature_map_tracker:
        print(len(feature_map_tracker[nam]))
        num_elements = 0
        non_positive = 0
        for output in feature_map_tracker[nam]:
            non_positive += (output <= 0).sum()
            num_elements += torch.numel(output)
        feature_stats.append(non_positive/num_elements)

    fig = plt.figure()
    plt.plot(train_loss, label="train_loss", linestyle="solid", color="black")
    plt.plot(val_loss, label="val_loss", linestyle="dashed", color="black")
    plt.legend()
        

    print(f"Computed statistic for 5 chosen layers:")
    for i, [stat, nam] in enumerate(zip(feature_stats, feature_map_tracker)):
        print(f"    {nam} = {stat.item()}")
    print()

    for handle in hook_handles:
        handle.remove()

    return best_epoch, best_measure, bestweights, fig