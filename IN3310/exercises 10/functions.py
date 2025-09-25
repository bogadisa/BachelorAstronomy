import torch
import torch.nn as nn

all_letters = string.ascii_letters + "0123456789 .,:!?’[]()/+-="

def get_data():
    category_lines = {}
    all_categories = ["st"]
    category_lines["st"] = []

    filterwords=["NEXTEPISODE"]
    with open("./star_trek_transcripts_all_episodes.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
                    # print(el)
                    v=el.strip().replace(";","").replace("\"","")
                    category_lines["st"].append(v)
    n_categories = len(all_categories)
    print(len(all_categories), len(category_lines["st"]))
    print("done")
    return category_lines, all_categories


def train_model(model, train_iter, epoch, losscriterion):
    model.train()
    for idx, batch in enumerate(train_iter):
    
        prediction = model(text)
        loss = losscriterion(prediction, target)
        print("Training Loss", loss.item())
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        print("Training Accuracy", acc.item())