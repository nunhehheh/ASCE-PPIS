import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, matthews_corrcoef, f1_score,recall_score, precision_score,roc_curve,auc
from dataload import *
from test import analysis
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from model import ASCEPPIS

Dataset_Path = "./Dataset/"
Model_Path_hand = "Model/handcrafted.pkl"
Model_Path_all = "Model/all-features.pkl"

def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Testset = pickle.load(f)
    Test_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    run_one_dataset(Testset, Test_psepos_Path)

def run_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    run_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    run_dataframe = pd.DataFrame(run_dic)
    run(run_dataframe, psepos_path)

def run(run_dataframe, psepos_path):
    model_hand=ASCEPPIS(LAYER, 62, HIDDEN_DIM, NUM_CLASSES)
    models = {}
    for i in range(4):
        model_name = f'model_chunks{i}'
        models[model_name] = ASCEPPIS(LAYER, 320, HIDDEN_DIM, NUM_CLASSES)
        models[model_name].to(device)
    model_all=ASCEPPIS(LAYER, 62+1280, HIDDEN_DIM*2, NUM_CLASSES)
    model_llm=ASCEPPIS(LAYER, 1280, HIDDEN_DIM*2, NUM_CLASSES)
    if torch.cuda.is_available():
        model_hand.to(device)
        model_all.to(device)
        model_llm.to(device)

    model_hand.load_state_dict(torch.load(Model_Path_hand, map_location='cuda:0'))
    for i in range(4):
        model_name = f'model_chunks{i}'
        models[model_name].load_state_dict(torch.load(f'Model/chunks{i}.pkl', map_location='cuda:0'))
    model_all.load_state_dict(torch.load(Model_Path_all, map_location='cuda:0'))
    test_loader = DataLoader(dataset=ProDataset(dataframe=run_dataframe, psepos_path=psepos_path),
                             batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)

    valid_pred = []
    valid_true = []
    pred_dict = {}
    for data in test_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, edge, pos, edge_feat = data
            edge = edge.cuda()
            node_features = node_features.float().squeeze()
            node_features, llm_features = torch.split(node_features, [62, 1280], dim=1)
            chunks = torch.chunk(llm_features, chunks=4, dim=1)
            node_features_all = torch.cat([node_features, llm_features], dim=1).cuda()
            node_features = node_features.to(device)
            pos = pos.to(device)
            y_true = labels.to(device)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()
            h_hand = model_hand(node_features, pos, edge)
            h = [torch.tensor([]) for _ in range(4)]
            for i in range(4):
                model_name = f'model_chunks{i}'
                h[i] = models[model_name](chunks[i].cuda(), pos, edge)
            h_chunks = torch.cat(h, dim=1)
            h_all = model_all(node_features_all,pos,edge)
            h = torch.cat((h_hand,h_chunks,h_all), dim=1).cpu().detach().numpy()

            if valid_pred==[]:
                valid_pred = h
            else:
                valid_pred = np.concatenate((valid_pred, h), axis=0)
            valid_true += list(y_true.cpu().numpy())
    X = torch.tensor(valid_pred)
    Y = torch.tensor(valid_true)
    metalearner = joblib.load('./Model/ensemble.model')
    predictions = metalearner.predict(X)
    predictions_proba = metalearner.predict_proba(X)[:, 1]
    accuracy = accuracy_score(Y, predictions)
    auprc = average_precision_score(Y, predictions_proba)
    auroc = roc_auc_score(Y, predictions_proba)
    mcc = matthews_corrcoef(Y, predictions)
    f1 = f1_score(Y, predictions)
    recall = recall_score(Y, predictions)
    precision = precision_score(Y, predictions)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"AUPRC: {auprc:.3f}")
    print(f"AUROC: {auroc:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"F1 Score: {f1:.3f}")

if __name__ == "__main__":
    main()