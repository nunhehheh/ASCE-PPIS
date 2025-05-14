import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from model import *
from dataload import *
from model import *
from torch.utils.data import DataLoader
# Path
Dataset_Path = "./Dataset/"
Model_Path = "Model/handcrafted.pkl"

def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 1
    valid_pred = []
    valid_true = []
    pred_dict = {}
    i = 1
    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, edge, pos, edge_feat = data
            edge = edge.to(device)
            node_features = node_features.to(device).float().squeeze()
            pos = pos.to(device)
            y_true = labels.to(device)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()
            y_pred = model(node_features, pos, edge)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]
            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred, pred_dict

def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1 :
                best_f1 = f1
                best_threshold = threshold
    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true
    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(test_dataframe, psepos_path):
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)
    model = ASCEPPIS(LAYER,INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(Model_Path, map_location='cuda:0'))
    epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)
    result_test = analysis(test_true, test_pred)
    print("========== Evaluate Test set ==========")
    print("Test loss: ", epoch_loss_test_avg)
    print("Test binary acc: ", result_test['binary_acc'])
    print("Test precision:", result_test['precision'])
    print("Test recall: ", result_test['recall'])
    print("Test f1: ", result_test['f1'])
    print("Test AUC: ", result_test['AUC'])
    print("Test AUPRC: ", result_test['AUPRC'])
    print("Test mcc: ", result_test['mcc'])
    print("Threshold: ", result_test['threshold'])


def test_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe, psepos_path)


def main():

    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Testset = pickle.load(f)

    Test_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'

    print("Evaluate ASCEPPIS")
    test_one_dataset(Testset, Test_psepos_Path)

if __name__ == "__main__":
    main()
