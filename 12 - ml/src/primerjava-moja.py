from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import plotting

import data_higgs as dh

# i know weird file names
from src.network import Network

from tqdm import tqdm


net = Network()

#-------- routines


def split_xy(rawdata):
    # split
    data_y = rawdata['hlabel']  # labels only: 0.=bkg, 1.=sig
    data_x = rawdata.drop(['hlabel'], axis=1)  # features only

    mu = data_x.mean()
    s = data_x.std()
    dmax = data_x.max()
    dmin = data_x.min()
    data_x = (data_x - dmin)/(dmax-dmin)

    return data_x, data_y


hdata = dh.load_data()
data_fnames = hdata['feature_names'].to_numpy()[1:]  # labels not needed
n_dims = data_fnames.shape[0]
print("Entries read {} with feature names {}".format(n_dims, data_fnames))

# training sample, should split a fraction for testing
x_trn, y_trn = split_xy(hdata['train'])
x_train, x_test, y_train, y_test = train_test_split(
    x_trn, y_trn, test_size=0.1)  # 10% split
x_val, y_val = split_xy(hdata['valid'])  # independent cross-valid sample

print("Shapes train:{} and test:{}".format(x_train.shape, x_test.shape))

net = Network()
net.addInputLayer(x_train.shape[1])
net.addLayer(50, 'ReLu')
net.addLayer(50, 'ReLu')
net.addLayer(1, 'Sigmoid')

nepoch = 4
batch_size = 100
learning_rate = 0.01

accuracy_train = []
accuracy_test = []

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
x_test = x_test.to_numpy()

for epoch in range(nepoch):
    acc_train = net.accuracy(x_train[0:4000], y_train[0:4000])
    acc_test = net.accuracy(x_test, y_test)

    for i in tqdm(range(0, len(x_train), batch_size)):

        net.train_batch(x_train[i:i+batch_size],
                        y_train[i:i+batch_size], alpha=learning_rate)

    print(f"Epoch {epoch}, has accuracy of", acc_train)

    accuracy_train.append(acc_train)
    accuracy_test.append(acc_test)


plt.figure(figsize=(8, 8))

t = plt.plot(accuracy_train, '--', label='train')
plt.plot(accuracy_test, label='test', color=t[0].get_color())

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("../grafi/accuracy_moja.pdf", bbox_inches='tight')

y_pred = net.predict(x_test)

fake, true, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fake, true)
plt.figure(figsize=[6, 6])
lw = 2
plt.plot(fake, true, color='darkorange', lw=lw,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC moja')
plt.legend(loc="lower right")
plt.savefig('roc-curve.png')
plt.show()

auc = roc_auc_score(y_test, y_pred)
print("AUC score: {}".format(auc))


plt.show()
