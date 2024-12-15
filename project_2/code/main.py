import os
import warnings
import sys
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel
from early_stopper import EarlyStopper
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import numpy as np

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMB_DIM = sys.argv[1]
model_name = sys.argv[2]
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.{}d.txt".format(EMB_DIM))

# 2 - set the correct dimensionality of the embeddings

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
le.fit(['positive','negative'])
temp = y_train[:10]
y_train = le.transform(y_train)  # EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size
print(n_classes)

print("First 10 labels before encoding are:",temp)
print("Corresponding encoded labels are:", y_train[:10])


# Define our PyTorch-based Dataset
temp = X_train[:10]
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

print("Before embedding transform, first 10 samples are: ", temp)
print("After embedding transform, first 10 samples are: ")
for i in range(10):
    print(train_set[i])

# EX7 - Define our PyTorch-based DataLoader
from training import torch_train_val_split
train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
test_loader = DataLoader(test_set,BATCH_SIZE,shuffle=True)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
if(model_name == 'DNN'):
    model = BaselineDNN(output_size=n_classes,  # EX8
                        embeddings=embeddings,
                        trainable_emb=EMB_TRAINABLE)
    attention = False
elif(model_name=='LSTM'): 
    model= LSTM(output_size=n_classes,  # EX8
                embeddings=embeddings,
                trainable_emb=EMB_TRAINABLE,
                bidirectional=True)
    attention = False
elif(model_name == 'SA'):
    attention = True
    model = SimpleSelfAttentionModel(output_size=n_classes,embeddings=embeddings)
elif(model_name == 'MA'):
    attention = True
    model = MultiHeadAttentionModel(output_size=n_classes,embeddings=embeddings,n_head=5)
elif(model_name == 'TA'):
    attention = True
    model = TransformerEncoderModel(output_size=n_classes,embeddings=embeddings,n_layer=4,n_head=5)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if(n_classes!=2):
    criterion = torch.nn.CrossEntropyLoss()  # EX8
else:
    criterion = torch.nn.BCEWithLogitsLoss()

parameters = model.parameters()  # EX8
optimizer = torch.optim.Adam(parameters)

#############################################################################
# Training Pipeline
#############################################################################
saved_train_loss = []
saved_valid_loss = []
saved_test_loss = []
early_stopper = EarlyStopper(model, './saved/best.pth', patience=5) 
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer,attention)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion,attention)

    # evaluate the performance of the model, on both data sets
    valid_loss, (y_valid_gold, y_valid_pred) = eval_dataset(val_loader,
                                                            model,
                                                            criterion,attention)


    saved_valid_loss.append(valid_loss)
    saved_train_loss.append(train_loss)

    saved_y_train_gold_preds = np.concatenate(y_train_gold)
    saved_y_train_preds = np.concatenate(y_train_pred)

    saved_y_valid_gold_preds= np.concatenate(y_valid_gold)
    saved_y_valid_preds= np.concatenate(y_valid_pred)

    if early_stopper.early_stop(valid_loss):
        print('Early Stopping was activated.')
        print(f'Epoch {epoch}/{EPOCHS}, Loss at training set: {train_loss}\n\tLoss at validation set: {valid_loss}')
        print('Training has been completed.\n')
        last_epoch = epoch
        break

model.load_state_dict(torch.load('./saved/best.pth'))
test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion,attention)

print("Accuracy of model on test set is :",accuracy_score(np.concatenate(y_test_gold),np.concatenate(y_test_pred)))
print("F1-Score of model on test set is :",f1_score(np.concatenate(y_test_gold),np.concatenate(y_test_pred),average='macro'))
print("Recall of model on test set is :",recall_score(np.concatenate(y_test_gold),np.concatenate(y_test_pred),average='macro'))

# Plot and label the training and validation loss values
plt.plot(range(last_epoch), saved_train_loss, label='Training Loss')
plt.plot(range(last_epoch), saved_valid_loss, label='Validation Loss')
 
# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

 
# Display the plot
plt.legend(loc='best')
plt.show()

