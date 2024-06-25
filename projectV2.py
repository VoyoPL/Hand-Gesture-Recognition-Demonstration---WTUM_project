import gradio as gr
import os
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knn
import sklearn.metrics as metrics
import torch
from torch import Tensor, nn
import torchvision
import torchvision.transforms as transforms
#import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
import pandas as pd

#Setup section

#default data path:
path = "./data/archive/leapGestRecog"

global m_name

m_name = 'knn'

global model

global train_size

global device

global train

global test

global dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_size = 0.8

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

#możliwe, że przy chmurze lepiej ładować dane przez api kaggle
def load_classes(path:str):
    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir(path):
        if not j.startswith('.'): #bez ukrytych plików
            lookup[j] = count
            reverselookup[count] = j
            count = count+1

    return lookup, reverselookup, count

def load_images(path:str, lookup:dict):
    x_data = list()
    y_data = list()
    datacount = 0
    for i in range(0, 10):
        for j in os.listdir(path+"/0"+str(i)+"/"):
            if not j.startswith('.'):
                count = 0
                for k in os.listdir(path+"/0"+str(i)+"/"+str(j)+"/"):
                    img = ski.io.imread(fname=path+"/0"+str(i)+"/"+str(j)+"/"+k, as_gray=True)
                    start_col = (640 - 240) // 2
                    end_col = start_col + 240
                    img = img[:, start_col:end_col]
                    img = ski.transform.rescale(image=img, scale=0.33, anti_aliasing=False )
                    
                    x_data.append(img)
                    count = count+1
                y_values = np.full((count, 1), lookup[j])
                y_data.append(y_values)
                datacount = datacount + count
                    
    x_data = np.array(x_data, dtype= 'float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1)
    return x_data, y_data



#train size zmienny w gui
def split_dataset(dataset, train_size):
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, device, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.y_data[idx]
        img = self.x_data[idx]
        if self.transform:
            img = self.transform(img)
            t = torch.tensor(label)
            label = t
            label = label.to(device)
            img.to(device)
        return (img, label)

def train_cnn(model, train_loader, test_loader, device, num_epochs, lr):

  if(device == 'cuda'):
      force_cudnn_initialization()
    
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
      model.train()  
      for batch_idx, (data, labels) in enumerate(train_loader):
          data, labels = data.to(device), labels.to(device)
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, labels)
          loss.backward()
          optimizer.step()
      print(f'loss: {loss}')
          
  print("training complete")
  model = model.to(device)
  return model







#End Setup section

#Models section

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,2), stride=1),
            nn.Conv2d(64, 32, kernel_size=(3,3)),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(161312, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=-1)
        )
        


    def forward(self, imgs):
        output = self.conv_net(imgs)

        return output 


def knnModel(train, test, n_neighbors):
    model = knn(n_neighbors = n_neighbors)

    x_train = train.dataset.x_data
    train_labels = train.dataset.y_data

    model.fit(x_train, train_labels)

    return model


def cnnModel(train, test, batch_size, num_epochs):
    #train, test = split_dataset(dataset, split)

    train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = batch_size,
                                           shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test,
                                           batch_size = batch_size,
                                           shuffle = False)
    model = Cnn()
    model.to(device)
    model = train_cnn(model, train_loader, test_loader, device, num_epochs, 0.001)
    return model

def train_model(split, b_size, n_neigh, num_epochs):
    global model
    global dataset
    global train
    global test
    if(m_name == 'cnn'):
        dataset = load_specific_dataset(m_name)
        train, test = split_dataset(dataset, split)
        model = cnnModel(train, test, b_size, num_epochs)

    elif(m_name == 'knn'):
        dataset = load_specific_dataset(m_name)
        train, test = split_dataset(dataset, split)
        model = knnModel(train, test, n_neigh) 
    print(model)
    return "trenowanie zakończone"
#End Models Section


#Eval section

#tą funkcję można przerobić w jakiś sposób, żeby nie ładować za każdym razem danych (zajmuje to zbyt
#wiele czasu, na wolniejszym sprzęcie będzie wooolno)
def load_specific_dataset(chosen_model):
    lookup, reverselookup, count = load_classes(path+"/00/")

    x_data, y_data = load_images(path, lookup)

    y_data = y_data.flatten()   
    
    dataset = CustomDataset(x_data, y_data, device, transform=transforms.Compose([transforms.ToPILImage(), 
                                                                       transforms.Resize((79, 79)),
                                                                        transforms.ToTensor()
                                                                      ]))
        
    if(chosen_model == 'cnn'):
       dataset.x_data = torch.tensor(dataset.x_data)
       dataset.y_data = torch.tensor(dataset.y_data)
       dataset.x_data.to(device)
       dataset.y_data.to(device)
    else:
        x = dataset.x_data
        print(x.dtype)
        print(x.shape)
        x = x.reshape((x.shape[0], x.shape[1]**2))
        dataset.x_data = x

    return dataset

#accuracy, precision, recall, f1, conf_matrix i to jakoś potem skleić w gui, żeby było w miarę ładnie
def model_eval():
    global model
    y_test = test.dataset.y_data
    x_test = test.dataset.x_data
    if(m_name == 'cnn'): #cnn
        #tutaj kreatywnie ewaluacja sieci neuronowej        
        x_test = torch.tensor(x_test, device=device)
        with torch.no_grad(): 
            preds = list()
            for x in x_test:
                sm = nn.Softmax(dim=1)
                x.unsqueeze_(0)
                x.unsqueeze_(0)
                pred = sm(model(x))
                pred = pred.cpu().detach().numpy()[0]
                pred = pred.argmax(0).item()
                preds.append(pred)

    #a tutaj kreatywnie ewaluacja knn
    else:
        preds = model.predict(x_test)        

    acc = metrics.accuracy_score(y_test, preds)
    prc = metrics.precision_score(y_test, preds, average='micro')
    rec = metrics.recall_score(y_test, preds, average='micro')
    f1 = metrics.f1_score(y_test, preds, average='micro')
    conf = metrics.confusion_matrix(y_test, preds)
    combined = f"accuracy: {acc}\nprecision: {prc}\nrecall: {rec}\nf1 score: {f1}\n conf\n{conf}"
    return combined       
    
def model_predict(img):
    #print(img)
    img = ski.color.rgb2gray(img)
    start_col = (640 - 240) // 2
    end_col = start_col + 240
    img = img[:, start_col:end_col]
    img = ski.transform.rescale(image=img, scale=0.33, anti_aliasing=False )
    global model
    print(model)
    if(m_name == 'cnn'): #cnn
        print("predicting with cnn model")
        with torch.no_grad():
            img = np.float32(img)
            img = torch.tensor(img, device=device)
            img.unsqueeze_(0)
            img.unsqueeze_(0)
            print(device)
            print(img.shape)
            sm = nn.Softmax(dim=1)
            pred = sm(model(img))
            pred = pred.cpu().detach().numpy()[0] 
            labels = [str(i) for i in range(len(pred))] #ogarnać labele normalne
            print(pred)
            print(labels)
            df = pd.DataFrame( {"labels": labels, "probabilities": pred.tolist()} )
            print(df)
            return df

    else:
        print("predicting with knn model")
        img = img.reshape((1, -1))
        pred = model.predict_proba(img)[0]
        print(pred)
        labels = [str(i) for i in range(len(pred))] #jak wyżej
        print(labels)
        df = pd.DataFrame( {"labels": labels, "probabilities": pred.tolist()} )
        print(df)
        return df
#End Eval section

#GUI section

batch_size = 64
n_neighbors = 3


def choose_model(check, t_size):
    global train_size
    global m_name
    train_size = t_size
    print(check)
    print(t_size)
    if(check):
        m_name = 'cnn'
        
    else:
        m_name = 'knn'
    print(m_name)

with gr.Blocks() as gui:
    #trening
    with gr.Tab("Trening modelu"):
        train_button = gr.Button("trenuj")
        chkbox = gr.Checkbox(label="model cnn - true, knn - false")
        sldr = gr.Slider(label="rozmiar zbioru treningowego", minimum=0.1, maximum=0.9, value=0.8)  
        b_size = gr.Number(label='rozmiar wsadu', value=64)
        n_neigh = gr.Number(label='liczba sąsiadów', value=3) 
        num_epochs = gr.Number(label='liczba epok', value=10) 
        chkbox.change(fn=choose_model, inputs=[chkbox, sldr])
        train_button.click(train_model, inputs=[sldr, b_size, n_neigh, num_epochs], outputs=gr.Textbox())
    #predykcja
    with gr.Tab("predykcja"):
        image_input = gr.Image(type='numpy')
        p_button = gr.Button("predykcja")
        label = gr.BarPlot(x='labels', y='probabilities')
        p_button.click(fn=model_predict, inputs=[image_input], outputs=label)
    #ewaluacja
    with gr.Tab("ewaluacja"):
        eval_button = gr.Button("ewaluacja")
        eval_button.click(model_eval, outputs=gr.Textbox())



if __name__ == "__main__":
    #print(model)
    #print(model.)
    gui.launch()
