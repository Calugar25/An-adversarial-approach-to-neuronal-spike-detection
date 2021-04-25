
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch 
import torchgan
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['figure.figsize'] = [16, 8]

class _DebugLayer(nn.Module):
    def __init__(self, log):
        super(_DebugLayer, self).__init__()
        self.log = log
    
    def forward(self, x):
        print(self.log)
        print(x.shape)
        return x

class _FFTLayer(nn.Module):
    def __init__(self):
        super(_FFTLayer,self).__init__()

    def forward(self, x):
        fft = torch.fft.rfft(x, dim=1)
        return torch.stack([torch.cat((x.real,x.imag),0) for x in fft])

class _Conv1dAdapter(nn.Module):
    def __init__(self, outputSize, isIn):
        super(_Conv1dAdapter, self).__init__()
        self.outputSize = outputSize
        self.isIn = isIn
    
    def forward(self, x):
        if self.isIn:
            out = x.view(-1, 1, self.outputSize)
        else:
            out = x.view(-1, self.outputSize)
        return out
    
class _netG(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(_netG, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        self.mainModule = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            nn.Softplus(),
            nn.Linear(self.hiddenSize, self.hiddenSize, bias=True),
            nn.Softplus(),
            nn.Linear(self.hiddenSize, self.outputSize, bias=True),
            # self._block(self.inputSize, 20),
            # self._block(20, 40),
            # self._block(40, 60),
            # self._block(60, self.outputSize),
        )
        
    def forward(self, x):
        return self.mainModule(x)

    def _block(self, inSize, outSize):
        return nn.Sequential(
            nn.Linear(inSize, outSize, bias=True),
            nn.Softplus(),
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=9,padding=4),
            nn.Softplus(),
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=9,padding=4),
            nn.Softplus(),
        )


class _netC(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(_netC, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        self.mainModule = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            nn.Tanh(),
            nn.Linear(self.hiddenSize, 1, bias=True),
        )
        
    def forward(self, x):
        return self.mainModule(x)

def Generator(inputSize, hiddenSize, outputSize):
    return _netG(inputSize, hiddenSize, outputSize).to(device)


def Critic(inputSize, hiddenSize):
    return _netC(inputSize, hiddenSize).to(device)

def _gradient_penalty(critic, real, fake):
    batchSize, L = real.shape
    epsilon = torch.rand((batchSize, 1)).repeat(1,L).to(device)

    interpolation = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic.forward(interpolation)

    gradient = torch.autograd.grad(
        inputs=interpolation,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)

    gradient_norm = gradient.norm(2, dim=1)

    gradient_penalty = torch.mean((gradient_norm -1) ** 2)

    return gradient_penalty


def adversarial_trainer( 
    train_loader, 
    generator, 
    critic, 
    device = device,
    epochs=5,
    learningRate=1e-4,
    Citers=10,
    Giters=1,
    noiseDim = 100,
    batchSize = 64,
    gpLambda = 10, 
    printEpochs=10
):
    optimizerC = optim.Adam(critic.parameters(), lr=learningRate, betas=(0.0, 0.9))
    optimizerG = optim.Adam(generator.parameters(), lr=learningRate, betas=(0.0, 0.9))
    #optimizerC = optim.RMSprop(critic.parameters(), lr=learningRate)
    #optimizerG = optim.RMSprop(generator.parameters(), lr=learningRate)
    
    criticLoss = []
    
    for epoch in range(epochs):
        for [real] in train_loader:
            real = real.squeeze().to(device)

            for _ in range(Citers):
                noise = torch.rand(real.shape[0], 1, noiseDim).to(device)
                fake = generator.forward(noise).reshape(-1,critic.inputSize)

                critic_real = critic.forward(real).reshape(-1)
                critic_fake = critic.forward(fake).reshape(-1)

                gp = _gradient_penalty(critic, real, fake)

                critic_loss = (
                    torch.mean(critic_fake)     # Tries to minimize critic_fake
                    -torch.mean(critic_real)    # Tries to maximize critic_real
                    + gpLambda * gp             # Tries to minimize gradient penalty
                )

                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                optimizerC.step()

                # for p in critic.parameters():
                #     p.data.clamp_(-0.01,0.01)

            for _ in range(Giters):
                noise = torch.rand(batchSize, 1, noiseDim).to(device)
                fake = generator.forward(noise).reshape(-1,critic.inputSize)
                critic_fake = critic.forward(fake).reshape(-1)
                
                generator_loss = -torch.mean(critic_fake) # Tries to maximize critic_fake

                generator.zero_grad()
                generator_loss.backward()
                optimizerG.step()
            
            cLoss = critic_loss.cpu().detach().numpy()
            criticLoss.append(cLoss)

        if (epoch + 1) % printEpochs == 0:
            plt.plot(criticLoss)
            plt.plot(torch.zeros(len(criticLoss)).numpy())
            plt.title("Critic loss")
            plt.show()

            print("Critic loss {}".format(critic_loss))
            
            print("\nEpoch {}".format(epoch))
            
            print("\nGenerated example:")
            
            for i in [7,29,41,61]:
                plt.plot(torch.fft.irfft(fake[i][0:40] + 1j * fake[i][40:]).cpu().detach().numpy())
                plt.show()





class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(3, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    #print(y_pred_tag)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    #print(acc)
    acc = torch.round(acc * 100)
    
    return acc

def classifiyTrain(
    train_loader,
    labels,
    epoch,
    model,
    device
):


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for e in range(1,epoch+1):
        epoch_loss=0
        epoch_acc=0
        i=0
        for X_batch,y_batch in train_loader:
            X_batch=X_batch.to(device)
            y_batch=y_batch.to(device)

            optimizer.zero_grad()

            y_pred=model(X_batch)
            

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            i+=1

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')



class trainData():
    def __init__(self, X_data, y_data):
        super(trainData, self).__init__()
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)



class testData():
    
    def __init__(self, X_data):
        super(testData, self).__init__()
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    



def getFeatures(extractedSpikeClean,extractedNoiseClean):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import torch.utils.data as data_utils

    extractedSpikesValidation = np.array(extractedSpikeClean)
    energySpike = []

    labels=[1,1,1,1,1,0,0,0,0]*20

    extractedNoisesValidation=np.array(extractedNoiseClean)
    energyNoise=[]

    featureData=[]
    featureVec=[]
    for index in range(len(extractedNoisesValidation)):
        maxS=max(extractedSpikesValidation[index])
        maxN=max(extractedNoisesValidation[index])
        featureVec.append(maxS/maxN)
        
        minS=min(extractedSpikesValidation[index])
        minN=min(extractedNoisesValidation[index])
    
        featureVec.append(minS/minN)
    
        energySpike=sum(np.power(extractedSpikesValidation[index], 2))
        energyNoise=sum(np.power(extractedNoisesValidation[index], 2))
    
        featureVec.append(energySpike/energyNoise)
        featureData.append(featureVec[index:index+3])
    
    featureData=np.array(featureData)

    featureData=trainData(torch.FloatTensor(featureData),torch.FloatTensor(labels))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(featureData)

    featureDataLoader=data_utils.DataLoader(featureData,batch_size=18,shuffle=False)

    for data in featureDataLoader:
    
        print(data)



    print(labels)
    return featureDataLoader


def getFeaturesTest(extractedSpikeClean,extractedNoiseClean):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import torch.utils.data as data_utils

    extractedSpikesValidation = np.array(extractedSpikeClean)
    energySpike = []

    labels=[1,1,1,1,1,0,0,0,0]*20

    extractedNoisesValidation=np.array(extractedNoiseClean)
    energyNoise=[]

    featureData=[]
    featureVec=[]
    for index in range(len(extractedNoisesValidation)):
        maxS=max(extractedSpikesValidation[index])
        maxN=max(extractedNoisesValidation[index])
        featureVec.append(maxS/maxN)
        
        minS=min(extractedSpikesValidation[index])
        minN=min(extractedNoisesValidation[index])
    
        featureVec.append(minS/minN)
    
        energySpike=sum(np.power(extractedSpikesValidation[index], 2))
        energyNoise=sum(np.power(extractedNoisesValidation[index], 2))
    
        featureVec.append(energySpike/energyNoise)
        featureData.append(featureVec[index:index+3])
    
    featureData=np.array(featureData)

    featureData=testData(torch.FloatTensor(featureData))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(featureData)

    featureDataLoader=data_utils.DataLoader(featureData,batch_size=1)

    for data in featureDataLoader:
    
        print(data[0])



    print(labels)
    return featureDataLoader


if __name__ =="__main__":
    print("No main module functionality.")