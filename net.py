import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Para gerar gráfico ao fim do script
import os
import numpy as np
import matplotlib.pyplot as plt

# Biblioteca random foi utilizada apenas para setar seed
import random


# Maximizando a reprodutibilidade dos resultados:
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Definindo o dispositivo onde serão executadas as tarefas:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Para a resolução do problema, utilizei uma Convolutional Neural Network,
# uma vez que esse tipo de rede costuma funcionar bem em problemas semelhantes
# aos apresentados pelo dataset MNIST (classificação de imagens)

class ConvNet (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(1568, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utilizei um Sampler para selecionar apenas as classes interessantes
# à resolução do problema.

class Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)

lr = 1e-3 # record com 1e-3 : 99.95% de acurácia

# Optei por utilizar a função de loss de Cross Entropy,
# uma vez que ela torna mais justa a comparação com o
# loss obtido no notebook do fastai (eles utilizaram Cross Entropy)

cnn = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(cnn.parameters(), lr=lr)

# Utilizei o otimizador ADAM por considerá-lo mais adequado.
# O valor de learning rate foi testado, o melhor valor foi 1e-4.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
batch_size = 4

# Criei uma função lambda para selecionar as imagens que fazem parte das classes
# definidas no enunciado do problema.

get_mask = lambda dset, classes : torch.tensor([1 if i[1] in classes else 0 for i in dset])

classes = (0, 5) # As classes foram dadas no enunciado...

# Aqui é carregado o dataset MNIST:

mnist_train = torchvision.datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
sampler = Sampler(get_mask(mnist_train, classes), mnist_train)
traindata = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, sampler=sampler, shuffle=False)

mnist_test = torchvision.datasets.MNIST(root="./dataset", train=False, transform=transform, download=True)
sampler = Sampler(get_mask(mnist_test, classes), mnist_test)
testdata = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, sampler=sampler, shuffle=False)

epochs = 8 # record com 8 : 99.95% de acurácia

# Foram testadas diversas quantidades de épocas
# De 2 até 12, a partir de 10 a acurácia estagnou em 99.84%

# Aqui a rede é treinada...

iters_loss = {} # O loss é salvo para futuras consultas
real_iter = 0
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(traindata, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        real_iter += 1

        # Salva o loss a cada 250 iterações:
        if real_iter % 250 == 249:
            iters_loss[real_iter + 1] = running_loss
        
        # Optei por mostrar o loss a cada 500 iterações
        if i % 500 == 499:
            iter_loss = running_loss / len(traindata.dataset)
            print(f"Epoch: {epoch + 1}, Iter: {i + 1}, Loss: {iter_loss:.10f}")
            running_loss = 0.0

print("Training finished!")

# O script salva o estado da rede para futuras consultas
torch.save(cnn.state_dict(), os.path.join('output', 'mnist_net.pth'))

# Aqui eu calculo a acurácia da rede:
correct = 0
total = 0
with torch.no_grad(): # Menos consumo de memória...
    for data in testdata:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = correct / total * 100
print(f"Final accuracy: {final_accuracy:.2f}%")

# Salvando o loss ao longo do treino em uma pasta:
plt_path = os.path.join('output', 'graph', f'mnist_{epochs}_{-np.log10(lr):.0f}.png')
plt.plot(iters_loss.keys(), iters_loss.values())
plt.savefig(plt_path)