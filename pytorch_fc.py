import torch.nn as nn
import torch.utils
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class neural_network(nn.Module):
    def __init__(self, hid_layers, input_size, output_size):
        super(neural_network, self).__init__()
        self.hid_layers = hid_layers
        self.input_layer = nn.Linear(input_size, hid_layers[0])

        #he initialization better for relu
        nn.init.kaiming_uniform_(self.input_layer.weight, mode='fan_in', nonlinearity='relu')

        #initialize hidden layers
        self.dense_layers = nn.ModuleList()
        for i in range(len(hid_layers) - 1):
            self.dense_layers.append(nn.Linear(hid_layers[i], hid_layers[i + 1]))
            nn.init.kaiming_uniform_(self.dense_layers[i].weight, mode='fan_in', nonlinearity='relu')

        #initialize last hidden layer
        self.dense_layers.append(nn.Linear(hid_layers[len(hid_layers) - 1], output_size))
        nn.init.kaiming_uniform_(self.dense_layers[len(hid_layers) - 1].weight, mode='fan_in', nonlinearity='relu')

        #to plot them later
        self.train_loss = []
        self.train_accur = []

        self.val_loss = []
        self.val_accur = []

    def forward(self, sample):
        out = torch.relu(self.input_layer(sample))
        for i in range(len(self.hid_layers) - 1):
            out = torch.relu(self.dense_layers[i](out))
        out = self.dense_layers[len(self.hid_layers) - 1](out)
        
        return out
    def train_network(self, train_dataloader, train_size, val_dataloader, val_size, epochs, eta):
        self.cr_entr_loss = nn.CrossEntropyLoss()
        self.gradient_descent = torch.optim.SGD(self.parameters(), lr = eta)
        for epoch in range(epochs):
            #for accuracy
            correct_classif_count = 0

            loss = 0
            for batch, (train_data, train_label) in enumerate(train_dataloader):
                train_data = (train_data.view(-1, 3072))

                self.gradient_descent.zero_grad()

                outputs = my_neural_network(train_data)

                my_loss = self.cr_entr_loss(outputs, train_label)   
                loss+=my_loss.item()
                my_loss.backward()

                self.gradient_descent.step()

                _, predictions = torch.max(outputs, dim = 1)
                if (predictions == train_label):
                    correct_classif_count += 1

            self.train_loss.append(loss / train_size)
            print(f"Training Accuracy of epoch {epoch + 1}:\n {correct_classif_count * 100 / train_size} %")
            self.train_accur.append(correct_classif_count * 100 / train_size)
            print(f"Validation accuracy of epoch {epoch + 1}: ")
            self.test_network(val_dataloader, val_size, "val")

    def test_network(self, dataloader, data_size, keyword = ""):

        if (keyword == "test"):
            print(f"Testing accuracy: ")

        with torch.no_grad():
            correct_classif_count = 0
            loss = 0
            for batch, (data, label) in enumerate(dataloader):
                data = (data.view(-1, 3072))
                outputs = my_neural_network(data)
                _, predictions = torch.max(outputs, dim = 1)
                my_loss = self.cr_entr_loss(outputs, label)   
                loss+=my_loss.item()
                if (predictions == label):
                    correct_classif_count += 1

            print(f"{correct_classif_count * 100 / data_size} %")

        if(keyword == "val"):
            self.val_loss.append(loss / data_size)
            self.val_accur.append(correct_classif_count * 100 / data_size)

    def plot_accur_loss(self, epochs):

        epochs = [i+1 for i in range(epochs)]
    
        plt.figure(1)
        plt.plot(epochs, self.train_loss, label = "Training Loss", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_loss, label = "Validation Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(2)
        plt.plot(epochs, self.train_accur, label = "Training Accuracy", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_accur, label = "Validation Accuracy", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()  # Show labels for each curve

        # Showthe plots
        plt.show()
        return 
    
if __name__ == "__main__":

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    
    #to ensure pseudo-randomness   
    torch.manual_seed(1)

    #loading data
    train_set = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform) #array of tuples-> tensor(image) and label(integer)
    test_set = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

    #splitting data to training and validation
    train_size = int(len(train_set) * 0.8) # 80% training data
    val_size = len(train_set) - train_size # 20% validation data

    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_dataloader = DataLoader(dataset = train_set, batch_size = 1, shuffle = False)
    val_data_loader = DataLoader(dataset = val_set, batch_size = 1, shuffle = False)
    test_dataloader = DataLoader(dataset = test_set, batch_size = 1, shuffle = False)
  
    epochs = 50
    
    my_neural_network = neural_network([110, 70, 70], input_size = 3072, output_size = 10)
    my_neural_network.train_network(train_dataloader, train_size, val_data_loader, val_size, epochs, eta = 1e-3)
    my_neural_network.test_network(test_dataloader, len(test_set), "test")
    my_neural_network.plot_accur_loss(epochs)

    


