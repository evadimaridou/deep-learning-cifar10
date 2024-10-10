import numpy as np
import matplotlib.pyplot as plt
import pickle


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return 1.0 * (x > 0)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def one_hot_encoding(a):
    b = np.zeros((len(a), max(a) + 1))
    b[np.arange(len(a)), a] = 1
    return b

#returns 1*output_size array, all probabilities sum to 1
def softmax(x):
    #preventing large numbers from going to exp
    exp_x = np.exp(x - np.max(x))
    sum_x = np.sum(exp_x)

    return exp_x/sum_x

class layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        #weights are property of the layer before
        self.weights = self.initialize_weights()    

        #biases are property of the layer before
        self.bias = np.zeros(output_size)   

    #he initialization better for relu
    def initialize_weights(self):    #the weights[0][1] refers to the connnection of first element of output and second of input               
        scale = np.sqrt(2.0 / self.input_size)
        weights = np.random.randn(self.output_size, self.input_size) * scale
    
        return weights       
    
class neural_network:
    def __init__(self, hid_layers, input_size, output_size):
        self.hid_layers = []
        #initialize the input layer (connects the input to h1)
        self.input_layer = layer(input_size, hid_layers[0]) 

        self.output_size = output_size
        self.hid_num = len(hid_layers)

        for i in range(self.hid_num-1):
            self.hid_layers.append(layer(hid_layers[i], hid_layers[i + 1]))

        #initialize the last hidden layer
        self.hid_layers.append(layer(hid_layers[self.hid_num - 1], output_size))    
        
        self.train_loss = []
        self.train_accur = []

        self.val_loss = []
        self.val_accur = []
        

    def feed_forward(self, data):  
        post_activation = []
        pre_activation = []

        my_pre_activation = np.dot(self.input_layer.weights, data) + self.input_layer.bias
        pre_activation.append(my_pre_activation)
        post_activation.append(relu(my_pre_activation))
                
        for layer in range(self.hid_num - 1):   
            my_pre_activation = np.dot(self.hid_layers[layer].weights, post_activation[layer]) + self.hid_layers[layer].bias
            pre_activation.append(my_pre_activation)
            post_activation.append(relu(my_pre_activation))

        my_pre_activation = np.dot(self.hid_layers[self.hid_num - 1].weights, post_activation[self.hid_num - 1] ) + self.hid_layers[self.hid_num - 1].bias
        pre_activation.append(my_pre_activation)
        post_activation.append(softmax(my_pre_activation))

        self.pre_activation = pre_activation
        self.post_activation = post_activation

    def compute_error_o(self, target):
        self.error_o = target - self.post_activation[self.hid_num]
       
    def back_propagation(self):        #compute delta
        delta_list = []
        delta = self.error_o   #this is the delta of output layer
        delta_list.append(delta)

        for layer in range(self.hid_num - 1, -1, -1):
            delta = (relu_deriv(self.pre_activation[layer]).reshape(1, -1)) * np.dot(delta.reshape(1, -1), self.hid_layers[layer].weights)
            delta = delta.flatten()
            delta_list.append(delta)

        delta_list.reverse()
        self.delta = delta_list

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

        # Show the plots
        plt.show()
        return 
    
    def gradient_descent(self, train_data, eta):    #update weights and biases
        for layer in range(self.hid_num - 1, -1, -1):
            self.hid_layers[layer].weights += eta * np.dot(self.delta[layer + 1].reshape(-1, 1), self.post_activation[layer].reshape(1, -1))
            self.hid_layers[layer].bias += eta * self.delta[layer + 1]

        self.input_layer.weights += eta * np.dot(self.delta[0].reshape(-1, 1), train_data.reshape(1, -1))
        self.input_layer.bias += eta * self.delta[0]
    
    def train_network(self, train_data, train_labels, val_data, val_labels, epoch_count, eta):
        for epoch in range(epoch_count):

            # Use the permutation to shuffle both data and labels
            permutation = np.random.permutation(train_data.shape[0])
            shuffled_data = train_data[permutation]
            shuffled_labels = train_labels[permutation]

            correct_classif_count = 0
            loss = 0
            for sample in range(shuffled_data.shape[0]):
                self.feed_forward(shuffled_data[sample])
                self.compute_error_o(shuffled_labels[sample])

                #compute cross entropy loss
                loss += np.sum(-shuffled_labels[sample]*np.log(self.post_activation[self.hid_num]))

                self.back_propagation()
                self.gradient_descent(shuffled_data[sample], eta)

                prediction = np.argmax(self.post_activation[self.hid_num]) 

                #compute accuracy
                if prediction == np.argmax(shuffled_labels[sample]):
                    correct_classif_count += 1
            
            self.train_loss.append(loss/train_data.shape[0])    
            print(f"Training accuracy of epoch {epoch + 1}:")
            print(f"{correct_classif_count * 100 / train_data.shape[0]} % \n")

            self.train_accur.append(correct_classif_count * 100 / train_data.shape[0])

            print(f"Validation accuracy of epoch {epoch + 1}:")

            self.test_network(val_data, val_labels, "val")

    def test_network(self, data, labels, keyword):
        
        if(keyword == "test"):
            print(f"Test accuracy :")
            
        correct_classif_count = 0
        loss = 0

        for sample in range(data.shape[0]):
            self.feed_forward(data[sample])  

            #compute corss entropy loss
            loss += np.sum(-labels[sample]*np.log(self.post_activation[self.hid_num]))

            prediction = np.argmax(self.post_activation[self.hid_num]) 

            #compute accuracy
            if prediction == np.argmax(labels[sample]):
                correct_classif_count += 1
        
        print(f"{correct_classif_count * 100 / data.shape[0]} % \n")
        
        if(keyword == "val"):
            self.val_loss.append(loss/data.shape[0])
            self.val_accur.append(correct_classif_count * 100 / data.shape[0])

if __name__ == "__main__":
    
    #to ensure pseudo-randomness        
    np.random.seed(0)   

    #loading data
    batch_1 = unpickle('data_batch_1')
    batch_2 = unpickle('data_batch_2')
    batch_3 = unpickle('data_batch_3')
    batch_4 = unpickle('data_batch_4')

    val_batch = unpickle('data_batch_5')
    test_batch = unpickle('test_batch')

    d_batch_1 = batch_1["data"]
    d_batch_2 = batch_2["data"]
    d_batch_3 = batch_3["data"]
    d_batch_4 = batch_4["data"]
    val_data = val_batch["data"]
    test_data = test_batch["data"]   #arrays 1000x3072

    l_batch_1 = batch_1["labels"]
    l_batch_2 = batch_2["labels"]
    l_batch_3 = batch_3["labels"]
    l_batch_4 = batch_4["labels"]
    val_labels = val_batch["labels"]

    test_labels = test_batch["labels"]

    train_data = np.concatenate((d_batch_1, d_batch_2, d_batch_3, d_batch_4), axis=0)
    train_labels = np.concatenate((l_batch_1, l_batch_2, l_batch_3, l_batch_4), axis=0)

    #for numerical stability
    train_data = train_data / 255
    val_data = val_data / 255
    test_data = test_data / 255

    #sample size
    input_size = train_data.shape[1]

    train_labels = one_hot_encoding(train_labels)
    val_labels = one_hot_encoding(val_labels)
    test_labels = one_hot_encoding(test_labels)

    epoch_count = 50

    my_neural_network = neural_network([110, 70, 70], input_size = input_size, output_size=10)
    my_neural_network.train_network(train_data, train_labels, val_data, val_labels, epoch_count = epoch_count, eta = 1e-3)
    my_neural_network.test_network(test_data, test_labels, "test")
    my_neural_network.plot_accur_loss(epoch_count)
