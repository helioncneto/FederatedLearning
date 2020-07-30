import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K



def load_mnist_bypath(paths, verbose=-1):
    '''Espera ler imagens em que cada classe esta em um diretorio separado,
    Por exemplo: imagens do digito 0 esta na pasta 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # aqui é feita a escala da img para [0, 1] para diminuir o impacto do brilho de cada pixel
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels

# Declara o caminho do conjunto de dados mnist
img_path = '/home/helio/PycharmProjects/FederatedLearning/mnist/trainingSet/trainingSet'

# Gera a lista de caminhos, utilizando a funcao list_images da biblioteca paths
image_paths = list(paths.list_images(img_path))

# carrega as imagens em arrays
image_list, label_list = load_mnist_bypath(image_paths, verbose=10000)

# Realiza o one-hot encoded para podermos utilizar a funcao de perda sparse-categorical-entropy
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)


#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.3, random_state=42)


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return:Um dicionario com o id dos cliente como chave do dicionario e o valor
                sera o fragmento de dados - tupla de imagens e rotulos.
        args:
            image_list: um objeto numpy array com as imagens
            label_list: lista de rotulos binarizados (one-hot encoded)
            num_client: numero de clientes (clients)
            initials:o prefixo dos clientes, e.g, clients_1

    '''

    # cria a lista de nomes de clientes
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # embaralha os dados
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # fragmenta os dados e divide para cada cliente
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # Verifica se o numero de fragmento é igual ao de clientes
    assert(len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}

# Cria os clientes
clients = create_clients(X_train, y_train, num_clients=100, initial='client')


def batch_data(data_shard, b=32):
    '''Recebe um fragmento de dados de um clientes e cria um objeto tensorflow data nele
    args:
        data_shard: dados e rotulos que constitui o fragmento de dados de um cliente
        b: tamanho do batch
    return:
        objeto tensorflow data'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(b)


# Processa e agrupa os dados de treinamento para cada cliente
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

# processar e agrupar conjunto de teste
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

class MLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

lr = 0.01
comms_round = 100
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, decay=lr / comms_round, momentum=0.9)


def weight_scalling_factor(clients_trn_data, client_name, participants):
    '''Calcula a proporção do tamanho dos dados de treinamento local de um cliente
    com todos os dados gerais de treinamento mantidos por todos os clientes'''
    #client_names = list(clients_trn_data.keys())
    # calcula o tamanho do batch
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # primeiro calcula o total de  dados de treinamento entre clientes
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()
                        for client_name in participants]) * bs
    # obter o número total de pontos de dados mantidos por um cliente
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    '''Escala os pesos do modelo'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Retorne a soma dos pesos dimensionados listados. O é equivalente ao peso médio dos pesos'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('Agregation Round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


def check_local_loss(client, model):
    # Verificar perda local
    cce_l = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    client_x = np.array([i[0] for i in clients[client]])
    client_y = np.array([i[1] for i in clients[client]])
    logits_l = model.predict(client_x)
    loss_l = cce_l(client_y, logits_l)
    acc_l = accuracy_score(tf.argmax(logits_l, axis=1), tf.argmax(client_y, axis=1))
    print('Local accuracy: {}. Local loss: {}'.format(acc_l, loss_l))
    return acc_l, loss_l


### Inicia o modelo global ###

smlp_global = MLP()
global_model = smlp_global.build(784, 10)

# Comecao loop de treino global
for comm_round in range(comms_round):

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    # randomize client data - using keys
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)
    client_select = client_names[0:55]
    #print(client_select)

    # percorre cada cliente e criar um novo modelo local
    for client in client_select:
        smlp_local = MLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # definir o peso do modelo local para o peso do modelo global
        local_model.set_weights(global_weights)

        # fit local model with client's data
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client, client_select)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        # Verifica a acuracia local
        #acc_l, loss_l = check_local_loss(client, local_model)

        # clear session to free memory after each communication round
        K.clear_session()

    # to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # update global model
    global_model.set_weights(average_weights)

    # test global model and print out metrics after each communications round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
