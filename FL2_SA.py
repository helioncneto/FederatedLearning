from SimulatedAnnealing import SimulatedAnnealing
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

# LR: 0.010000 to 0.00001
# local update: 15 to 1
sa = SimulatedAnnealing(initial_temperature=100, cooling=0.8, number_variables=55,
                        upper_bounds=[99, 99, 99, 99, 99, 99, 99, 99, 99, 99
                                      , 99, 99, 99, 99, 99, 99, 99, 99, 99, 99
                                      , 99, 99, 99, 99, 99, 99, 99, 99, 99, 99
                                      , 99, 99, 99, 99, 99, 99, 99, 99, 99, 99
                                      , 99, 99, 99, 99, 99, 99, 99, 99, 99, 99
                                      , 99, 99, 99, 99, 99],
                        lower_bounds=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                      , 0, 0, 0, 0, 0],
                        computing_time=10000, no_attempts=2, lr=False)

'''
sa = SimulatedAnnealing(initial_temperature=100, cooling=0.8, number_variables=7,#57,
                        upper_bounds=[0.010000, 15, 9, 9, 9, 9, 9],
                        lower_bounds=[0.00001, 1, 0, 0, 0, 0, 0], computing_time=10000, no_attempts=2)
'''

NUM_USERS = 100
# Declara o caminho do conjunto de dados mnist
IMG_PATH = '/home/helio/PycharmProjects/FederatedLearning/mnist/trainingSet/trainingSet'
LR = 0.01
GLOBAL_ROUNDS = 100
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']


def load_mnist_bypath(paths, verbose=-1):
    '''Le imagens em que cada classe esta em um diretorio separado,
    Por exemplo: imagens do digito 0 esta na pasta 0 '''
    data = list()
    labels = list()
    # faz um loop por todas as imagens em todos os caminhos
    for (i, imgpath) in enumerate(paths):
        # Carrega as imagens e suas classes
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # aqui é feita a escala da img para [0, 1] para diminuir o impacto do brilho de cada pixel
        data.append(image/255)
        labels.append(label)
        # Verbose: Retirar?
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # Retorna uma tupla com aa matriz das imagens e seu rotulo
    return data, labels


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


def batch_data(data_shard, b=32):
    '''Recebe um fragmento de dados de um clientes e cria um objeto tensorflow data nele
    args:
        data_shard: dados e rotulos que constitui o fragmento de dados de um cliente
        b: tamanho do batch
    return:
        objeto tensorflow data'''
    # Separa os dados e os rotulos do fragmento em uma lista
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(b)


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
    predictions = model.predict(X_test)
    loss = cce(Y_test, predictions)
    acc = accuracy_score(tf.argmax(predictions, axis=1), tf.argmax(Y_test, axis=1))
    print('Agregation Round: {} | global_acc: {:.2%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


def check_local_loss(client, model):
    # Verificar perda local
    cce_l = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    client_x = np.array([i[0] for i in clients[client]])
    client_y = np.array([i[1] for i in clients[client]])
    local_predictions = model.predict(client_x)
    loss_l = cce_l(client_y, local_predictions)
    acc_l = accuracy_score(tf.argmax(local_predictions, axis=1), tf.argmax(client_y, axis=1))
    print('Local accuracy: {}. Local loss: {}'.format(acc_l, loss_l))
    return acc_l, loss_l


# Gera a lista de caminhos, utilizando a funcao list_images da biblioteca paths
image_paths = list(paths.list_images(IMG_PATH))

# carrega as imagens em arrays
image_list, label_list = load_mnist_bypath(image_paths)

# Realiza o one-hot encoded para podermos utilizar a funcao de perda sparse-categorical-entropy
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)


#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.3, random_state=42)

# Cria os clientes
clients = create_clients(X_train, y_train, num_clients=NUM_USERS, initial='client')
#print(clients.keys())

# Processa e agrupa os dados de treinamento para cada cliente
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

# processar e agrupar conjunto de teste
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

# Define o otimizador
optimizer = SGD(lr=LR, decay=LR / GLOBAL_ROUNDS, momentum=0.9)

### Inicia o modelo global ###

smlp_global = MLP()
global_model = smlp_global.build(784, 10)

def train_loss_SA(X, global_model, comm_round, clients_batched):
    #lr = X[0]
    #print('Learning rate: {}'.format(lr))
    #local_epoch = int(X[1])
    user_ids = [int(X[i]) for i in range(0, len(X) - 1)]

    global_weights = global_model.get_weights()
    scaled_local_weight_list = list()

    client_names = list(clients_batched.keys())
    client_select = [client_names[i] for i in user_ids]
    print(client_select)
    #optimizer_t = SGD(lr=lr, decay=lr / GLOBAL_ROUNDS, momentum=0.9)

    for client in client_select:
        smlp_local = MLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=LOSS, optimizer=optimizer, metrics=METRICS)

        # definir o peso do modelo local para o peso do modelo global
        local_model.set_weights(global_weights)

        # Treina o modelo local do usuario com seu respectivo dado
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # scala os pesos dos usuarios com base em seu conjunto de dados
        scaling_factor = weight_scalling_factor(clients_batched, client, client_select)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        # Verifica a acuracia local
        #acc_l, loss_l = check_local_loss(client, local_model)

        # Limpa a sessão para liberar memória após cada rodada de comunicação
        K.clear_session()

    # Para obter a média dos modelos locaia, é feita a soma dos pesos já dimensionados
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # Atualiza o modelo global
    global_model.set_weights(average_weights)

    # Testa o modelo global e printa as metricas a cada iteracao global
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

    return global_model, global_loss, global_acc


def federated_train(global_rounds, global_model, clients_batched):
    global_loss = list()
    global_accuracy = list()

    # Comecao loop de treino global
    for t in range(global_rounds):

        # Pega os pesos do modelo global
        global_weights = global_model.get_weights()

        # Lista inicial para coletar pesos do modelo local após o dimensionamento
        scaled_local_weight_list = list()

        # Embaralha os clientes e seleciona uma parte
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)
        client_select = client_names[0:55]
        #print(client_select)

        # percorre cada cliente e criar um novo modelo local
        for client in client_select:
            smlp_local = MLP()
            local_model = smlp_local.build(784, 10)
            local_model.compile(loss=LOSS,
                                optimizer=optimizer,
                                metrics=METRICS)

            # definir o peso do modelo local para o peso do modelo global
            local_model.set_weights(global_weights)

            # Treina o modelo local do usuario com seu respectivo dado
            local_model.fit(clients_batched[client], epochs=l_update, verbose=0)

            # scala os pesos dos usuarios com base em seu conjunto de dados
            scaling_factor = weight_scalling_factor(clients_batched, client, client_select)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # Verifica a acuracia local
            #acc_l, loss_l = check_local_loss(client, local_model)

            # Limpa a sessão para liberar memória após cada rodada de comunicação
            K.clear_session()

        # Para obter a média dos modelos locaia, é feita a soma dos pesos já dimensionados
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # Atualiza o modelo global
        global_model.set_weights(average_weights)

        # Testa o modelo global e printa as metricas a cada iteracao global
        for (X_test, Y_test) in test_batched:
            g_acc, g_loss = test_model(X_test, Y_test, global_model, t)
        global_loss.append(g_loss.numpy())
        global_accuracy.append(round(g_acc * 100, 2))
    return global_loss, global_accuracy


def save(name="name", **kwargs):
    for (n, dt) in kwargs.items():
        f = open('{}_{}.txt'.format(name, n), 'a')
        f.write(str(dt)[1:-1])
        f.close()


sa.initialize(GLOBAL_ROUNDS, train_loss_SA, global_model, clients_batched)
sa.save('proposta_user_select')
#federated_train(GLOBAL_ROUNDS,global_model, clients_batched)
#global_loss_list, global_accuracy_list = federated_train(GLOBAL_ROUNDS, global_model, clients_batched)
#save("baseline", loss=global_loss_list, accuracy=global_accuracy_list)

