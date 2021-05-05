#!/usr/bin/env python
# coding: utf-8

# # A célula abaixo apresenta o algoritmo Simulated Annealing Federado

# In[1]:


# Ative essa célula para desabilitar GPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'


# In[2]:


import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import warnings
from keras.layers import Dense, Dropout
from keras.metrics import Precision, Recall, Accuracy
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
warnings.simplefilter("ignore")

# ------------------------------------------------------------------------------
# Customization section:


class SimulatedAnnealing:

    def __init__(self, initial_temperature=0.5, cooling=0.08, lr=(0.0001, 0.1), local_update=(1, 20),
                 participants=(0, 99, 30), computing_time=1, no_attempts=100, threshold=0.1):
        self.initial_temperature = initial_temperature
        self.cooling = cooling  # cooling coefficient
        #self.number_variables = number_variables
        self.local_update = local_update
        self.participants = participants
        self.lr = lr
        self.computing_time = computing_time  # second(s)
        self.no_attempts = no_attempts
        self.record_best_fitness = []
        self.record_best_fitness_acc = []
        self.plus = True
        self.threshold = threshold
        
    def _select_range_int(self, list_space, choice, best=None, neigh_size=2, plus=False):
        print(plus)
        lista_escolhidos = list()
        list_space = list(np.arange(list_space[0], list_space[1] + 1))
        if best == None:
            for _ in range(choice):
                escolhido = False
                escolha = random.choice(list_space)
                while(not escolhido):
                    if escolha in list_space:
                        idx_escolha = list_space.index(escolha)
                        list_space.pop(idx_escolha)
                        lista_escolhidos.append(escolha)
                        escolhido = True
                    else:
                        escolha = random.choice(list_space)
        else:
            for i in best:
                escolhido = False
                giveup = False
                neigh = 1
                if not plus:
                    escolha = i - neigh
                else:
                    escolha = i + neigh
                while(not escolhido):
                    #print("I: ", i)
                    #print("Escolha: ", escolha)
                    if escolha in list_space:
                        idx_escolha = list_space.index(escolha)
                        list_space.pop(idx_escolha)
                        lista_escolhidos.append(escolha)
                        escolhido = True
                    else:
                        neigh += 1
                        if neigh <= neigh_size:
                            if not plus:
                                #print("subtrai")
                                escolha = i - neigh  
                            else:
                                #print("soma")
                                escolha = i + neigh
                        elif not giveup:
                            plus = not plus
                            giveup = True
                            neigh = 1
                            if plus:
                                escolha = i + neigh
                            else:
                                escolha = i - neigh
                    if escolha == (i + neigh_size):
                        if i in list_space:
                            lista_escolhidos.append(i)
                            escolhido = True
                        else:
                            escolha = random.choice(list_space)
                            lista_escolhidos.append(escolha)
                            escolhido = True
        return lista_escolhidos

    def _select_range_float(self, list_space, best=None, plus=False):
        if best is None:
            choice = random.uniform(list_space[0], list_space[1])
        else:
            if plus:
                choice = best + 0.1 * (random.uniform(list_space[0], list_space[1]))
            else:
                choice = best - 0.1 * (random.uniform(list_space[0], list_space[1]))
        return abs(choice)

    def save_model(self, nome, model):
        np.savetxt(nome + '_weights.txt', model['weights'], delimiter=',')
        np.savetxt(nome + '_bias.txt', model['bias'], delimiter=',')

    def objective_function(self, X, ob, model, data, r):
        model, loss, acc = ob(X, model, data, r)
        return model, loss, acc

    # ------------------------------------------------------------------------------
    # Simulated Annealing Algorithm:
    def initialize(self, epoch, obj, model, data):
        initial_solution = list()
        '''
        for v in range(self.number_variables):
            if v == 0 and self.lr is True:
                initial_solution[v] = random.uniform(self.lower_bounds[v], self.upper_bounds[v])
            else:
                initial_solution[v] = int(random.randrange(self.lower_bounds[v], self.upper_bounds[v]))
        '''
        initial_solution.append(self._select_range_float(self.lr))
        initial_solution.append(self._select_range_int(self.local_update, 1)[0])
        [initial_solution.append(i) for i in self._select_range_int(self.participants[:2], self.participants[2])]
        current_solution = initial_solution
        #print(initial_solution)
        best_solution = initial_solution
        n = 1  # no of solutions accepted
        model, best_fitness, acc = self.objective_function(best_solution, obj, model, 0, data) # Melhor perda
        current_temperature = self.initial_temperature  # current temperature
        start = time.time()
          # number of attempts in each level of temperature

        for i in range(1, epoch):
            for j in range(self.no_attempts):
                print("Escolha de novo parâmetros")
                current_solution = list()
                current_solution.append(self._select_range_float(self.lr, best=best_solution[0]))
                current_solution.append(self._select_range_int(self.local_update, 1, best=[best_solution[1]], 
                                        plus=self.plus)[0])
                [current_solution.append(i) for i in self._select_range_int(self.participants[:2], self.participants[2], 
                                                                            best=best_solution[2:], plus=self.plus)]
                print("Teste dos novos parâmetros")
                model, current_fitness, acc = self.objective_function(current_solution, obj, model, i, data)
                energy = abs(current_fitness - best_fitness)
                #print(current_solution)
                if i == 1 and j == 0:
                    EA = energy
                print("Fitness atual: ", current_fitness)
                print("Best Fitness: ", best_fitness)
                print("Energia", energy)
                print("Limiar", self.threshold)
                if current_fitness > best_fitness or energy < self.threshold:
                    print("Solução atual é pior")
                    self.plus = not self.plus
                    #p = math.exp(-energy.numpy() / (EA * current_temperature))
                    p = math.exp(-energy.numpy() / (current_temperature))
                    aleatorio = random.random()
                    #print("Energia: ", energy)
                    #print("Temperatura: ", current_temperature)
                    #print("P: ", p)
                    #print("Aleatorio: ", aleatorio)
                    # make a decision to accept the worse solution or not
                    if aleatorio < p:
                        print("A solução pior foi aceita")                
                        accept = True  # this worse solution is accepted
                    else:
                        print("A solução pior não foi aceita")
                        accept = False  # this worse solution is not accepted
                        print("Avaliando o best")
                        model, test_best_fitness, acc = self.objective_function(best_solution, obj, model, i, data)
                        if best_fitness > test_best_fitness:
                            print("A best solution não é mais a melhor")
                            print("Escolhendo novos parâmetros")
                            current_temperature = min(self.initial_temperature, current_temperature + (self.initial_temperature * 0.4))
                            best_solution = list()
                            best_solution.append(self._select_range_float(self.lr))
                            best_solution.append(self._select_range_int(self.local_update, 1)[0])
                            [best_solution.append(i) for i in self._select_range_int(self.participants[:2], 
                                                                                        self.participants[2])]
                        else:
                            print("O best continua sendo o melhor")
                else:
                    print("Solução atual é melhor")
                    accept = True  # accept better solution
                if accept:
                    best_solution = current_solution  # update the best solution
                    model, best_fitness, acc = self.objective_function(best_solution, obj, model, i, data)
                    n = n + 1  # count the solutions accepted
                    EA = (EA * (n - 1) + energy) / n  # update EA
                print(best_solution)

            #print('interation: {}, best_solution: {}, best_fitness: {}'.format(i, best_solution, best_fitness))
            self.record_best_fitness.append(best_fitness.numpy())
            self.record_best_fitness_acc.append(acc)
            # Cooling the temperature
            current_temperature = current_temperature * self.cooling
            # Stop by computing time
            end = time.time()
            if end - start >= self.computing_time:
                pass
                #break
        #self.save_model('proposta', model)



    def plot(self):
        plt.plot(self.record_best_fitness)
        plt.show()

    def save(self, name="name"):

        f = open(name + '_loss.txt', 'a')
        f.write(str(self.record_best_fitness)[1:-1])
        f.close()
        f = open(name + '_accuracy.txt', 'a')
        f.write(str(self.record_best_fitness_acc)[1:-1])
        f.close()


# In[3]:


import numpy as np
import pandas as pd
from keras.utils import np_utils
import random
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K

# LR: 0.010000 to 0.00001
# local update: 15 to 1
sa = SimulatedAnnealing(initial_temperature=0.8, cooling=0.05, lr=(0.0001, 0.001), local_update=(0, 10),
                        participants=(0, 99, 30), computing_time=1, no_attempts=3, threshold=0.01)


# In[4]:


#from keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
# Constantes
NUM_USERS = 100
# Declara o caminho do conjunto de dados mnist
#IMG_PATH = '/home/helio/PycharmProjects/FederatedLearning/mnist/trainingSet/trainingSet'
LR = 0.01
GLOBAL_ROUNDS = 50
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
#(x_train, y_train), (x_test, y_test) = load_data()
#x = np.load('x_new.npy')
#y = np.load('y_new.npy')
x = np.load('../x_novo.npy')
y = np.load('../y_novo.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=31)


# In[5]:


# Cria um DataFrame Pandas para ordenar o conjunto de dados para distribuição Non-IID
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train, columns=['Normal', 'Attack'])
train = pd.concat([x_train_pd, y_train_pd], axis=1)
train.sort_values(by='Normal', inplace=True)
x_train = train.drop(['Normal', 'Attack'], axis=1).to_numpy()
y_train = pd.concat([train['Normal'], train['Attack']], axis=1).to_numpy()

# As caracteristicas viram um array, aqui colocamos no formato correto
#x_train_shaped = []
#[x_train_shaped.append(x_train[i]) for i in range(len(x_train))]
#x_train = np.array(x_train_shaped)
#y_train_pd.sort_values(by='Normal')


# In[6]:


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


# In[7]:


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


# In[8]:


class MLP:
    @staticmethod
    def build(shape, classes):
        model =  Sequential()
        model.add(Dense(10, input_dim=shape, kernel_initializer='normal', activation='relu'))#, kernel_regularizer=l2(0.1)))
        #model.add(Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(Dense(classes, kernel_initializer='normal', activation='softmax'))#, kernel_regularizer=l2(0.1)))
        return model


# In[9]:


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


# In[10]:


def scale_model_weights(weight, scalar):
    '''Escala os pesos do modelo'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


# In[11]:


def sum_scaled_weights(scaled_weight_list):
    '''Retorne a soma dos pesos dimensionados listados. O é equivalente ao peso médio dos pesos'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


# In[12]:


def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    predictions = model.predict(X_test)
    loss = cce(Y_test, predictions)
    acc = accuracy_score(tf.argmax(predictions, axis=1), tf.argmax(Y_test, axis=1))
    print('Agregation Round: {} | global_acc: {:.2%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss 


# In[13]:


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


# In[14]:


# Gera a lista de caminhos, utilizando a funcao list_images da biblioteca paths
#image_paths = list(paths.list_images(IMG_PATH))

# carrega as imagens em arrays
#image_list, label_list = load_mnist_bypath(image_paths)

# Realiza o one-hot encoded para podermos utilizar a funcao de perda sparse-categorical-entropy
#lb = LabelBinarizer()
#label_list = lb.fit_transform(label_list)


#split data into training and test set
#X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.3, random_state=42)

# Cria os clientes
clients = create_clients(x_train, y_train, num_clients=NUM_USERS, initial='client')
#print(clients.keys())

# Processa e agrupa os dados de treinamento para cada cliente
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

# processar e agrupar conjunto de teste
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

# Define o otimizador
optimizer = Adam(lr=LR, amsgrad=False) #(lr=LR, decay=LR / GLOBAL_ROUNDS, momentum=0.9)

### Inicia o modelo global ###

smlp_global = MLP()
global_model = smlp_global.build(64, 2)


# In[15]:


def train_loss_SA(X, global_model, comm_round, clients_batched):
    lr = X[0]
    #print('Learning rate: {}'.format(lr))
    local_epoch = int(X[1])
    #local_epoch = int(X[0])
    user_ids = [int(X[i]) for i in range(2, len(X) - 1)]

    global_weights = global_model.get_weights()
    scaled_local_weight_list = list()

    client_names = list(clients_batched.keys())
    client_select = [client_names[i] for i in user_ids]
    #print(client_select)
    optimizer_t = SGD(lr=lr, decay=lr / GLOBAL_ROUNDS, momentum=0.9)

    for client in client_select:
        smlp_local = MLP()
        local_model = smlp_local.build(64, 2)
        local_model.compile(loss=LOSS, optimizer=optimizer_t, metrics=METRICS)

        # definir o peso do modelo local para o peso do modelo global
        local_model.set_weights(global_weights)

        # Treina o modelo local do usuario com seu respectivo dado
        local_model.fit(clients_batched[client], epochs=local_epoch, verbose=0)

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


# In[16]:


def train_loss_SA_multi(X, global_model, comm_round, clients_batched):
    def client_train(client):
        #print("Iniciando o cliente: ", client)
        smlp_local = MLP()
        local_model = smlp_local.build(64, 2)
        local_model.compile(loss=LOSS,
                            optimizer=optimizer,
                            metrics=METRICS)

        # definir o peso do modelo local para o peso do modelo global
        local_model.set_weights(global_model.get_weights())

        # Treina o modelo local do usuario com seu respectivo dado
        #print("Treinando o modelo local do cliente: ", client)
        local_model.fit(clients_batched[client], epochs=local_epoch, verbose=0)
        #print("Fim do treino do cliente: ", client)

        # scala os pesos dos usuarios com base em seu conjunto de dados
        scaling_factor = weight_scalling_factor(clients_batched, client, client_select)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        #K.clear_session()
        return scaled_weights
    lr = X[0]
    #print('Learning rate: {}'.format(lr))
    local_epoch = int(X[1])
    #local_epoch = int(X[0])
    user_ids = [int(X[i]) for i in range(2, len(X) - 1)]

    global_weights = global_model.get_weights()
    scaled_local_weight_list = list()

    client_names = list(clients_batched.keys())
    client_select = [client_names[i] for i in user_ids]
    #print(client_select)
    optimizer_t = SGD(lr=lr, decay=lr / GLOBAL_ROUNDS, momentum=0.9)
    
    # percorre cada cliente e criar um novo modelo local
    # Melhor 9 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for x in executor.map(client_train, client_select):
            scaled_local_weight_list.append(x)
        K.clear_session()
    # Limpa a sessão para liberar memória após cada rodada de comunicação
    
    #K.clear_session()

    # Para obter a média dos modelos locaia, é feita a soma dos pesos já dimensionados
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    

    # Atualiza o modelo global
    global_model.set_weights(average_weights)

    # Testa o modelo global e printa as metricas a cada iteracao global
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

    return global_model, global_loss, global_acc


# In[17]:


# Função para multi-processing
def local_training(user):
    smlp_local = MLP()
    local_model = smlp_local.build(64, 2)
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


# In[18]:


def federated_train(global_rounds, global_model, clients_batched, l_update):
    global_loss = list()
    global_accuracy = list()

    # Comecao loop de treino global
    for t in range(global_rounds):

        # Pega os pesos do modelo global
        print("Iteração global: ", t)
        global_weights = global_model.get_weights()

        # Lista inicial para coletar pesos do modelo local após o dimensionamento
        scaled_local_weight_list = list()

        # Embaralha os clientes e seleciona uma parte
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)
        client_select = client_names[0:30]
        #print(client_select)

        # percorre cada cliente e criar um novo modelo local
        for client in client_select:
            #print("Iniciando o cliente: ", client)
            smlp_local = MLP()
            local_model = smlp_local.build(64, 2)
            local_model.compile(loss=LOSS,
                                optimizer=optimizer,
                                metrics=METRICS)

            # definir o peso do modelo local para o peso do modelo global
            local_model.set_weights(global_weights)

            # Treina o modelo local do usuario com seu respectivo dado
            #print("Treinando o modelo local do cliente: ", client)
            local_model.fit(clients_batched[client], epochs=l_update, verbose=0)
            #print("Fim do treino do cliente: ", client)

            # scala os pesos dos usuarios com base em seu conjunto de dados
            scaling_factor = weight_scalling_factor(clients_batched, client, client_select)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            print(scaled_weights)
            scaled_local_weight_list.append(scaled_weights)

            # Verifica a acuracia local
            #acc_l, loss_l = check_local_loss(client, local_model)

            # Limpa a sessão para liberar memória após cada rodada de comunicação
            K.clear_session()

        # Para obter a média dos modelos locaia, é feita a soma dos pesos já dimensionados
        #print("Etapa de agregação da iteração: ", t)
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # Atualiza o modelo global
        global_model.set_weights(average_weights)

        # Testa o modelo global e printa as metricas a cada iteracao global
        for (X_test, Y_test) in test_batched:
            g_acc, g_loss = test_model(X_test, Y_test, global_model, t)
        global_loss.append(g_loss.numpy())
        global_accuracy.append(round(g_acc * 100, 2))
    return global_loss, global_accuracy


# In[19]:


import concurrent.futures
            
            
def federated_train_multiproc(global_rounds, global_model, clients_batched, l_update):
    def client_train(client):
        #print("Iniciando o cliente: ", client)
        smlp_local = MLP()
        local_model = smlp_local.build(64, 2)
        local_model.compile(loss=LOSS,
                            optimizer=optimizer,
                            metrics=METRICS)

        # definir o peso do modelo local para o peso do modelo global
        local_model.set_weights(global_model.get_weights())

        # Treina o modelo local do usuario com seu respectivo dado
        #print("Treinando o modelo local do cliente: ", client)
        local_model.fit(clients_batched[client], epochs=l_update, verbose=0)
        #print("Fim do treino do cliente: ", client)

        # scala os pesos dos usuarios com base em seu conjunto de dados
        scaling_factor = weight_scalling_factor(clients_batched, client, client_select)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        return scaled_weights
    global_loss = list()
    global_accuracy = list()

    # Comecao loop de treino global
    for t in range(global_rounds):
        # Pega os pesos do modelo global
        print("Iteração global: ", t)
        global_weights = global_model.get_weights()

        # Lista inicial para coletar pesos do modelo local após o dimensionamento
        scaled_local_weight_list = list()

        # Embaralha os clientes e seleciona uma parte
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)
        client_select = client_names[0:30]
        #print(client_select)

        # percorre cada cliente e criar um novo modelo local
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            for x in executor.map(client_train, client_select):
                scaled_local_weight_list.append(x)
        # Limpa a sessão para liberar memória após cada rodada de comunicação
            K.clear_session()

        # Para obter a média dos modelos locaia, é feita a soma dos pesos já dimensionados
        #print("Etapa de agregação da iteração: ", t)
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # Atualiza o modelo global
        global_model.set_weights(average_weights)

        # Testa o modelo global e printa as metricas a cada iteracao global
        for (X_test, Y_test) in test_batched:
            g_acc, g_loss = test_model(X_test, Y_test, global_model, t)
        global_loss.append(g_loss.numpy())
        global_accuracy.append(round(g_acc * 100, 2))
    return global_loss, global_accuracy


# In[20]:


def save(name="name", sv=False, **kwargs):
    for (n, dt) in kwargs.items():
        f = open('{}_{}.txt'.format(name, n), 'a')
        if sv:
            f.write(str(dt)[:-1])
        else:
            f.write(str(dt)[1:-1])
        f.close()


# In[21]:


#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
#sa.initialize(GLOBAL_ROUNDS, train_loss_SA_multi, global_model, clients_batched)
#sa.save('resultados/proposta/proposta_funcional/proposta_1_10_4th')
#federated_train(GLOBAL_ROUNDS, global_model, clients_batched)
#global_loss_list, global_accuracy_list = federated_train(GLOBAL_ROUNDS, global_model, clients_batched, 100)
#global_loss_list, global_accuracy_list = federated_train_multiproc(10, global_model, clients_batched, 2)
#for i in range(1, 16):
    #smlp_global = MLP()
    #global_model = smlp_global.build(64, 2)
    #global_loss_list, global_accuracy_list = federated_train_multiproc(90, global_model, clients_batched, i)
    #for (X_test, Y_test) in test_batched:
        #preds = global_model.predict(X_test)
        #preds_class = [np.argmax(preds[i]) for i in range(len(preds))]
        #y_class = [np.argmax(Y_test[i]) for i in range(len(Y_test))]
        #cm = confusion_matrix(y_class, preds_class)
        #tn, fp, fn, tp = confusion_matrix(y_class, preds_class).ravel()
        #accuracy = accuracy_score(y_class, preds_class) * 100
        #precision = (tp / (tp + fp)) * 100
        #sensibilidade = (tp / (tp + fn)) * 100
        #especificidade = (tn / (tn + fp)) * 100
    #save("resultados/baseline/baseline_30ag_window_" + str(i), sv=True, accuracy=accuracy, precision=precision,
        #sensibilidade=sensibilidade, especificidade=especificidade)


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
#sa.initialize(GLOBAL_ROUNDS, train_loss_SA_multi, global_model, clients_batched)
#sa.save('resultados/proposta/proposta_funcional/proposta_1_10_4th')
#federated_train(GLOBAL_ROUNDS, global_model, clients_batched)
#global_loss_list, global_accuracy_list = federated_train(GLOBAL_ROUNDS, global_model, clients_batched, 100)
#global_loss_list, global_accuracy_list = federated_train_multiproc(10, global_model, clients_batched, 2)
#for i in [10, 20, 30]:
for i in [10]:
    smlp_global = MLP()
    global_model = smlp_global.build(64, 2)
    federated_train_multiproc(i, global_model, clients_batched, 7)
    #sa.initialize(i, train_loss_SA_multi, global_model, clients_batched)
    for (X_test, Y_test) in test_batched:
        preds = global_model.predict(X_test)
        preds_class = [np.argmax(preds[i]) for i in range(len(preds))]
        y_class = [np.argmax(Y_test[i]) for i in range(len(Y_test))]
        cm = confusion_matrix(y_class, preds_class)
        tn, fp, fn, tp = confusion_matrix(y_class, preds_class).ravel()
        accuracy = accuracy_score(y_class, preds_class) * 100
        precision = (tp / (tp + fp)) * 100
        sensibilidade = (tp / (tp + fn)) * 100
        especificidade = (tn / (tn + fp)) * 100
    save("resultados/baseline/baseline_"+ str(i) +"ag_window_7_3rd", sv=True, accuracy=accuracy, precision=precision,
        sensibilidade=sensibilidade, especificidade=especificidade)
    #save("resultados/proposta/proposta_funcional/proposta_"+ str(i) +"ag_3", sv=True, accuracy=accuracy, precision=precision,
        #sensibilidade=sensibilidade, especificidade=especificidade)


# In[23]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
for (X_test, Y_test) in test_batched:
    preds = global_model.predict(X_test)
    preds_class = [np.argmax(preds[i]) for i in range(len(preds))]
    y_class = [np.argmax(Y_test[i]) for i in range(len(Y_test))]
    cm = confusion_matrix(y_class, preds_class)
    tn, fp, fn, tp = confusion_matrix(y_class, preds_class).ravel()


# In[24]:


precision = tp / (tp + fp)
sensibilidade = tp / (tp + fn)
especificidade = tn / (tn + fp)


# In[25]:


print(precision)
print(sensibilidade)
print(especificidade)


# In[26]:


print(classification_report(y_class, preds_class))

