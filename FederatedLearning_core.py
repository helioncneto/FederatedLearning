import collections
import tensorflow as tf
import numpy as np
import tensorflow_federated as tff
from time import sleep
import matplotlib.pyplot as plt
import random

#tf.autograph.set_verbosity(0)

#set_updates(1)

# TODO(b/148678573,b/148685415): must use the ReferenceExecutor because it supports unbounded references and
#  tff.sequence_* intrinsics.
tff.framework.set_default_executor(tff.test.ReferenceExecutor())

NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100
LOCAL_UPS = 10

# Carrega os dados de treino e test do mnsit da base do Kears
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()


def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence

# Criacao do dataset de dez usuarios
federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

# Aqui é definido o tipo de input como uma tupla do TFF. Como o tamanho dos lotes de dados podem varias
# a dimensao do lote e declarada como None. As variaveis precisam serem criadas pelos tipos de dados
# da abstracao TensoFlow que  o TensorSpec, o float32 e o int32.
BATCH_SPEC = collections.OrderedDict(x=tf.TensorSpec(shape=[None, 784],
                                                     dtype=tf.float32), y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)

# Aqui sao criadas as especificacoes do modelo. Pesos e bias
MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)

# Nessa parte e calculada a perda do modelo.
@tf.function
def forward_pass(model, batch):
    # Faz a predicao. Multiplica as caracteristicas pelos pesos e soma com o bias.
    predicted_y = tf.nn.softmax(tf.matmul(batch['x'], model['weights']) + model['bias'])
    return -tf.reduce_mean(tf.reduce_sum(tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1]))


# O batch loss retorna uma perda em float32 dado um modelo e um lote.
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)


# Criando um modelo de teste com os pesos e bias iniciados com zeros e vamos computar a perda de um dado lote
initial_model = collections.OrderedDict(weights=np.zeros([784, 10], dtype=np.float32),
                                                  bias=np.zeros([10], dtype=np.float32))


def save_model(nome, model):
    np.savetxt(nome+'_weights.txt', model['weights'], delimiter=',')
    np.savetxt(nome+'_bias.txt', model['bias'], delimiter=',')



'''
test_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))
'''
test_batch = federated_train_data[5][-1]
'''
initial_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))

sample_batch = federated_train_data[5][-1]

print(batch_loss(initial_model, sample_batch))
'''

# Nesta etapa será computado o Gradient Descent de um dado lote de acordo com sua perda.

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`. Must
    # be defined outside the @tf.function.
    model_vars = collections.OrderedDict([(name, tf.Variable(name=name, initial_value=value)) for name, value in
                                          initial_model.items()])
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
        # Executa um passo do gradient descent utilizando a perda de 'batch_loss'
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        gradients = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(zip(tf.nest.flatten(gradients), tf.nest.flatten(model_vars)))
        return model_vars
    return _train_on_batch(model_vars, batch)


#print(str(batch_train.type_signature))


#model = initial_model
#losses = []

#for _ in range(5):
#  model = batch_train(model, test_batch, 0.1)
#  losses.append(batch_loss(model, test_batch))
#print(losses)

#Nesta parte iremos treinaer todos os lotes de um usuario, vamos criar uma funcao que treine todos os lotes.
# Para isso, vamos criar um novo tipo de dados 'SequenceType' que ira receber uma sequencis de Tipos BATCH.
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    # A funcao abaixo sera aplicada a cada lote. Essa funcao e criada pq a funcao batch_train necessita
    # de learning_rate como parametro
    #md = initial_model

    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model_, batch):
        return batch_train(model_, batch, learning_rate)

    #for _ in range(1):
        #md = tff.sequence_reduce(all_batches, md, batch_fn)
    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


#print(str(local_train.type_signature))


#locally_trained_model = local_train(test_model[0], 0.1, federated_train_data[5])

# Avaliacao  do modelo treinado localmente com base em todos os lotes
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
  # TODO(b/120157713): Replace with `tff.sequence_average()` once implemented.
  return tff.sequence_sum(tff.sequence_map(tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
                                           all_batches))

#print(str(local_eval.type_signature))

# Verificando a diferenca da perda do modelo inicial com o modelo treinado com os dados do usuario
#print('initial_model loss =', local_eval(test_model[0], federated_train_data[5]))
#print('locally_trained_model loss =', local_eval(locally_trained_model, federated_train_data[5]))

# Verificando a diferenca de perda do modelo inicial e o modelo treinado com o dataset de outro usuario
#print('initial_model loss =', local_eval(test_model[0], federated_train_data[0]))
#print('locally_trained_model loss =', local_eval(locally_trained_model, federated_train_data[0]))

# Constantes globais do ambiente federado
SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)
SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER)


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))

#print('initial_model loss =', federated_eval(test_model[0], federated_train_data))
#print('locally_trained_model loss =', federated_eval(locally_trained_model, federated_train_data))


@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    return tff.federated_mean(tff.federated_map(local_train, [tff.federated_broadcast(model),
                                                              tff.federated_broadcast(learning_rate), data]))

#model = federated_train(model, learning_rate, federated_train_data)

model = initial_model
learning_rate = 0.1
federated_losses = []
#model = federated_train(model, learning_rate, federated_train_data)
#loss = federated_eval(model, federated_train_data)
#print(loss)


model = federated_train(model, learning_rate, federated_train_data)
loss = federated_eval(model, federated_train_data)
print(loss)


model = initial_model
learning_rate = 0.1
federated_losses = []

for round_num in range(10):
    #randomUsers = [federated_train_data[random.randint(0,9)], federated_train_data[random.randint(0,9)],
    #     federated_train_data[random.randint(0,9)], federated_train_data[random.randint(0,9)],
    #     federated_train_data[random.randint(0,9)]]
    model = federated_train(model, learning_rate, federated_train_data)
    learning_rate = learning_rate * 0.9
    loss = federated_eval(model, federated_train_data)
    federated_losses.append(loss)
    print('round {}, loss={}'.format(round_num, loss))

print("The list of losses: ", federated_losses)

plt.plot(federated_losses)
plt.show()

