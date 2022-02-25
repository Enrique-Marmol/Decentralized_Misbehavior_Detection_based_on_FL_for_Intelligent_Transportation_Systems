import os
import time

import numpy
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import flwr as fl
#import emlearn
from keras.layers import Embedding, GRU

from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import  f1_score, precision_score, \
    recall_score, matthews_corrcoef, cohen_kappa_score

from typing import Any, Callable, Dict, List, Optional, Tuple


def load_data(path):
    # load dataset
    training_dataset = pd.read_csv(path)
    # testing_dataset = pd.read_csv("/home/enrique/IBMFL1.0.4/testings/testing_semicomun_vehicles.csv", delimiter=",")

    # pre-process the data
    training_dataset = preprocess(training_dataset)
    # testing_dataset = preprocess(testing_dataset)

    # split the data
    x_0 = training_dataset.iloc[:, :-1]
    y_0 = training_dataset.iloc[:, -1]

    x = np.array(x_0)
    y = np.array(y_0)
    """
    x_t_0 = testing_dataset.iloc[:, :-1]
    y_t_0 = testing_dataset.iloc[:, -1]

    x_train = np.array(x_0)
    sy_train = np.array(y_0)

    self.x_test = np.array(x_t_0)
    self.y_test = np.array(y_t_0)
    """
    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)  # , random_state=42)
    """
    x_test = np.concatenate((self.x_test, x_t_0), axis=0)
    y_test = np.concatenate((self.y_test, y_t_0), axis=0)
    """
    return (x_train, y_tr), (x_test, y_te)


def preprocess(training_data):
    # Transform INF and NaN values to median
    pd.set_option('use_inf_as_na', True)
    training_data.fillna(training_data.median(), inplace=True)

    # Shuffle samples
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    # Normalize values
    scaler = MinMaxScaler()

    features_to_normalize = training_data.columns.difference(['Label'])

    training_data[features_to_normalize] = scaler.fit_transform(training_data[features_to_normalize])

    # Return preprocessed data
    return training_data


def my_FPR(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    TC = confusion_matrix.sum(axis=1)
    TC_s = TC.sum()
    TC = TC / TC_s

    FPR = FP / (FP + TN)
    FPR_div = FPR * TC
    FPR_w = FPR_div.sum()

    return FPR_w


def create_model_mlp(input_shape):
    model = Sequential()
    model.add(Dense(350, input_shape=(input_shape,), activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.0))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    nadam = Nadam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'],  # , my_recall, my_precision, my_f1_score, my_mcc, my_cks],

                  run_eagerly=True)

    return model


class tfmlpClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, party_number):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.party_number = party_number

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        dir_base = os.getcwd()
        # Update local model parameters

        # self.model.set_weights(parameters)
        mean_weights = parameters

        # Get hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        rnd = config["round"]
        steps = config["val_steps"]
        rondas = config["rounds"]

        if rnd == 1:
            self.model.set_weights(parameters)


        print(("Ronda: " + str(rnd)))
        """
        if self.datasize < data_threshold:
            sys.exit()
        """
        """
        if rnd == rondas & self.party_number == 1:
            dir_ = os.getcwd()
            modelo_50 = self.model
            modelo_50.set_weights(parameters)
            modelo_50.save(dir_ + '/modelo_50.h5')
        """

        if rnd > 1:
            for epoch in range(epochs):
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    batch_size,
                    1,
                    validation_split=0.1,
                )
                theta = 0.98
                new_param = fedplus(self.model.get_weights(), mean_weights, theta)
                self.model.set_weights(new_param)
        else:

            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs,
                validation_split=0.1,
            )
            new_param = self.model.get_weights()

        #actual = history.history["accuracy"][-1]
        """
        if self.pre_accuracy - actual > diff:
            sys.exit()
        """
        """
        theta = 0.8
        if rnd > 1:
            new_param = fedplus(self.model.get_weights(), mean_weights, theta)

        else:
            new_param = self.model.get_weights()
        """
        """
        cmodel = emlearn.convert(self.model)
        cmodel.save(file='/home/enrique/Flower/sklearn-logreg-mnist/tf red neuronal/exportado.h', name="classifier")
        """
        # Return updated model parameters and results
        # parameters_prime = self.model.get_weights()
        parameters_prime = new_param
        del new_param
        num_examples_train = len(self.x_train)

        y_pred = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred, axis=1)
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],

            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],

        }

        # Guardar en csv externo
        aux = []
        path = dir_base + '/metricas_parties/history_' + str(self.party_number) + '.csv'
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score']
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        lista = {
            "Accuracy": accuracy,
            "Recall": recall_score(self.y_test, y_pred, average='weighted'),
            "Precision": precision_score(self.y_test, y_pred, average='weighted'),
            "F1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "FPR": my_FPR(self.y_test, y_pred),
            "Matthew_Correlation_coefficient": matthews_corrcoef(self.y_test, y_pred),
            "Cohen_Kappa_Score": cohen_kappa_score(self.y_test, y_pred)
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
        del loss, accuracy, lista

        aux = []
        aux_pre = []
        path = dir_base + '/class acc/Acc_class_party_' + str(self.party_number) + '.csv'
        path_precision = dir_base + '/precision_classes/Pre_class_party_' + str(self.party_number) + '.csv'
        col_name = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']

        acc_classes = sklearn.metrics.classification_report(self.y_test, y_pred, output_dict=True)
        acc_classes = pd.DataFrame(acc_classes).transpose()

        accuracy_c = {
            "class 0": acc_classes.iloc[0, 1],
            "class 1": acc_classes.iloc[1, 1],
            "class 2": acc_classes.iloc[2, 1],
            "class 3": acc_classes.iloc[3, 1],
            "class 4": acc_classes.iloc[4, 1],
            "class 5": acc_classes.iloc[5, 1],
        }

        precision_class = {
            "class 0": acc_classes.iloc[0, 0],
            "class 1": acc_classes.iloc[1, 0],
            "class 2": acc_classes.iloc[2, 0],
            "class 3": acc_classes.iloc[3, 0],
            "class 4": acc_classes.iloc[4, 0],
            "class 5": acc_classes.iloc[5, 0],
        }

        aux.append(accuracy_c)
        aux_pre.append(precision_class)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        df2 = pd.DataFrame(aux_pre, columns=col_name)
        df2.to_csv(path_precision, index=None, mode="a", header=not os.path.isfile(path_precision))
        del aux, aux_pre, col_name, acc_classes, accuracy_c, precision_class, df1, df2

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def fedplus(weights, mean, theta):
    z = numpy.asarray(mean)
    weights = numpy.asarray(weights)

    fedp = theta * weights + (1 - theta) * z
    return fedp


def start_client(client_n):
    time.sleep(10)
    # lista_clientes = [-1,8,16,17,12,28]
    dir_base = os.getcwd()
    dir_lista_tam = dir_base + "/SMOTETomek/AaTabla_tamanos.csv"

    lista_clientes = pd.read_csv(dir_lista_tam)
    lista_clientes = np.array(lista_clientes['Vehicle'])
    del dir_lista_tam

    # Limpiamos el csv
    carpeta_nueva = "metricas_parties"
    try:
        os.mkdir(carpeta_nueva)
    except FileExistsError:
        aux = 0

    vacio = []
    col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                'Cohen_Kappa_Score']
    path = dir_base + '/metricas_parties/history_' + str(client_n) + '.csv'
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)

    carpeta_nueva = "class acc"
    try:
        os.mkdir(carpeta_nueva)
    except FileExistsError:
        aux = 0

    carpeta_nueva = "precision_classes"
    try:
        os.mkdir(carpeta_nueva)
    except FileExistsError:
        aux = 0

    del carpeta_nueva

    #limpamos csv classes ass
    vacio = []
    col_name = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    path = dir_base + '/class acc/Acc_class_party_' + str(client_n) + '.csv'
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)

    # limpamos csv classes precision
    vacio = []
    col_name = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    path = dir_base + '/precision_classes/Pre_class_party_' + str(client_n) + '.csv'
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)
    del vacio, col_name, path, df

    # Load a subset of CIFAR-10 to simulate the local data partition
    lc_ind = client_n - 1
    dir_datos = "/SMOTETomek/data_party" + str(lista_clientes[lc_ind]) + ".csv"
    # dir_datos = "originales/data_party" + str(lista_clientes[lc_ind]) + ".csv"
    dir_lista_datos = dir_base + dir_datos
    del dir_datos, lc_ind, lista_clientes

    (x_train, y_train), (x_test, y_test) = load_data(dir_lista_datos)
    del dir_lista_datos

    input_shape = len(x_train[0])
    model = create_model_mlp(input_shape)
    del input_shape
    """
    history = model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1)
    acc = history.history["accuracy"]
    loss, acc_true = model.evaluate(x_test, y_test, 32, steps=5)
    print("Party " + str(client_n) + " accuracy solo: " + str(acc_true))
    """
    #datos_total = len(y_train) + len(y_test)

    # Start Flower client
    client = tfmlpClient(model, x_train, y_train, x_test, y_test,# 0, datos_total, dir_metricas_total, dir_classes_total,
                         client_n)
    # client = sklearnClient(model, x_train, y_train, x_test, y_test)
    print(("party " + str(client_n) + " lista"))
    del model, x_train, y_train,x_test,y_test,client_n

    # IP lorien 155.54.95.95
    fl.client.start_numpy_client("[::]:8080", client=client)


def start_server(parties, rounds):
    """
    vacio = []
    col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                'Cohen_Kappa_Score']
    path = '/home/enrique/Flower/sklearn-logreg-mnist/tf red neuronal/metricas_parties/history_Server.csv'
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)
    """

    dir_base = os.getcwd()
    dir_lista_datos = dir_base + "/SMOTETomek/data_party" + str(8) + ".csv"

    (x_train, y_train), _ = load_data(dir_lista_datos)
    input_shape = len(x_train[0])
    model = create_model_mlp(input_shape)
    del dir_lista_datos, input_shape

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=parties,
        min_eval_clients=parties,
        min_available_clients=parties,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    # IP lorien 155.54.95.95
    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": rounds}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    dir_base = os.getcwd()
    dir_datos = "SMOTETomek/data_party" + str(1) + ".csv"
    dir_lista_datos = os.path.join(dir_base, dir_datos)
    dir_lista_datos = os.path.abspath(dir_lista_datos)

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _, (x_val, y_val) = load_data(dir_lista_datos)

    # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
            weights,
    ):  # -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]: model.set_weights(weights)

        # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        """
        aux = []
        path = '/home/enrique/Flower/sklearn-logreg-mnist/tf red neuronal/metricas_parties/history_Server.csv'
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score']
        lista = {
            "Accuracy": accuracy,
            "Recall": 0,
            "Precision": 0,
            "F1_score": 0,
            "Matthew_Correlation_coefficient": 0,
            "Cohen_Kappa_Score": 0
        }

        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
        """
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 5,  # if rnd < 2 else 2,
        "round": rnd,
        "val_steps": 5,
        "rounds": 20

    }
    return config


def evaluate_config(rnd):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}
