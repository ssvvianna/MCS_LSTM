########### import libraries
from keras.layers.core import Lambda
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import time
import warnings
warnings.filterwarnings("ignore")
import os
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import winsound
import seaborn as sns
import random
from openpyxl import load_workbook
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from keras.regularizers import l2
def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

class MonteCarloLSTM(tf.keras.layers.LSTM):
   def call(self, inputs):
      return super().call(inputs, training=True)

address = os.getcwd()

line_width=2; marker_size=20; size=25
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)   # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 2  # set the value globally
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

##################### Import CFD simulation data ##############
csv = {}
Vel = [2, 4, 6, 8] # 2, 4, 6, 8
Mod = ['NL']
Desc = [0]
Dir = ['NORTE', 'NORDESTE','LESTE', 'SUDESTE', 'SUL', 'SUDOESTE', 'OESTE', 'NOROESTE'] # 'NORTE', 'NORDESTE','LESTE', 'SUDESTE', 'SUL', 'SUDOESTE', 'OESTE', 'NOROESTE'
first = 0
direcao = []
ponto = []
velocidade = []
linhas = 33
ordem = []
############ Import data - no leak 

for m in Mod:
    for j in Dir:
        for w in Desc:
            for i in Vel:
                path = address + '\\Final_data\\M{}-DV{}-TD{}-V{}'.format(m, j, w, i) + '.xlsx'
                csv[path] = pd.read_excel(path, usecols = "D:CU")
                if first == 0:
                    saida_aux= np.array(csv[path])
                    colunas = saida_aux.shape[1]
                    out = []
                    for k in range(linhas):
                        out.append(saida_aux[k,:])
                    saida = np.array(out).reshape([int(linhas), colunas])
                    first = 1
                    ordem.append('M{}-DV{}-TD{}-V{}'.format(m, j, w, i))
                else:
                    out = []
                    saida_aux= np.array(csv[path])
                    for k in range(linhas):
                        out.append(saida_aux[k, :])
                    saida_aux = np.array(out).reshape([int(linhas), colunas])
                    saida = np.concatenate((saida, saida_aux), axis=0)
                    ordem.append('M{}-DV{}-TD{}-V{}'.format(m, j, w, i))
                for k in range(int(linhas)):
                    direcao.append(j)
                    ponto.append(m)
                    velocidade.append(i)

################# Import data - leak 

Mod = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4'] # 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4'
Desc = [10, 50, 100]

for m in Mod:
    for j in Dir:
        for i in Vel:
            for w in Desc:
                path = address + '\\Final_data\\M{}-DV{}-TD{}-V{}'.format(m, j, w, i) + '.xlsx'
                csv[path] = pd.read_excel(path, usecols = "C:CT")
                out = []
                saida_aux= np.array(csv[path])
                for k in range(linhas):
                    out.append(saida_aux[k, :])
                saida_aux = np.array(out).reshape([int(linhas), colunas])
                saida = np.concatenate((saida, saida_aux), axis=0)
                ordem.append('M{}-DV{}-TD{}-V{}'.format(m, j, w, i))
                for k in range(int(linhas)):
                    direcao.append(j)
                    ponto.append(m)
                    velocidade.append(i)

######### Normalisation and housekeeping 
velocidade = np.reshape(velocidade, [len(velocidade),1])
saida = np.concatenate((saida, velocidade), axis=1)
colunas = saida.shape[1]
normalizar = MinMaxScaler(feature_range=(0, 1))
norm_dataset = normalizar.fit_transform(saida)

encoder_direcao = LabelEncoder()
encoder_direcao.fit(direcao)
encoded_direcao = encoder_direcao.transform(direcao)
onehot_direcao = np_utils.to_categorical(encoded_direcao)

encoder_ponto = LabelEncoder()
encoder_ponto.fit(ponto)
encoded_ponto = encoder_ponto.transform(ponto)
onehot_ponto = np_utils.to_categorical(encoded_ponto)

saida = np.reshape(norm_dataset, [-1, int(linhas), colunas])
direcao = np.reshape(onehot_direcao, [-1, int(linhas), onehot_direcao.shape[1]])
ponto = np.reshape(onehot_ponto, [-1, int(linhas), onehot_ponto.shape[1]])

input_variables = np.concatenate((saida, direcao), axis=2)
##################### Scenario selection - training and test ##############
# 26/6
# random_list = np.array(pd.read_excel(address + '\\selection.xlsx'))
# list_training, list_test = [], []
# for i in range(len(random_list)):
#     if random_list[i, 1] == 'Teste':
#         list_test.append(i)
#     else:
#         list_training.append(i)
list_test = [296, 495, 218, 620, 684, 260, 414, 317, 390, 226, 187, 595, 587, 690, 128, 488, 116, 524, 468, 335, 432,
 310, 718, 13, 702, 355, 396, 209, 19, 357, 477, 500, 351, 695, 433, 728, 63, 589, 3, 381, 518, 69, 658, 766, 747, 184,
 55, 784, 713, 442, 551, 291, 765, 783, 145, 111, 246, 382, 68, 576, 522, 735, 506, 557, 404, 791, 717, 427, 532, 120,
 444, 141, 637, 448, 207, 721, 694, 679, 217, 703, 339, 84, 626, 129, 591, 333, 397, 316, 563, 287, 498, 125, 20, 362,
 606, 286, 198, 692, 257, 332, 729, 151, 670, 162, 123, 649, 88, 418, 711, 175, 672, 45, 281, 669, 604, 73, 86, 487,
 704, 389, 23, 367, 508, 211, 71, 751, 631, 100, 290, 635, 278, 238, 233, 726, 546, 214, 743, 10, 529, 545, 566, 446,
 327, 153, 230, 469, 261, 89, 166, 36]
list_training = [237, 117, 638, 24, 676, 42, 715, 610, 180, 294, 30, 344, 709, 399, 202, 212, 797, 580, 325, 250, 177,
 654, 422, 251, 507, 408, 179, 43, 155, 636, 105, 603, 646, 172, 74, 657, 51, 255, 611, 106, 282, 41, 531, 758, 489,
 537, 410, 76, 499, 65, 146, 196, 302, 496, 544, 752, 501, 475, 780, 329, 632, 288, 462, 511, 130, 167, 64, 249, 376,
 513, 135, 447, 467, 276, 199, 409, 486, 605, 643, 224, 774, 168, 34, 227, 37, 421, 295, 268, 252, 314, 5, 420, 639,
 443, 416, 738, 170, 629, 38, 481, 528, 560, 470, 460, 67, 597, 742, 798, 361, 451, 185, 660, 102, 415, 570, 732, 39,
 173, 479, 568, 182, 164, 454, 66, 338, 457, 565, 308, 305, 104, 452, 666, 387, 664, 143, 25, 699, 80, 298, 493, 307,
 178, 450, 793, 788, 247, 517, 588, 436, 79, 312, 163, 342, 220, 235, 763, 525, 490, 193, 512, 773, 127, 683, 85, 22,
 330, 478, 91, 471, 374, 189, 668, 269, 161, 337, 567, 348, 401, 574, 248, 7, 270, 57, 236, 154, 558, 682, 494, 12, 530,
 345, 708, 693, 740, 82, 313, 315, 253, 794, 340, 103, 503, 548, 194, 346, 256, 585, 75, 688, 94, 262, 556, 795, 319,
 411, 736, 283, 710, 243, 547, 716, 535, 60, 109, 579, 383, 394, 412, 364, 299, 572, 228, 360, 28, 2, 136, 746, 569,
 304, 614, 673, 523, 445, 289, 320, 419, 612, 27, 311, 54, 40, 755, 601, 204, 697, 762, 536, 358, 455, 624, 661, 700,
 201, 31, 277, 641, 229, 586, 280, 159, 356, 627, 555, 210, 232, 393, 674, 8, 318, 516, 662, 96, 157, 698, 583, 429,
 119, 786, 594, 392, 371, 388, 652, 359, 273, 292, 328, 160, 559, 671, 778, 267, 324, 463, 192, 301, 197, 705, 275, 126,
 370, 242, 156, 87, 543, 140, 375, 482, 121, 465, 407, 354, 596, 677, 165, 549, 368, 35, 271, 131, 527, 714, 183, 787,
 16, 734, 621, 642, 550, 689, 622, 95, 53, 744, 52, 598, 17, 365, 417, 737, 341, 723, 203, 573, 366, 653, 613, 113, 696,
 686, 97, 33, 274, 14, 99, 640, 331, 757, 584, 745, 423, 112, 21, 384, 727, 644, 303, 347, 552, 616, 739, 466, 459, 725,
 147, 413, 634, 254, 174, 380, 607, 206, 582, 789, 1, 213, 750, 691, 628, 480, 741, 492, 577, 796, 369, 200, 134, 144,
 439, 195, 215, 323, 108, 171, 541, 373, 124, 435, 216, 9, 768, 406, 47, 176, 792, 402, 630, 234, 93, 764, 377, 239,
 656, 519, 352, 98, 609, 633, 476, 32, 651, 44, 191, 272, 592, 77, 502, 90, 83, 770, 619, 118, 259, 754, 426, 363, 78,
 562, 509, 685, 790, 114, 149, 534, 733, 379, 687, 571, 430, 681, 186, 6, 231, 300, 665, 70, 181, 497, 485, 748, 453,
 343, 533, 428, 526, 775, 284, 405, 398, 139, 264, 62, 49, 372, 491, 400, 188, 190, 761, 385, 81, 625, 223, 440, 772,
 322, 593, 434, 756, 263, 11, 18, 753, 48, 799, 473, 306, 759, 505, 719, 776, 602, 72, 675, 540, 15, 46, 731, 707, 169,
 158, 542, 424, 132, 148, 336, 779, 395, 618, 61, 29, 553, 538, 241, 647, 205, 219, 581, 706, 92, 152, 403, 578, 441,
 782, 659, 101, 309, 575, 722, 138, 133, 514, 645, 386, 520, 474, 608, 771, 483, 222, 599, 461, 285, 326, 221, 749,
 438, 712, 137, 391, 564, 521, 240, 781, 245, 456, 293, 464, 353, 617, 777, 265, 110, 297, 484, 615, 58, 539, 655, 349,
 321, 720, 258, 767, 760, 350, 279, 115, 510, 425, 554, 122, 0, 142, 650, 472, 623, 724, 600, 26, 449, 4, 50, 225, 680,
 504, 648, 663, 730, 107, 769, 334, 266, 431, 515, 437, 378, 678, 667, 590, 701, 785, 208, 59, 244, 56, 561, 458]
descricao = np.array(pd.read_excel(address+'\\selection.xlsx', usecols='A'))

############# Creating a data structure with look_back n timesteps ##########
def create_dataset(index_list, look_back, saida, ponto, descricao):
    x_train, y_train = [], []
    scenario = []
    for j in index_list:
        data = np.array(saida[j,:,:])
        ponto_spec = np.array(ponto[j,:,:])
        for i in range(look_back, len(data)):
            x_train.append(data[i - look_back:i, :])
            y_train.append(ponto_spec[i, :])
            scenario.append([j, descricao[j]])
    return np.array(x_train), np.array(y_train), np.array(scenario).reshape([-1, 2])

################## Creating the parameters set #####################
def model_configs():
    configs = list()
    for i in neurons:
        for k in l_1:
            for l in l_2:
                for m in do:
                    for j in init_mode:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

################### Parameters to modify ######################
lstm = 1
look_back=5        # number of time steps used to predict
ni=input_variables.shape[2]                # number of input variables
no = ponto.shape[2]              # number of output variables
f_train=.8          # fraction of dataset used in the training stage
en=1000              # epochs
bs=32               # batch size
vs=.1               # validation split percentage
patience=25         # early stopping check number
neurons = [10]      # number of neurons 9, 12, 15, 18, 21, 24
init_mode=['random_normal'] # weights distribution 'random_normal', 'random_uniform', 'identity'
l_1=[0]              # L1 regularization weight
l_2=[0]              # L2 regularization weight
do=[.3]             # dropout percentage rate
weight_decay = 1e-4
########################## Search Tree ############################
x_train, y_train, scenario_train = create_dataset(list_training, look_back, input_variables, ponto, descricao)
x_test, y_test, scenario_test = create_dataset(list_test, look_back, input_variables, ponto, descricao)

cfg_list = model_configs()
print(cfg_list)
t1 = int(round(time.time()))
for i in cfg_list:
    np.random.seed(1)  # reproducibility
    neurons = i[0];
    init_mode = i[1];
    l_1 = i[2];
    l_2 = i[3];
    do = i[4]
    weight_decay = float(1e-4)
    save_path = address + '\\Resultados\\LSTM{}-lb{}-n{}-init_{}-do{}'.format(lstm, look_back, neurons, init_mode, do)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Definindo a rede
    regressor = Sequential()
    # 1st layer
    regressor.add(MonteCarloLSTM(units=neurons, input_shape=(look_back, ni), kernel_initializer=init_mode,
                       kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay), dropout=do, recurrent_dropout=do))
    # Adding the output layer
    regressor.add(Dense(units=no, activation='softmax'))
    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # simple early stopping and model checkpoint
    es = EarlyStopping(monitor='loss', mode='auto', verbose=0, patience=patience, min_delta=0.01)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='auto', verbose=0, save_best_only=True)
    # Fitting the RNN to the training set
    history = regressor.fit(x_train, y_train, epochs=en,  validation_split=vs, batch_size=bs, verbose=1, callbacks=[es, mc])
    j1, w1 = np.shape(y_train)
    j2, w2 = np.shape(y_test)
    i_samples = 50
    train_predict, test_predict = np.zeros((i_samples, j1, w1)), np.zeros((i_samples, j2, w2))
    for j in range(i_samples):
        train_predict_aux = regressor.predict(x_train, batch_size=None, verbose=0)
        regressor.reset_states()
        test_predict_aux = regressor.predict(x_test, batch_size=None, verbose=0)
        train_predict[j][:][:] = train_predict_aux
        test_predict[j][:][:] = test_predict_aux
    media_treino = np.mean(train_predict, axis=0)
    desvio_padrao_treino = np.std(train_predict, axis=0)
    media_test = np.mean(test_predict, axis=0)
    desvio_padrao_test = np.std(test_predict, axis=0)

    ############### Save RNN (Weights and biases) #####################
    # serialize model to JSON
    model_json = regressor.to_json()
    with open(save_path+"\\model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    regressor.save_weights(save_path+"\\model.h5")
    print(i)

    label_int_aux_train = np.amax(media_treino, axis=1, keepdims=True).tolist()
    label_int_train = media_treino
    label_int_aux_test = np.amax(media_test, axis=1, keepdims=True).tolist()
    label_int_test = media_test
    ############## Plots ##############
    log_treino = log_loss(y_train, media_treino)
    d = np.argmax(media_treino, axis = -1)
    d0 = np.argmax(y_train, axis = -1)
    precisao_treino = accuracy_score(encoder_ponto.inverse_transform(d0), encoder_ponto.inverse_transform(d))
    metricas = metrics.classification_report(encoder_ponto.inverse_transform(d0), encoder_ponto.inverse_transform(d), digits=3, output_dict=True)
    metricas = pd.DataFrame(metricas.items())
    metricas.to_excel(save_path+'\\precisao_treinamento.xlsx')
    # print("Accuracy = %f" % precisao_treino)
    # print("Categorical crossentropy = %f" % log_treino)
    target_names = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'NL']
    cm = confusion_matrix(encoder_ponto.inverse_transform(d0), encoder_ponto.inverse_transform(d))
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, cmap="Greens")
    ax.set_yticklabels(target_names, va='center', rotation=90)
    plt.ylabel('Actual source')
    plt.xlabel('Predicted source')
    plt.tight_layout()
    plt.savefig(save_path+'\\treinamento.png', dpi=600)

    log_teste = log_loss(y_test, media_test)
    d = np.argmax(media_test, axis = -1)
    d0 = np.argmax(y_test, axis = -1)
    precisao_teste = accuracy_score(encoder_ponto.inverse_transform(d0), encoder_ponto.inverse_transform(d))
    metricas = metrics.classification_report(encoder_ponto.inverse_transform(d0), encoder_ponto.inverse_transform(d), digits=3, output_dict=True)
    metricas = pd.DataFrame(metricas.items())
    metricas.to_excel(save_path+'\\metricas_teste.xlsx')
    # print("Accuracy = %f" % precisao_teste)
    # print("Categorical crossentropy = %f" % log_teste)
    cm = confusion_matrix(encoder_ponto.inverse_transform(d0), encoder_ponto.inverse_transform(d))
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, cmap="Greens")
    ax.set_yticklabels(target_names, va='center')
    plt.ylabel('Actual source')
    plt.xlabel('Predicted source')
    plt.tight_layout()
    plt.savefig(save_path+'\\teste.png', dpi=600)

    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'], color='red')
    plt.plot(history.history['val_loss'], color='blue')
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    # plt.title('Custo do modelo')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Test loss'], loc='upper right')
    plt.grid('True')
    plt.tight_layout()
    plt.savefig(save_path+'\\loss.png', dpi=600)
    ############### Save data #########
    df1 = pd.DataFrame(np.concatenate((media_treino, desvio_padrao_treino, scenario_train), axis=1))
    df1.columns = ['Mean A1', 'Mean A2', 'Mean A3', 'Mean A4', 'Mean B1', 'Mean B2', 'Mean B3', 'Mean B4'
        , 'Mean NL', 'dev A1', 'dev A2', 'dev A3', 'dev A4', 'dev B1', 'dev B2', 'dev B3', 'dev B4', 'dev NL', 'scenario', 'descricao']
    df1.to_excel(save_path+'\\dados_calculados_treinamento.xlsx', sheet_name='Treinamento')
    df2 = pd.DataFrame(np.concatenate((media_test, desvio_padrao_test, scenario_test), axis=1))
    df2.columns = ['Mean A1', 'Mean A2', 'Mean A3', 'Mean A4', 'Mean B1', 'Mean B2', 'Mean B3', 'Mean B4'
        , 'Mean NL', 'dev A1', 'dev A2', 'dev A3', 'dev A4', 'dev B1', 'dev B2', 'dev B3', 'dev B4', 'dev NL', 'scenario', 'descricao']
    df2.to_excel(save_path+'\\dados_calculados_teste.xlsx', sheet_name='Teste')
    df = pd.DataFrame(np.array([lstm, neurons, look_back, init_mode, do, log_treino, precisao_treino, log_teste,
                                precisao_teste, regressor.count_params(), len(history.history['loss'])]).reshape(1,-1))
    reader = pd.read_excel(address+'\\Resultados_Final.xlsx')
    writer = pd.ExcelWriter(address+'\\Resultados_Final.xlsx', engine='openpyxl')
    writer.book = load_workbook(address+'\\Resultados_Final.xlsx')
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    df.to_excel(writer, index=False, header=False, startrow=len(reader)+1)
    writer.close()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=[15, 15])
    for i in range(len(target_names)):
        fpr, tpr, _ = roc_curve(y_test[:, i], media_test[:, i])
        roc_auc = auc(fpr, tpr)
        k = target_names[i]
        plt.plot(fpr, tpr, lw = 3, label='{} (AUC = %0.3f)'.format(k) % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(save_path + '\\ROC_test.png', dpi=600)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=[15, 15])
    for i in range(len(target_names)):
        fpr, tpr, _ = roc_curve(y_train[:, i], media_treino[:, i])
        roc_auc = auc(fpr, tpr)
        k = target_names[i]
        plt.plot(fpr, tpr, lw = 3, label='{} (AUC = %0.3f)'.format(k) % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(save_path + '\\ROC_train.png', dpi=600)

tf = int(round(time.time())) - t1
print('O processo de busca durou %f segundos' % tf)
freq = 2500
duration = 1000
winsound.Beep(freq, duration)
