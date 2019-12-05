import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import ElmanSimpleRecurrent
from pyneurgen.recurrent import JordanRecurrent
import random
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
from sklearn.metrics import mean_squared_error

# Importa os dados 
data = pd.read_csv('data_stocks.csv',index_col='Date')

#filtra os dados pelo numero de entradas vazias, caso um ativo tenha mais do que 100 entradas vazias nao sera considerado no experimento
filtered_data = data[data.columns[(data.apply(np.isnan).sum() < 100)]]

#interpolação linear das entradas em branco
interpolated_data = filtered_data.interpolate(method='linear',limit_direction='both')
data = interpolated_data

#transformação logaritmica (log)
data = data.apply(np.log)

#selecionando os dados, x contem a serie temporal do ibovespa e y as series temporais das outras acoes
x = data.iloc[:,1:]
y = data.iloc[:,0]

x = np.array(x).reshape((len(x),198))
y = np.array(y).reshape((len(y),1))


######################################################################################
####
####  Elman Neural Network
####
######################################################################################
#estrutura do modelo com numero de nos de entrada, da camada escondida e da camada de saida
input_nodes = 198
hidden_nodes = 7
output_nodes = 1

#one step ahead forecasting
learn_end_point = int(len(x)*0.95)
real_y_test = []
predicted_y_test = []
all_mse = []

#treinamento da rede neural
for i in range(learn_end_point, len(y)):
    random.seed(2016)
    x_train = x[0:i]
    x_test = x[i:i+1]
    y_train = y[0:i]
    y_test = y[i:i+1]
    y_train = np.array(y_train).reshape((len(y_train),1))
    y_test = np.array(y_test).reshape((len(y_test),1))
    #transformando os dados para estar no intervalo de 0 a 1
    scaler_x = MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    x_input = np.concatenate((x_train,x_test,np.zeros((1,np.shape(x_train)[1]))))
    y_input = np.concatenate((y_train,y_test,np.zeros((1,1))))
    #elaboracao do modelo de rede neural com os parametros definidos
    fit1 = NeuralNet()
    fit1.init_layers(input_nodes,[hidden_nodes],output_nodes,ElmanSimpleRecurrent())
    fit1.randomize_network()
    fit1.layers[1].set_activation_type('sigmoid')
    fit1.set_learnrate(0.05)
    fit1.set_all_inputs(x_input)
    fit1.set_all_targets(y_input)
    fit1.set_learn_range(0,i)
    fit1.set_test_range(i, i+1)
    fit1.learn(epochs=100, show_epoch_results = True, random_testing = False)
    mse = fit1.test()
    all_mse.append(mse)
    print("test set MSE = ", np.round(mse,6))
    target = [item[0][0] for item in fit1.test_targets_activations]
    target = scaler_y.inverse_transform(np.array(target).reshape((len(target),1)))
    pred = [item[1][0] for item in fit1.test_targets_activations]
    pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred),1)))
    real_y_test.append(target[0][0])
    predicted_y_test.append(pred[0][0])
    filehandler = open('objects/elman/el_'+str(i)+'.obj','w')
    pickle.dump(fit1,filehandler)
    filehandler.close()

#calculo do erro dos minimos quadrados
total_mse = mean_squared_error(real_y_test,predicted_y_test)
#Result total_mse = 0.00032483173558465

#salvando os dados obtidos
filehandler = open('objects/elman/el_real_y_test.obj','w')
pickle.dump(real_y_test,filehandler)
filehandler.close()
filehandler = open('objects/elman/el_predicted_y_test.obj','w')
pickle.dump(predicted_y_test,filehandler)
filehandler.close()
filehandler = open('objects/elman/el_all_mse.obj','w')
pickle.dump(all_mse,filehandler)
filehandler.close()
filehandler = open('objects/elman/el_total_mse.obj','w')
pickle.dump(total_mse,filehandler)
filehandler.close()

#criacao de graficos
predicted = np.exp(predicted_y_test)
real = np.exp(real_y_test)

plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.plot(predicted, color='red', label='valores previstos')
plt.plot(real, color='black', label='valores reais')
plt.title('Rede Neural de Elman')
plt.legend()
#figure name input nodes 198, hidden nodes 7, output nodes 1
plt.savefig('plots/elman_in_198_h_7_o_1.png')
plt.close()



######################################################################################
####
####  Jordan Neural Network
####
######################################################################################
#estrutura do modelo com numero de nos de entrada, da camada escondida e da camada de saida
input_nodes_jordan = 198
hidden_nodes_jordan = 7
output_nodes_jordan = 1
existing_weight_factor = 0.9

#one step ahead forecasting
learn_end_point = int(len(x)*0.95)
real_y_test_jordan = []
predicted_y_test_jordan = []
all_mse_jordan = []

#treinamento da rede neural de Jordan
for i in range(learn_end_point, len(y)):
    random.seed(2016)
    x_train = x[0:i]
    x_test = x[i:i+1]
    y_train = y[0:i]
    y_test = y[i:i+1]
    y_train = np.array(y_train).reshape((len(y_train),1))
    y_test = np.array(y_test).reshape((len(y_test),1))
    #transformando os dados para estar no intervalo de 0 a 1
    scaler_x = MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    x_input = np.concatenate((x_train,x_test,np.zeros((1,np.shape(x_train)[1]))))
    y_input = np.concatenate((y_train,y_test,np.zeros((1,1))))
    #elaboracao do modelo de rede neural com os parametros definidos
    fit2 = NeuralNet()
    fit2.init_layers(input_nodes_jordan,[hidden_nodes_jordan],output_nodes_jordan,JordanRecurrent(existing_weight_factor))
    fit2.randomize_network()
    fit2.layers[1].set_activation_type('sigmoid')
    fit2.set_learnrate(0.05)
    fit2.set_all_inputs(x_input)
    fit2.set_all_targets(y_input)
    fit2.set_learn_range(0,i)
    fit2.set_test_range(i, i+1)
    fit2.learn(epochs=100, show_epoch_results = True, random_testing = False)
    mse = fit2.test()
    all_mse_jordan.append(mse)
    print("test set MSE = ", np.round(mse,6))
    target = [item[0][0] for item in fit2.test_targets_activations]
    target = scaler_y.inverse_transform(np.array(target).reshape((len(target),1)))
    pred = [item[1][0] for item in fit2.test_targets_activations]
    pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred),1)))
    real_y_test_jordan.append(target[0][0])
    predicted_y_test_jordan.append(pred[0][0])
    filehandler = open('objects/jordan/jd_'+str(i)+'.obj','w')
    pickle.dump(fit2,filehandler)
    filehandler.close()

#calculo do erro dos minimos quadrados
total_mse_jordan = mean_squared_error(real_y_test_jordan,predicted_y_test_jordan)
#Result total_mse = 0.0002277697659651197

#salvando os resultados obtidos
filehandler = open('objects/jordan/jd_real_y_test.obj','w')
pickle.dump(real_y_test_jordan,filehandler)
filehandler.close()
filehandler = open('objects/jordan/jd_predicted_y_test.obj','w')
pickle.dump(predicted_y_test_jordan,filehandler)
filehandler.close()
filehandler = open('objects/jordan/jd_all_mse.obj','w')
pickle.dump(all_mse_jordan,filehandler)
filehandler.close()
filehandler = open('objects/jordan/jd_total_mse.obj','w')
pickle.dump(total_mse_jordan,filehandler)
filehandler.close()

# criacao de graficos 
predicted = np.exp(predicted_y_test_jordan)
real = np.exp(real_y_test_jordan)

plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.plot(predicted, color='red', label='valores previstos')
plt.plot(real, color='black', label='valores reais')
plt.title('Rede Neural de Jordan')
plt.legend()
#figure name input nodes 198, hidden nodes 7, output nodes 1
plt.savefig('plots/jordan_in_198_h_7_o_1.png')
plt.close()


######################################################################################
####
####  LSTM Neural Network
####
######################################################################################

#time steps = quantas unidades temporais anteriores você deseja considerar
#exemplo: se você deseja considerar na sua previsão 6 dias anteriores, entao o time_steps deve ser igual a 6
#aqui, foi utilizado time_steps=40, logo serao considerados na previsão 2 meses de dados (20 dias uteis por mes) para a previsao dos precos futuros
TIME_STEPS = 40
BATCH_SIZE = 1
lr = 0.0001000

# construcao da serie temporal  para considerar 2 meses de dados na previsao
def build_timeseries(x_before, y_before):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = x_before.shape[0] - TIME_STEPS
    dim_1 = x_before.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = x_before[i:TIME_STEPS+i]
        y[i] = y_before[TIME_STEPS+i]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

input_nodes_lstm = 198
hidden_nodes_lstm = 7
output_nodes_lstm = 1
existing_weight_factor = 0.9

#one step ahead forecasting
learn_end_point = int(len(x)*0.95)
real_y_test_lstm = []
predicted_y_test_lstm = []
all_mse_lstm = []

#treinamento da rede neural de LSTM
for i in range(learn_end_point, len(y)):
    random.seed(2016)
    x_train = x[0:i]
    x_test = x[i-40:i+1]
    y_train = y[0:i]
    y_test = y[i-40:i+1]
    y_train = np.array(y_train).reshape((len(y_train),1))
    y_test = np.array(y_test).reshape((len(y_test),1))
    #transformando os dados para estar no intervalo de 0 a 1
    scaler_x = MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    x_t, y_t = build_timeseries(x_train, y_train)
    x_temp, y_temp = build_timeseries(x_test, y_test)
    #elaboracao do modelo de rede neural com os parametros definidos
    fit3 = Sequential()
    fit3.add(LSTM(units = input_nodes_lstm, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0,recurrent_dropout=0.0,activation='sigmoid', kernel_initializer='random_uniform', recurrent_activation = 'hard_sigmoid', return_sequences=True))
    fit3.add(LSTM(hidden_nodes_lstm, return_sequences=False,activation='sigmoid'))
    fit3.add(Dense(units =1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=lr)
    fit3.compile(loss='mean_squared_error',optimizer=optimizer)
    fit3.fit(x_t,y_t,batch_size=BATCH_SIZE,epochs=100, shuffle=True)
    mse = fit3.evaluate(x_temp,y_temp,batch_size=1)
    all_mse_lstm.append(mse)
    target = scaler_y.inverse_transform(np.array(y_temp).reshape((len(y_temp),1)))
    pred = fit3.predict(x_temp)
    pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred),1)))
    real_y_test_lstm.append(target[0][0])
    predicted_y_test_lstm.append(pred[0][0])
    filehandler = open('objects/lstm/lstm_'+str(i)+'.obj','wb')
    pickle.dump(fit3,filehandler)
    filehandler.close()

#calculo do erro dos minimos quadrados
total_mse_lstm = mean_squared_error(real_y_test_lstm,predicted_y_test_lstm)

#salvando os resultados obtidos
filehandler = open('objects/lstm/lstm_real_y_test.obj','w')
pickle.dump(real_y_test_lstm,filehandler)
filehandler.close()
filehandler = open('objects/lstm/lstm_predicted_y_test.obj','w')
pickle.dump(predicted_y_test_lstm,filehandler)
filehandler.close()
filehandler = open('objects/lstm/lstm_all_mse.obj','w')
pickle.dump(all_mse_lstm,filehandler)
filehandler.close()
filehandler = open('objects/lstm/lstm_total_mse.obj','w')
pickle.dump(total_mse_lstm,filehandler)
filehandler.close()

# criacao de graficos 
predicted = np.exp(predicted_y_test_lstm)
real = np.exp(real_y_test_lstm)

plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.plot(predicted, color='red', label='valores previstos')
plt.plot(real, color='black', label='valores reais')
plt.title('Long Short Term Memory')
plt.legend()
#figure name input nodes 198, hidden nodes 7, output nodes 1
plt.savefig('plots/lstm_in_198_h_7_o_1.png')
plt.close()
