import mne
import time
import numpy as np
from nolitsa import dimension, delay, utils
from scipy import signal
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import h5py
import re


def getData ():
    
     # Ler o arquivo edf    
    Tk().withdraw() # mudar isso depois para aparecer a janela do tkinter
    fileNamePath = askopenfilename() # abre o explorador de arquivo para selecionar o arquivo edf a ser lido
    data = mne.io.read_raw_edf(fileNamePath)
    raw_data = data.get_data()
    
    # you can get the metadata included in the file and a list of all channels:
    global info
    info = data.info
    channels = data.ch_names
    
    # Para salvar o nome do arquivo, a partir do seu caminho
    aux = re.split('/', fileNamePath)
    fileName = aux[len(aux)-1]
    fileName = fileName[:-4]
    
    
    print (fileName)
    #print (channels)
    
    return channels, raw_data, fileName


def preprocessamento (rawData):
        
    butterFiltLowPass = signal.butter(4, 100, 'lowpass', fs=300, output='sos')
    filteredData = signal.sosfilt(butterFiltLowPass, rawData)
    butterFiltHighPass = signal.butter(4, 0.5, 'highpass', fs=300, output='sos')
    filteredData = signal.sosfilt(butterFiltHighPass, filteredData)
    
    fig, ax = plt.subplots()
    plt.plot(rawData[1:3000], 'k')
    plt.plot(filteredData[1:3000], 'g')
    plt.title('Preprocessamento: Antes vs Depois da filtragem')
    
    print('Fim do preprocessamento');
    return filteredData

def falseNearestNeighbours(analizedData,tau,mMax,j,i):
    f3Array = np.zeros((mMax,1))
    f3Array[0] = 1 
    for m in range(1, mMax):
    
        f1,f2,f3=dimension.fnn(analizedData,  [m], tau, 15, 2,maxnum=400)
        print(f3)
        f3Array[m] = f3 
        
        if f3<0.10:
            minimalM = m
            break
        else:
            minimalM = m
    fig, ax = plt.subplots()
    plt.plot(f3Array, 'g')
    plt.hlines(0.10, 0, 8, 'r')
    title = "False Nearest Neightbours arquivo: ", j ," eletrodo: ", i
    plt.title(title)
    
    return minimalM

    
    
def minimalMutualInformation(analizedData, rangeTau):
    
    #Para calcular o valor ideal de tau vamos explorar os valores de tau utilizando mutual information 
    #para explorar um alcance Range a partir do valor tauInicial
    #e o melhor valor de tao será determinado através de valor minimo da mutual
    #information obtido.
    mutualInfo = [None] * rangeTau  
    mutualInfo = delay.dmi(analizedData, rangeTau)
    
    mutualInfoValue = min(mutualInfo)
    mutualInfoIndex = np.where(mutualInfo == mutualInfoValue)
    return mutualInfoValue, mutualInfoIndex[0]
    
    print(mutualInfoIndex[0])
    

def main():
    
    start = time.time() # iniciar a contagem de tempo de execução
    
    
    # Declaração de variáveis iniciais
    
    fileQuantity = 1
    
    
    # Inicio do loop principal
    for j in range(1,fileQuantity+1): 
        
        print ('\nInicio: Arquivo ', j ,' de ', fileQuantity)
        
        channels, raw_data, fileName = getData();
        numElectrodes = len(channels)#-4 
        
        # Criação do arquivo H5 em que as características estarão salvas
        dataSaveFile = h5py.File(fileName + '.h5', 'w')
        
        for i in range(1,numElectrodes+1):
            
            # Passar o filtro passa alta e passa baixa
            embeddedData = raw_data[i,:]
            print('\nDado filtrado do eletrodo ', i,' de ', numElectrodes);
            filteredData = embeddedData #preprocessamento (embeddedData)
            
            # Calculo do False Nearest Neighbours (FNN)
            
            # Escolher os parâmetros iniciais para a busca no FNN
            tau = 25; # tau - time delay    
            mMax = 9; # Valor até o qual o programa irá procurar por Ms com FNN menor que 10%
            minimalM = falseNearestNeighbours(filteredData, tau, mMax,j,i)
            print ("Para arquivo: ", j ," eletrodo: ", i , "Minimal M: ", minimalM)
            
            # Calculo da Informação Mútua
            tauMax = 90
            mutualInfoValue, mutualInfoIndex = minimalMutualInformation(filteredData, tauMax)
            tau = mutualInfoIndex[0]
            
            # A partir dos valores de M e tau calcular o DCE dos arquivos
            delayedArray = utils.reconstruct(filteredData,minimalM,tau)
            
            # Armazenar o dado DCE e as características em um grupo do arquivo h5
            dataSaveFile.create_dataset('eletrodo_{}'.format(i) + ': ' + channels[i-1], data=delayedArray)
            dataSaveFile.attrs['minimalM'] = minimalM
            dataSaveFile.attrs['tau'] = minimalM
            #group.create_dataset('caracteristicas_eletrodo_{}'.format(i), data=[minimalM, c])
    
    
    
    end = time.time()  # finalizar a contagem de tempo de execução
    print('Tempo de Processamento:',end - start) #exibir a contagem de tempo de execução
    
    dataSaveFile.close() # Fecha o arquivo H5 com as databases geradas
    
    #Tk.window.destroy() # destroi a janela do TKinter para
    

    
if __name__ == "__main__":
    main()



