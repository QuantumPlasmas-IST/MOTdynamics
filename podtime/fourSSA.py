import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from scipy.linalg import eigh as largest_eigh
import pickle
import os
from SSA import SSA


from PIL import Image
import csv
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from statsmodels.tsa.stattools import acf, pacf
#from statsmodels.tsa.stattools import periodogram
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
#from sklearn.utils import SingularSpectrumAnalysis
from pyts.decomposition import SingularSpectrumAnalysis

import sklearn
print(sklearn.__version__)
from sklearn import decomposition





# Read the CSV file
for m in range(0,15): #para todos os detunings
#m=2 #fazendo para este detuning em particualr
    data=[]
    with open('./podresults/m%d/1k/coeffs.csv'%(m), 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # read the first row as headers
        for row in reader:
            data.append([float(row[i]) for i in range(len(row))])

    modos=5 #número de modos a usar
    for i in range(0, modos):




        dataF=data[i]

        # Perform Fourier analysis
        fft = np.fft.rfft(dataF)
        freq = np.fft.rfftfreq(len(dataF))

        # Plot the Fourier spectrum
        plt.figure()
        plt.plot(freq, np.abs(fft))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/spectrum.png"%(m,i))
        plt.close()

        dataFF=np.array(dataF)
        # Perform spectrogram
        plt.figure()
        f, t, Sxx = spectrogram(dataFF, fs=1.0, nperseg=256, noverlap=128)
        plt.pcolormesh(t, f, np.log(Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/spectogram.png"%(m,i))
        plt.close()

        

        # Perform SSA protocol
        t = np.arange( 0 , 1000 , 1 )  
        plt.figure(figsize=(20,20))
        plt.xlim([0, len(t)-1])
        plt.plot(t, dataFF)
        plt.title('Original coef data')
        plt.xlabel('Frame (tempo)')
        plt.ylabel('Peso do coeficiente')
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/original.png"%(m, i))







        plt.figure()
        F_ssa_L300 = SSA(dataFF, 300)
        F_ssa_L300.plot_wcorr()
        plt.title("W-Correlation for Toy Time Series, $L=300$");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlation.png"%(m, i))
        plt.close()

        plt.figure()
        F_ssa_L300.plot_wcorr(max=49)
        plt.title("W-Correlation for Walking Time Series (Zoomed)");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlationfirst50.png"%(m, i))
        plt.close()

        plt.figure()
        F_ssa_L300.reconstruct(0).plot()
        F_ssa_L300.reconstruct(1).plot()
        F_ssa_L300.reconstruct(2).plot()
        F_ssa_L300.reconstruct(3).plot()
        F_ssa_L300.reconstruct(4).plot()
        F_ssa_L300.reconstruct(5).plot()
        F_ssa_L300.reconstruct(6).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: First Three Groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(7)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlation0_6.png"%(m, i))
        plt.close()

        plt.figure()
        F_ssa_L300.reconstruct(4).plot()
        F_ssa_L300.reconstruct(5).plot()
        #F_ssa_L300.reconstruct(6).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: First Three Groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(2)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlation45.png"%(m, i))
        plt.close()


        plt.figure()
        F_ssa_L300.reconstruct(7).plot()
        F_ssa_L300.reconstruct(8).plot()
        F_ssa_L300.reconstruct(9).plot()
        F_ssa_L300.reconstruct(10).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: First Three Groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(4)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlation7_10.png"%(m, i))
        plt.close()

        plt.figure()
        plt.figure().set_figwidth(25)
        F_ssa_L2 = SSA(dataFF, 2)
        F_ssa_L2.components_to_df().plot() #dá plot a todas as componentes da decomposição
        F_ssa_L2.orig_TS.plot(alpha=0.4) #dá plot à serie temporal original com coro esbatida (alfa)
        plt.xlabel("$t$")
        plt.ylabel(r"$\tilde{F}_i(t)$")
        plt.title(r"$L=2$ for the Toy Time Series");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow2.png"%(m, i))
        plt.close()

        plt.figure().set_figwidth(25)
        F_ssa_L5 = SSA(dataFF, 5)
        F_ssa_L5.components_to_df().plot()
        F_ssa_L5.orig_TS.plot(alpha=0.4)
        plt.xlabel("$t$")
        plt.ylabel(r"$\tilde{F}_i(t)$")
        plt.title(r"$L=5$ for the Toy Time Series");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow5.png"%(m, i))
        plt.close()

        plt.figure()
        F_ssa_L20 = SSA(dataFF, 20)
        F_ssa_L20.plot_wcorr()
        plt.title("W-Correlation for Toy Time Series, $L=20$");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow20Correlation.png"%(m, i)) #dá a correlação entre modos

        plt.figure()
        F_ssa_L20.reconstruct(0).plot()
        F_ssa_L20.reconstruct([1,2,3,4,5]).plot() #faz-se a resconstrução dos modos que, visualmente, pareceram mais correlacionados na figura anterior. Esta linha é preciso ser alterada para cada modo...
        F_ssa_L20.reconstruct(slice(6,20)).plot()
        F_ssa_L20.reconstruct(5).plot() #plota-se o elemento fronteira em separado para ver se conotribui para ambos o barulho e para a periodicidade. queremos ter um window size tal que esta separação esteja mais clara
        plt.xlabel("$t$")
        plt.ylabel(r"$\tilde{F}_i(t)$")
        plt.title("Component Groupings for Toy Time Series, $L=20$");
        plt.legend([r"$\tilde{F}_0$", 
                    r"$\tilde{F}_1+\tilde{F}_2+\tilde{F}_3$+\tilde{F}_4$\tilde{F}_5$", 
                    r"$\tilde{F}_6+ \ldots + \tilde{F}_{19}$",
                    r"$\tilde{F}_5$"]);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow20CompGroupings.png"%(m, i))
        plt.close()

        #é suposto no plot final, as combinações terem alguma periodicidade, e a outra combinação ser ruído

        plt.figure()
        F_ssa_L60 = SSA(dataFF, 60)
        F_ssa_L60.plot_wcorr()
        plt.title("W-Correlation for Toy Time Series, $L=60$");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow60Correlation.png"%(m, i))

        plt.figure()
        F_ssa_L60.reconstruct(slice(0,7)).plot()
        F_ssa_L60.reconstruct(slice(7,60)).plot()

        plt.legend([r"$\tilde{F}^{\mathrm{(signal)}}$", r"$\tilde{F}^{\mathrm{(noise)}}$"])
        plt.title("Signal and Noise Components of Toy Time Series, $L = 60$")
        plt.xlabel(r"$t$");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow60SignalNoise.png"%(m, i))
        plt.close()

        plt.figure()
        plt.figure().set_figwidth(25)
        plt.figure().set_figheight(20)
        F_ssa_L60.components_to_df(n=7).plot()
        plt.title(r"The First 7 Components of the Toy Time Series, $L=60$")
        plt.xlabel(r"$t$");
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow60First7components.png"%(m, i))
        plt.close()



        plt.figure()
        F_ssa_L300.reconstruct(slice(2,300)).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: Taking out the first 2 groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(1)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlationfirst50witout01.png"%(m, i))
        plt.close()

        #tirar apenas os primerios 2 componentes, que neste modo (modo 1 (na verdade o segundo modo, mas i=1) do detuning 2 correspondem a trend
        #)para fazer a análise fourier sem estas componentes, e ver a que frequências correspondem a trend
        plt.figure()
        F_ssa_L300.reconstruct(slice(2,300)).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: Taking out the first 2 groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(1)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlationfirst50witout01.png"%(m, i))
        plt.close()

        fft12 = np.fft.rfft(F_ssa_L300.reconstruct(slice(2,300)))
        freq12 = np.fft.rfftfreq(len(F_ssa_L300.reconstruct(slice(2,300))))


        plt.figure()
        plt.plot(freq12, np.abs(fft12))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/spectrumreb12.png"%(m, i))
        plt.close()

        ## fim do fourier

        #ver só se os modos seguintes são ruído

        plt.figure()
        F_ssa_L300.reconstruct(slice(2,8)).plot()
        #F_ssa_L300.reconstruct(3).plot()
        #F_ssa_L300.reconstruct(4).plot()
        #F_ssa_L300.reconstruct(5).plot()
        #F_ssa_L300.reconstruct(6).plot()
        #F_ssa_L300.reconstruct(12).plot()
        #F_ssa_L300.reconstruct(13).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: First Three Groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(1)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlationfirst50firstgroups2_8sum.png"%(m, i))
        plt.close()



        #fazer fourier apenas do ruído
        plt.figure()
        F_ssa_L300.reconstruct(slice(7,300)).plot()
        F_ssa_L300.orig_TS.plot(alpha=0.4)
        plt.title("Walking Time Series: Taking out the first 2 groups")
        plt.xlabel(r"$t$ (s)")
        plt.ylabel("Coeff")
        legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(1)] + ["Original TS"]
        plt.legend(legend);
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/SSAwindow300Correlationfirst50noise.png"%(m, i))
        plt.close()

        fftnoise = np.fft.rfft(F_ssa_L300.reconstruct(slice(8,300)))
        freqnoise = np.fft.rfftfreq(len(F_ssa_L300.reconstruct(slice(8,300))))


        plt.figure()
        plt.plot(freqnoise, np.abs(fftnoise))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.savefig("./podresults/m%d/1k/fourSSA/mode%d/spectrumrebnoise8.png"%(m, i))
        plt.close()

        ## fim do fourier





        def plotgroup(components):
            plt.figure()
            F_ssa_L300.reconstruct(components).plot()
            F_ssa_L300.orig_TS.plot(alpha=0.4)
            plt.title("Walking Time Series: First Three Groups")
            plt.xlabel(r"$t$ (s)")
            plt.ylabel("Coeff")
            legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(1)] + ["Original TS"]
            plt.legend(legend);
            my_string = "_".join([str(num) for num in components])

            plt.savefig(f"./podresults/m{m}/1k/fourSSA/mode{i}/SSAwindow300groups{my_string}sum.png")
            plt.close()



        plotgroup([2,3,4,5,6,7,8])
        plotgroup([2,3,5,6,8,9,10])
        plotgroup([2,3,4,5,6])
        plotgroup([2,3,4,5])
        plotgroup([2,3,4])
        plotgroup([3,4,5,6,7,8])
        plotgroup([3,4,5,6])
        plotgroup([3,6,7,8,9])
        plotgroup([4,5])
        plotgroup([4,5,6])
        plotgroup([4,5,6,7,8])
        plotgroup([4,5,6,7,8,9,10])
        plotgroup([3,6,7,8,9,10])
        plotgroup([6,7,8,9])
        plotgroup([6,7,8,9,10])
        plotgroup([6,7,8,9,10,11])
        plotgroup([7,8,9,10])
        plotgroup([7,8,9,10,11])
        plotgroup([7,8,9,10,11,12])
        plotgroup([8,9,10,11,12])
        plotgroup([8,9,10,11])
        plotgroup([8,9,10])
        plotgroup([5,6,7,8,9,10])
        plotgroup([5,6,7,8,9])
        plotgroup([5,6,7,8,9,10,11])


    





    
    
    
