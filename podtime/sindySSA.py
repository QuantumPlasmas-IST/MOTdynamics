import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from scipy.linalg import eigh as largest_eigh
import pickle
import os
from SSA import SSA
import pysindy as ps
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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars


import sklearn
print(sklearn.__version__)
from sklearn import decomposition
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d


# Função que dá plot de X ponto e de X em função de lamdba, para ver o melhor threshold
def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test):
    #dt = t_test[1] - t_test[0]
    dt=1
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = 1e4
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
    plt.figure()
    plt.semilogy(threshold_scan, mse, "bo")
    plt.semilogy(threshold_scan, mse, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.figure()
    plt.semilogy(threshold_scan, mse_sim, "bo")
    plt.semilogy(threshold_scan, mse_sim, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)


#for m in range(0,14): #para todos os detunings
m=2 #fazendo para este detuning em particualr
data=[]
with open('./podresults/m%d/1k/coeffs.csv'%(m), 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # read the first row as headers
    for row in reader:
        data.append([float(row[i]) for i in range(len(row))])



modo0=data[0]
modo1=data[1]
modo2=data[2]

modo0F=np.array(modo0)
modo1F=np.array(modo1)
modo2F=np.array(modo2)

F_ssa_L300_m0 = SSA(modo0F, 300)
F_ssa_L300_m1 = SSA(modo1F, 300)
F_ssa_L300_m2 = SSA(modo2F, 300)

#apenas os modos já selecionados conforme a análise no ficheiro fourSSA.py:

mode0ssa=F_ssa_L300_m0.reconstruct([8,9,10])
mode1ssa=F_ssa_L300_m1.reconstruct([2,3,4])
mode2ssa=F_ssa_L300_m2.reconstruct([7,8,9,10])

#print(len(mode0ssa))
#print(len(mode1ssa))
#print(len(mode2ssa)) dá tudo 1000, correto

#análise sindy:


dt = 1
t = np.arange( 0 , 1000 , dt )  #array de tempos
x0=[mode0ssa[0].real, mode1ssa[0].real, mode2ssa[0].real]

#pôr os coeficientes na matriz X:
matrix=np.column_stack((mode0ssa, mode1ssa, mode2ssa))
#print(np.shape(matrix)) 1000 por 3, correto

#opt=ps.SR3(threshold=.5,thresholder='l0') 
opt=ps.STLSQ(threshold=0)
differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={'window_length': 4})
polinomial_library = ps.PolynomialLibrary ( degree=3 , include_bias=False )
identity_lib=ps.IdentityLibrary()
fourier_lib=ps.FourierLibrary()
concat_lib= identity_lib + fourier_lib #temos senos e cossenos!




#este código é para ver qual é o melhor threshold (lambda), mas de momento quase qualquer lamdba não nulo dá problemas
#threshold_scan= np.linspace(0,1,11)
#coefs=[]
#for i, threshold in enumerate(threshold_scan):
   # optscan= ps.STLSQ(threshold=threshold)
  #  modelscan=ps.SINDy(feature_names=[ "a0" , "a1" , "a2" ], optimizer=optscan, feature_library=concat_lib)
   # modelscan.fit(matrix, t=dt)
  #  coefs.append(modelscan.coefficients())

#plot_pareto(coefs, optscan, modelscan, threshold_scan, matrix, t)

model= ps.SINDy(
    optimizer=opt,
    differentiation_method=differentiation_method ,
    feature_library=fourier_lib,
    feature_names=[ "a0" , "a1" , "a2" ]

)


model.fit(matrix, t=dt)
print("SINDy Model for m=%d"%(m))
model.print()


x_test_sim= model.simulate(x0, t)

plt.figure()
plt.plot(t,matrix[::,0]) #indexação ::,0 significa que se selecionam todas as linhas, e a primeira coluna, significando a variável x
plt.plot(t,x_test_sim[::,0])
plt.savefig("./podresults/m%d/1k/sindySSA/coef0"%(m))
#plt.show()

plt.figure()
plt.plot(t,matrix[::,1])
plt.plot(t,x_test_sim[::,1])
plt.savefig("./podresults/m%d/1k/sindySSA/coef1"%(m))
#plt.show()

plt.figure()
plt.plot(t,matrix[::,2])
plt.plot(t,x_test_sim[::,2])
plt.savefig("./podresults/m%d/1k/sindySSA/coef2"%(m))
#plt.show()


#plotting coefficients in function of each other:
plt.figure()
plt.plot(mode0ssa, mode1ssa, color='blue')
# Add title and axis labels
plt.title('Coeficients of the PCA modes for $\delta$= -3.5 $\Gamma$')
plt.xlabel('Coefs First mode')
plt.ylabel('Coefs Second mode')



plt.savefig("./podresults/m%d/1k/coeffsSSA/coefs12.png"%(m))
# Show plot



#  
plt.figure()
plt.plot(mode0ssa, mode2ssa, color='blue')


# Add title and axis labels
plt.title('Coeficients of the PCA modes for $\delta$= -3.5 $\Gamma$')
plt.xlabel('Coefs First mode')
plt.ylabel('Coefs Third mode')



plt.savefig("./podresults/m%d/1k/coeffsSSA/coefs13.png"%(m))
# Show plot



#  
plt.figure()
plt.plot(mode1ssa, mode2ssa,  color='blue')


# Add title and axis labels
plt.title('Coeficients of the PCA modes for $\delta$= -3.5 $\Gamma$')
plt.xlabel('Coefs Second mode')
plt.ylabel('Coefs Third mode')



plt.savefig("./podresults/m%d/1k/coeffsSSA/coefs23.png"%(m))
# Show plot


# Create a 3D figure for -1 gamma
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter plot
ax.scatter(mode0ssa, mode1ssa, mode2ssa, c=mode0ssa, cmap='cool')

# Add labels and title
ax.set_xlabel('1st mode coef')
ax.set_ylabel('2nd mode coef')
ax.set_zlabel('3rd mode coef')
ax.set_title('Coeficientes para os primeiros 3 modos detuning = -1 gamma')

# Show the plot
plt.savefig("./podresults/m%d/1k/coeffsSSA/coefs3D.png"%(m))

