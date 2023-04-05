import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
from scipy.linalg import eigh as largest_eigh
import pickle
import os
import pysindy as ps
from PIL import Image
import csv
from mpl_toolkits.mplot3d import Axes3D
import pyspod
from pyspod.spod.standard import Standard as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils     as utils_spod
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post


n_detuning = 15
n_frames = 1000
length = 124
height = 124
frame_start=0
n_pix = length*height
n_pca_vec=100
n_new_pixels=62
nsq= n_new_pixels*n_new_pixels
parsPOD= 'parPOD.yaml'
params = utils_io.read_config (parsPOD)
timeint=1 #valor de intervalo de tempo que eu defini no ficheiro parPOD.yaml
imgs=['0_4k', '1k', '5k', '12k','20k'] #nomes de diferentes diretorias a serem criadas (para a quantidade de imagens a serem analisadas). A maior parte da análise será na com 1000 frames (1K)
pastas=['coeffs', 'fourSSA', 'modes', 'original', 'rebuilt', 'sindy', 'sindySSA', 'coeffsSSA']

for m in range(0,n_detuning):
    #criar as diretorias todas:
    for i in imgs:
        for j in pastas:
            if j=='fourSSA':
                for a in range(0,5):
                    os.makedirs(f'./podresults/m{m}/{i}/{j}/mode{a}', exist_ok=True)
            else:
                os.makedirs(f'./podresults/m{m}/{i}/{j}', exist_ok=True)



    params['savedir']="spod_results_m%d"%(m)
    t0 = time.time()
    data2 = np.zeros((n_frames,length + 20, height + 20))
    for j in range(frame_start, frame_start + n_frames):
        s_frame = '../Time_resolved_10G/Exp%d/Cam2_0_0_%d.bmp' % (m, j)
        data2[j - frame_start] = np.array(Image.open(s_frame).convert('L'))
        #data[i,j - frame_start] = data[i,j - frame_start]/(data[i,j-frame_start].sum())
    data2 = data2[:,20:144,20:144]
    # reshape the data array to have dimensions (time, space)
    #data = np.reshape(data, (n_frames, -1))

    standard=spod_standard(params=params)
    streaming=spod_streaming(params=params)
    spod=standard.fit(data_list=data2)
    results_dir = spod.savedir_sim
    flag, ortho = utils_spod.check_orthogonality(
        results_dir=results_dir, mode_idx1=[1],
        mode_idx2=[0], freq_idx=[5], dtype='single',)
    print(f'flag = {flag},  ortho = {ortho}')
   
    file_coeffs, coeffs_dir = utils_spod.compute_coeffs_op(
    data=data2, results_dir=results_dir)
    
        #plotting time series coeffs
    coeffs = np.load(file_coeffs)
    post.plot_coeffs(coeffs, coeffs_idx=[0,1,2],
        path=results_dir, filename="./podresults/m%d/1k/coeffs/"%(m))

    n_modes = 5 #número de modos usados na reconstrução

    file_dynamics, coeffs_dir = utils_spod.compute_reconstruction(
    coeffs_dir=coeffs_dir, time_idx="all", n_modes_save=n_modes)
    T1 = 50
    f1, f1_idx = spod.find_nearest_freq(freq_req=1/T1, freq=spod.freq)
    

    # plot 2d modes at frequency of 
    #esta função já faz a transformada de fourier dos coeficientes e faz o plot apenas de uma certa frequência da evolução do modo.
    spod.plot_2d_modes_at_frequency(freq_req=f1, freq=spod.freq,
        modes_idx=[0,1,2], x1=np.arange(124), x2=np.arange(124)[::-1],
        equal_axes=True, filename="./podresults/m%d/1k/modes/period%d"%(m,T1))

    print(coeffs[1,:].real)
    print(coeffs[0,0].real)
    #guardar os coefs em cvs
    import pandas as pd 
    import numpy as np
    #coefTP= coeffs.real.transpose()
    pd.DataFrame(coeffs.real).to_csv('./podresults/m%d/1k/coeffs.csv'%(m), index=False)    
 

    #função que fiz para fazer plot dos coefs um em função do outro
    post.plot_coeffspar(coeffs, coeffs_idx=[0,1], path=results_dir,filename="./podresults/m%d/1k/coeffs/"%(m))
    post.plot_coeffspar(coeffs, coeffs_idx=[1,2], path=results_dir,filename="./podresults/m%d/1k/coeffs/"%(m))
    post.plot_coeffspar(coeffs, coeffs_idx=[0,2], path=results_dir,filename="./podresults/m%d/1k/coeffs/"%(m))

    #plotting the rebuilt image
    rebuiltframes=[ 100, 200, 300,399]
    recons = np.load(file_dynamics)
    post.plot_2d_data(recons, time_idx=rebuiltframes,
        path=results_dir, x1=np.arange(124), x2=np.arange(124)[::-1], equal_axes=True,filename="./podresults/m%d/1k/rebuilt/%dmodes"%(m,n_modes))
    

    #original data:
    ## plot data
    data = spod.get_data(data2)
    post.plot_2d_data(data, time_idx=rebuiltframes,
        path=results_dir, x1=np.arange(124), x2=np.arange(124)[::-1], coastlines='centred',
        equal_axes=True,filename="./podresults/m%d/1k/original/"%(m))
    

    #Sindy protocol:
    #first, getting the coeffs a0, a1 and a2 into a matrix
    matrix= post.matrix_coeffs(coeffs, coeffs_idx=[0,1,2]) #também adicionei esta função ao pod
    opt=ps.SR3(threshold=.5,thresholder='l0') 
    differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10})
    feature_library = ps.PolynomialLibrary ( degree=2 , include_bias=False )
    model= ps.SINDy(
        optimizer=opt,
        differentiation_method=differentiation_method ,
        feature_library=feature_library ,
        feature_names=[ "a0" , "a1" , "a2" ]
    )

    dt = 0.01
    t = np.arange( 0 , 10 , dt )  #array de tempos
    model.fit(matrix, t=dt)
    print("SINDy Model for m=%d"%(m))
    model.print()
    x0=[coeffs[0,0].real, coeffs[1,0].real, coeffs[2,0].real]

    x_test_sim= model.simulate(x0, t)

    plt.figure()
    plt.plot(t,matrix[::,0]) #indexação ::,0 significa que se selecionam todas as linhas, e a primeira coluna, significando a variável x
    plt.plot(t,x_test_sim[::,0])
    plt.savefig("./podresults/m%d/1k/sindy/coef0"%(m))
    #plt.show()

    plt.figure()
    plt.plot(t,matrix[::,1])
    plt.plot(t,x_test_sim[::,1])
    plt.savefig("./podresults/m%d/1k/sindy/coef1"%(m))
    #plt.show()

    plt.figure()
    plt.plot(t,matrix[::,2])
    plt.plot(t,x_test_sim[::,2])
    plt.savefig("./podresults/m%d/1k/sindy/coef2"%(m))
    #plt.show()