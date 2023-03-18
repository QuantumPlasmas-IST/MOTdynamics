import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from pysindy.feature_library import CustomLibrary

dt = 1.0 / 200.0
X_arr = np.load('POD_coeff_array_smooth.npy')
n_samples, n_components = X_arr.shape
t = np.arange(n_samples)

opt = ps.SR3(threshold=.5, thresholder='l1')
differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 15})
feature_library = ps.PolynomialLibrary(degree=3, include_bias=False)
model = ps.SINDy(
    optimizer=opt,
    differentiation_method=differentiation_method,
    feature_library=feature_library  # ,
    #    feature_names=["a1", "a2", "a3"]
)

model.fit(X_arr, t=dt)
print(model.coefficients())
model.print()

# t_test = np.arange(0, 10, dt)
t_test = np.arange(n_samples)
x0_test = X_arr[0]
X_test_sim = model.simulate(x0_test, t_test)

plt.plot(t, X_arr[::, 0])
plt.plot(t, X_test_sim[::, 0])
plt.show()

plt.plot(t, X_arr[::, 1])
plt.plot(t, X_test_sim[::, 1])
plt.show()

plt.plot(t, X_arr[::, 2])
plt.plot(t, X_test_sim[::, 2])
plt.show()
