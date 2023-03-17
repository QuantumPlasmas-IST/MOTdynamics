import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from pysindy.feature_library import CustomLibrary


def lorenz(x, t): return [10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - (8 / 3) * x[2]]


dt = 0.002
t = np.arange(0, 10, dt)
x0 = [-8, 8, 27]
X = odeint(lorenz, x0, t)

noise = np.random.normal(0, 1, X.shape)
Xnoise = X + noise


functions = [lambda x: x, lambda x, y: x * y]
function_names = [lambda x: x, lambda x, y: x + y]
costum_lib = CustomLibrary(library_functions=functions, function_names=function_names)

n_targets = 3
n_features = 9
constraint_rhs = np.asarray([0, 28])
constraint_lhs = np.zeros((2, n_targets * n_features))

constraint_lhs[0, 0] = 1
constraint_lhs[0, 1] = 1
constraint_lhs[1, 0 + n_features] = 1

opt = ps.SR3(threshold=.5, thresholder='l0')

optConstrains = ps.ConstrainedSR3(constraint_lhs=constraint_lhs, constraint_rhs=constraint_rhs, threshold=.5,
                                  thresholder='l0')

differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10})
feature_library = ps.PolynomialLibrary(degree=2, include_bias=False)
model = ps.SINDy(
    optimizer=opt,
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    feature_names=["x", "y", "z"]
)

# model = ps.SINDy()
print("Com ruido")
model.fit(Xnoise, t=dt)
# print(costum_lib.get_feature_names())
# print(model.coefficients())
model.print()

t_test = np.arange(0, 10, dt)
x0_test = np.array([-8, 8, 27])
X_test_sim = model.simulate(x0_test, t_test)

plt.plot(t, Xnoise[::, 0])
plt.plot(t, X_test_sim[::, 0])
plt.show()

plt.plot(t, Xnoise[::, 1])
plt.plot(t, X_test_sim[::, 1])
plt.show()

plt.plot(t, Xnoise[::, 2])
plt.plot(t, X_test_sim[::, 2])
plt.show()
