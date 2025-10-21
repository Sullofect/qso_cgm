import statsmodels.api as sm
import numpy as np
import pandas as pd

# Simulate data
np.random.seed(42)
X = np.random.randn(1000, 3)
beta = np.array([0.5, -0.3, 0.2])
y = X @ beta + np.random.randn(1000) * 0.5

# Add intercept
X = sm.add_constant(X)

# Fit OLS
model = sm.OLS(y, X).fit()
print(model.summary())