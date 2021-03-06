import numpy as np

from CategoricalArrays import CategoricalArray

def inverse_logit(eta):
    pi = 1/(1 + np.exp(-eta))
    return pi

# Set up parameters
n = 2000
mu = [1.0, -0.5, 0.5]
sigma = [[1.0, 0.3, 0.0],
         [0.3, 1.0, 0.2],
         [0.0, 0.2, 0.25]]
betas = np.array([-3, 1, 2, 3])

# Simulate continuous predictors
X = np.random.multivariate_normal(mu, sigma, n)

# Confirm correlations we expect
np.corrcoef(np.transpose(X))

# Bind column of 1s for intercept coef
X = np.column_stack((np.ones(n), X))

# Compute linear predictor
eta = np.dot(X, betas)

# Compute π_i for each x_i
pi = inverse_logit(eta)

# Generate outcome variable
y = np.random.binomial(1, pi)






## Final dataset columns:
#
# left_program, degree_program(phd, ms), dept, undergrad_gpa, year_in_prog,
# classes_taken, cohort_size, age, gender, international_student, year_entered,
# num_publications, mean_impact_factor, max_impact_factor, n_times_ta, grant_funded,


## Tables to aggregate
# students, grad_school_application, department, publications, ta_data
