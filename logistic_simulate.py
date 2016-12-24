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
pi = inv_logit(eta)

# Generate outcome variable
y = np.random.binomial(1, pi)






## Final dataset columns:
#
# [1] left_program, degree_program(phd, ms), dept, undergrad_gpa, year_in_prog,
# [6] classes_taken, cohort_size, age, gender, international_student, 
# [11] year_entered, num_publications, mean_impact_factor, max_impact_factor, 
# [15] n_times_ta, funded_pos, distance_to_hometown, advisor_rank, lab_size, 
# [20] is_married, has_children, undergrad_research_experience, num_grants
# [24] num_ext_grants, 


## Tables to aggregate
# students, grad_school_application, department, publications, ta_data, grants


## professors_tbl:
# prof_id, rank, lab_size, dept, 

## publications_tbl:
# student_id, impact_factor, with_advisor, 

## students_tbl:
# student_id, dept, fulltime, taken_semester_off, providence_local, 
# hometown, funded_position, year_in_prog, is_married, has_children,
# 

## grad_school_applications_tbl:
# student_id, undergrad_gpa, undergrad_institution, research_experience

## grants_tbl:
# student_id, amount, funding_agency

## ta_data_tbl:
# student_id, course_id,

## courses_tbl:
# student_id, course_id, final_grade, semester




def sim_grants_table(n, seed):
    agencies = ['NIH', 'NSF', 'NIMH', 'DoD']
    amounts =  [x for x in range(3000, 10000, 500)] + [15000, 20000, \
     25000, 30000, 35000, 40000]
    
    probs = [0.13, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05, \
     0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.019, 0.015, 0.01, 0.005, 0.001]

    df_out = pd.DataFrame()
    df_out['amount'] = np.random.choice(amounts, n, p = probs)
    df_out['agency'] = np.random.choice(agencies, n, p = [0.45, 0.35, 0.1, 0.1])

    return df_out

