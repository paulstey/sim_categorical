
import pandas as pd 
import numpy as np



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
# degree


## grad_school_applications_tbl:
# student_id, undergrad_gpa, undergrad_institution, research_experience

## grants_tbl:
# student_id, amount, funding_agency

## ta_data_tbl:
# student_id, course_id,

## courses_tbl:
# student_id, course_id, final_grade, semester


def gen_student_ids(n):
    ids = ['SN' + str(x) for x in range(1001000, 1001000 + n)]
    # np.random.shuffle(ids) 
    return ids


def sim_students_table(student_ids):
    ''' 
    This function needs to be fixed...
    '''
    n = len(student_ids)

    married_children_xtab = pd.DataFrame()
    married_children_xtab['married'] = [0.20, 0.15]
    married_children_xtab['not married'] = [0.05, 0.60]
    married_children_xtab.index = ['has children', 'no children']

    x1, x2 = xtabs2vars(married_children_xtab, n)
    married_children_df = pd.DataFrame([x1, x2]).transpose()

    return married_children_df

sim_students_table(range(10))


def sim_grants_table(student_ids, seed):
    n = len(student_ids)

    agencies = ['NIH', 'NSF', 'NIMH', 'DoD']
    amounts =  [x for x in range(3000, 10000, 500)] + [15000, 20000, \
     25000, 30000, 35000, 40000]
    
    # Assign probabilities for grant amounts such 
    # that larger grants are much less probable.
    probs = [0.13, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05,   \
     0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.019, 0.015, 0.01, 0.005, 0.001]

    df_out = pd.DataFrame()

    df_out['grant_id'] = range(n)
    df_out['student_id'] = student_ids
    df_out['amount'] = np.random.choice(amounts, n, p = probs)
    df_out['agency'] = np.random.choice(agencies, n, p = [0.45, 0.35, 0.1, 0.1])
    return df_out


sim_grants_table(100, 100, 1111)


