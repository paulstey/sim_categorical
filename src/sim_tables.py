
import pandas as pd 
import numpy as np

from xtabs2vars import *


PHD_PROGS = ['Anthropology', 'Applied Mathematics', 'Behavioral and Social Health Sciences',\
  'Biology', 'Biotechnology', 'Biomedical Engineering', 'Biostatistics', 'Chemistry', 'Classics'\
  'Cognitive Science', 'Comparative Literature', 'Computational Biology', 'Computer Science', \
  'Economics', 'Engineering', 'English', 'Epidemiology', 'History', 'Linguistics', 'Mathematics'\
  'Philosophy', 'Physics', 'Political Science', 'Psychology', 'Sociology']

MA_PROGS = ['Biotechnology', 'Biomedical Engineering', 'Biostatistics', 'Computer Science', \
  'Cybersecurity', 'Education', 'Engineering', 'English', 'Epidemiology', 'History', \
  'Literary Arts', 'Medical Sciences', 'Physics', 'Political Science', 'Public Affairs', \
  'Public Health', 'Social Analysis and Research', 'Theatre Arts and Performance Studies']

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
# student_id, dept, fulltime, hometown, funded_position, year_in_prog, 
# is_married, has_children, degree

## grad_school_applications_tbl:
# student_id, undergrad_gpa, undergrad_institution, research_experience

## GRANTS_TBL:
# student_id, amount, funding_agency

## ta_data_tbl:
# student_id, course_id,

## courses_tbl:
# student_id, course_id, final_grade, semester


def gen_student_ids(n):
    ids = ['SN' + str(x) for x in range(1001000, 1001000 + n)]
    # np.random.shuffle(ids) 
    return ids

def gen_age(yr):
    '''This function simulates a student's age from year_in_prog'''
    age =  yr + 20 + np.random.choice([-1, 0, 1, 2, 3, 4, 5])
    return age


def sim_students_table(student_ids, phd_progs, ma_progs):
    ''' 
    This function simulates the student_table. Note we make the naïve assumption 
    that all degree programs are equally likely.
    '''
    n = len(student_ids)

    married_children_xtab = pd.DataFrame()
    married_children_xtab['married'] = [0.1, 0.2]
    married_children_xtab['not married'] = [0.02, 0.68]
    married_children_xtab.index = ['has children', 'no children']

    xkids, xmarried = xtabs2vars(married_children_xtab, n)
    students = pd.DataFrame([xkids, xmarried]).transpose()
    
    students['sex'] = np.random.choice(["Male", "Female"], n)
    students['ethnicity'] = np.random.choice(['American Indian or Alskan Native', \
                                              'Asian', \
                                              'Black or African American', \
                                              'Hispanic or Latino', \
                                              'Native Hawaiian or Pacific Islander', \
                                              'Race/Ethnicity Unknown', \
                                              'Two or More Races', \
                                              'White'], 
                                              n, p = [0.02, 0.26, 0.11, 0.09, 0.02, 0.09, 0.02, 0.39])

    students['degree'] = np.random.choice(['PhD', 'MA/MS'], n, p = [0.7, 0.3])
    students['funded_position'] = True
    students['fulltime'] = True
    for i in range(n):
        if students.loc[i, 'degree'] == "PhD":
            students.loc[i, 'department'] = np.random.choice(phd_progs)
            students.loc[i, 'year_in_prog'] = np.random.choice([1, 2, 3, 4, 5, 6, 7])
        else:
            students.loc[i, 'department'] = np.random.choice(ma_progs)
            students.loc[i, 'year_in_prog'] = np.random.choice([1, 2, 3])
            
            # For masters students, we probabilistically assign 
            # funded and fulltime status (both are all TRUE for PhD)
            students.loc[i, 'funded_position'] = np.random.choice([True, False], p = [0.3, 0.7])
            students.loc[i, 'fulltime'] = np.random.choice([True, False], p = [0.8, 0.2])
        students.loc[i, 'age'] = gen_age(students.loc[i, 'year_in_prog'])

    return students

sim_students_table(range(20), PHD_PROGS, MA_PROGS)


def sim_grants_table(student_ids):
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


sim_grants_table([1, 2, 3, 4, 5, 6])


