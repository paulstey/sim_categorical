
import pandas as pd 
import numpy as np
import json

from xtabs2vars import *

PHD_PROGS = ['Anthropology', 'Applied Mathematics', 'Behavioral and Social Health Sciences',\
  'Biology', 'Biotechnology', 'Biomedical Engineering', 'Biostatistics', 'Chemistry', 'Classics', \
  'Cognitive Science', 'Comparative Literature', 'Computational Biology', 'Computer Science', \
  'Economics', 'Engineering', 'English', 'Epidemiology', 'History', 'Linguistics', 'Mathematics', \
  'Philosophy', 'Physics', 'Political Science', 'Psychology', 'Sociology']

MASTERS_PROGS = ['Applied Mathematics', 'Biotechnology', 'Biomedical Engineering', 'Biostatistics', \
  'Behavioral and Social Health Sciences', 'Computer Science', 'Cybersecurity', 'Education', \
  'Engineering', 'English', 'Epidemiology', 'History', 'Literary Arts', 'Physics', \
  'Political Science', 'Public Health', 'Theatre Arts and Performance Studies']

MS_PROGS = ['Applied Mathematics', 'Behavioral and Social Health Sciences', 'Biotechnology', \
  'Biomedical Engineering', 'Biostatistics', 'Computer Science', 'Cybersecurity', 'Engineering', \
  'Epidemiology', 'Medical Sciences', 'Physics', 'Public Health', 'Social Analysis and Research']

STEM_PROGS = ['Applied Mathematics', 'Biotechnology', 'Biomedical Engineering', 'Biostatistics', \
  'Chemistry', 'Computational Biology', 'Computer Science', 'Economics', 'Engineering', \
  'Mathematics', 'Physics']

ETHNICITY = ['American Indian or Alskan Native', 'Asian', 'Black or African American', \
    'Hispanic or Latino', 'Native Hawaiian or Pacific Islander', 'Race/Ethnicity Unknown', \
    'Two or More Races', 'White']

ETHNICITY_PROBS = [0.02, 0.26, 0.11, 0.09, 0.02, 0.09, 0.02, 0.39]

CITIES = pd.read_csv('top_500_cities.csv')
CITIES['probability'] = CITIES.iloc[:, 2]/sum(CITIES.iloc[:, 2])
CITIES['citystate'] = CITIES['city'].map(str) + ', ' + CITIES['state']

UNDERGRAD_INST = pd.read_csv("colleges_and_universities_subset.csv")

## Final dataset columns:
#
# [1] left_program, degree_program(phd, ms), dept, undergrad_gpa, year_in_program,
# [6] classes_taken, cohort_size, age, gender, international_student, 
# [11] year_entered, num_publications, mean_impact_factor, max_impact_factor, 
# [15] n_times_ta, funded_pos, distance_to_hometown, advisor_rank, lab_size, 
# [20] is_married, has_children, undergrad_research_experience, num_grants
# [24] num_grants, 


## Tables to aggregate
# students, grad_school_application, department, publications, ta_data, grants


## professors_table:
# prof_id, rank, lab_size, dept, 

## publications_table:
# student_id, impact_factor, with_advisor, 

## STUDENTS_TABLE:
# student_id, program, degree, fulltime, funded_position, year_in_program,
# age, sex, ethnicity, hometown, marital_status, has_children 

## GRAD_SCHOOL_APPLICATIONS_TABLE:
# student_id, undergrad_gpa, gre_verbal, gre_quant, gre_writing,
# undergrad_institution, research_experience

## GRANTS_TABLE:
# grant_id, student_id, amount, funding_agency

## ta_data_table:
# student_id, course_id, 

## COURSES_TABLE:
# id, student_id, course_id, final_grade

def inverse_logit(eta):
    pi = 1/(1 + np.exp(-eta))
    return pi

def gen_student_ids(n):
    ids = ['SR' + str(x) for x in range(1001000, 1001000 + n)]
    # np.random.shuffle(ids) 
    return ids

def gen_age(yr):
    '''This function simulates a student's age from year_in_program'''
    age =  yr + 20 + np.random.choice([-1, 0, 1, 2, 3, 4, 5])
    return age


def sim_demographic_data(n, ethnicity, ethn_probs):
    married_children_xtab = pd.DataFrame()
    married_children_xtab['Married'] = [0.1, 0.18]
    married_children_xtab['Single'] = [0.02, 0.70]
    married_children_xtab.index = ['has children', 'no children']

    xkids, xmarried = xtabs2vars(married_children_xtab, n)
    students = pd.DataFrame([xkids, xmarried]).transpose()
    students.columns = ['has_children', 'marital_status']

    students['has_children'] = students['has_children'] == 'has children'
    students['sex'] = np.random.choice(['Male', 'Female'], n)
    students['ethnicity'] = np.random.choice(ethnicity, n, p = ethn_probs)
    return students 

student_dem = sim_demographic_data(200, ETHNICITY, ETHNICITY_PROBS)


def sim_students_program_table(students_demo, phd_progs, masters_progs, ms_progs, cities_df):
    ''' 
    This function simulates the student_table. Note we make the naïve assumption 
    that all degree programs are equally likely.
    '''
    students = students_demo.copy()
    n = students.shape[0]
    
    students['degree'] = np.random.choice(['PhD', 'MS'], n, p = [0.7, 0.3])
    students['funded_position'] = True
    students['fulltime'] = True
    
    for i in range(n):
        # PhD and masters students are hanlded quite differently, with 
        # more nuances in the masters students data
        if students.loc[i, 'degree'] == 'PhD':
            students.loc[i, 'program'] = np.random.choice(phd_progs)
            students.loc[i, 'year_in_program'] = np.random.choice([1, 2, 3, 4, 5, 6, 7])
        else:
            students.loc[i, 'program'] = np.random.choice(masters_progs)
            if students.loc[i, 'program'] not in ms_progs:
                students.loc[i, 'degree'] = 'MA'
            students.loc[i, 'year_in_program'] = np.random.choice([1, 2, 3])
            
            # For masters students, we probabilistically assign 
            # funded and fulltime status (both are all TRUE for PhD)
            students.loc[i, 'fulltime'] = np.random.choice([True, False], p = [0.8, 0.2])
            if not students.loc[i, 'fulltime']:
                students.loc[i, 'funded_position'] = False 
            else: 
                students.loc[i, 'funded_position'] = np.random.choice([True, False], p = [0.3, 0.7])
        students.loc[i, 'age'] = gen_age(students.loc[i, 'year_in_program'])

    students['hometown'] = np.random.choice(cities_df['citystate'], n, p = cities_df['probability'])
    students['student_id'] = gen_student_ids(n)
    col_order = ['student_id', 'program', 'degree', 'fulltime', 'funded_position', 'year_in_program', \
      'age', 'sex', 'ethnicity', 'hometown', 'marital_status', 'has_children']
    
    students = students[col_order]
    return students

s1 = sim_students_program_table(student_dem, PHD_PROGS, MASTERS_PROGS, MS_PROGS, CITIES)


def gen_grant_ids(n):
    ids = ['F' + str(x) for x in range(50100, 50100 + n)]
    # np.random.shuffle(ids) 
    return ids


def sim_grants_table(student_ids):
    n = round(len(student_ids)/4)

    agencies = ['NIH', 'NSF', 'NIMH', 'DoD']
    amounts =  [x for x in range(3000, 10000, 500)] + [15000, 20000, \
     25000, 30000, 35000, 40000]
    
    # Assign probabilities for grant amounts such 
    # that larger grants are much less probable.
    probs = [0.13, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05,   \
     0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.019, 0.015, 0.01, 0.005, 0.001]

    df_out = pd.DataFrame()
    df_out['grant_id'] = gen_grant_ids(n)
    df_out['student_id'] = np.random.choice(student_ids, n, replace = True)
    df_out['amount'] = np.random.choice(amounts, n, p = probs)
    df_out['agency'] = np.random.choice(agencies, n, p = [0.45, 0.35, 0.1, 0.1])
    return df_out


sim_grants_table(range(20))



# This gets our course data from the JSON
def get_course_data(filename):
    with open(filename) as data_file:    
        course_data = json.load(data_file)
    return course_data

course_data = get_course_data('dept_courses.json')

def gen_courses(prog, num_courses, course_data):
    courses = np.random.choice(course_data[prog], num_courses, replace = False)
    return courses 

gen_courses("Applied Mathematics", 5, course_data)


def sim_student_courses(row, course_data):
    yrs = row.loc['year_in_program']
    cnt = int(4 * np.log(5*yrs) + np.random.choice([-1, 0, 1, 2]))
    df = pd.DataFrame()

    df['student_id'] = [row.loc['student_id'] for i in range(cnt)]
    df['course_num'] = gen_courses(row.loc['program'], cnt, course_data)
    df['final_grade'] = np.random.choice(['A', 'B', 'C'], cnt, p = [0.56, 0.30, 0.14])
    return df


def sim_courses_table(student_df, course_data):
    n = student_df.shape[0]
    df = sim_student_courses(student_df.loc[0, :], course_data)

    for i in range(1, n):
        newdat = sim_student_courses(student_df.loc[i, :], course_data)
        df = df.append(newdat, ignore_index = True, verify_integrity = True)
    return df 

sim_courses_table(s1, course_data)


def gen_gre_quant(degree_prog, stem_progs):
    if degree_prog in stem_progs:
        eta = np.random.normal(3.5, 0.8)
        gre = inverse_logit(eta)
    else:
        eta = np.random.normal(2.1, 1.3)
        gre = inverse_logit(eta)
    return gre 

gen_gre_quant('Appled Mathematics', STEM_PROGS)

def gen_gre_verbal(gre_quant):
    eta = 3.5 * gre_quant + np.random.normal(0.0, 0.6)
    gre = inverse_logit(eta)
    return gre


def gen_gre_write(gre_verbal, gre_quant):
    eta = 1.5 * gre_verbal + 0.6 * gre_quant + np.random.normal(0.0, 1.2)
    gre = inverse_logit(eta)
    return gre


def gen_gpa(gre_verb, gre_quant, gre_writing):
    err = np.random.normal(0.0, 0.15)
    val = 3.3 * np.log(1.25 * gre_verb + 1.35 * gre_quant + 1.05 * gre_writing + err)
    if val > 4.0: 
        gpa = 4.0
    else:
        gpa = val
    return gpa 


def sim_applications_table(student_df, stem_progs, universities):
    n = student_df.shape[0]
    df = pd.DataFrame()
    df['student_id'] = student_df['student_id']
    df['research_experience'] = np.random.choice([True, False], n, p = [0.4, 0.6])
    df['undergrad_institution'] = np.random.choice(universities, n)

    for i in range(n):
        df.loc[i, 'gre_quant'] = gen_gre_quant(student_df.loc[i, 'program'], stem_progs)
        df.loc[i, 'gre_verbal'] = gen_gre_verbal(df.loc[i, 'gre_quant'])
        df.loc[i, 'gre_writing'] = gen_gre_write(df.loc[i, 'gre_verbal'], df.loc[i, 'gre_quant'])
        df.loc[i, 'gpa'] = gen_gpa(df.loc[i, 'gre_verbal'], \
                                   df.loc[i, 'gre_quant'],  \
                                   df.loc[i, 'gre_writing'])
    return df

sim_applications_table(s1, STEM_PROGS, UNDERGRAD_INST['institution_name'])




