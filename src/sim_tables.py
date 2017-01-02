#!/usr/bin/env python

import pandas as pd 
import numpy as np
import json

from xtabs2vars import *


# Here we load city data for generating students' hometowns. We assign the 
# probability of selecting a city in proportion to the size of the city. 
cities = pd.read_csv('../input_data/top_500_cities.csv')
cities['probability'] = cities.iloc[:, 2]/sum(cities.iloc[:, 2])
cities['citystate'] = cities['city'].map(str) + ', ' + cities['state']

undergrad_inst = pd.read_csv("../input_data/colleges_and_universities_subset.csv")

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



## PUBLICATIONS_TABLE:
# student_id, impact_factor, issue, volume  

## STUDENT_DEMOGRAPHICS_TABLE:
# student_id, age, sex, ethnicity, hometown, marital_status, has_children 

## STUDENTS_PRGRM_TABLE:
# student_id, program, degree, fulltime, funded_position, year_in_program

## GRAD_SCHOOL_APPLICATIONS_TABLE:
# student_id, undergrad_gpa, gre_verbal, gre_quant, gre_writing,
# undergrad_institution, research_experience

## GRANTS_TABLE:
# grant_id, student_id, amount, funding_agency

## COURSES_TABLE:
# id, student_id, course_id, final_grade


## ta_data_table:
# student_id, course_id, 

## professors_table:
# prof_id, rank, lab_size, dept, 


inverse_logit = lambda x: 1/(1 + np.exp(-x))

def shuffle_rows(df):
    '''
    This function is used to shuffle rows to in order to lessen 
    the impression of simulated data in many of the generated tables.
    '''
    n = df.shape[0]
    row_order = np.arange(n)
    np.random.shuffle(row_order)
    
    df = df.iloc[row_order, :].reset_index(drop = True)
    return df


def gen_student_ids(n):
    ids = ['SR' + str(x) for x in range(1001000, 1001000 + n)]
    np.random.shuffle(ids) 
    return ids

def gen_age(yr):
    '''This function simulates a student's age from their year_in_prgm'''
    age =  yr + 20 + np.random.choice([-1, 0, 1, 2, 3, 4, 5])
    return age


def sim_students_prgrm_table(n):
    ''' 
    This function simulates the student_table. Note we make the naïve assumption 
    that all degree programs are equally likely. 
    '''
    f = open('phd_prgms.txt', 'r')
    phd_prgms = f.readline().strip('\n').split(', ')

    f = open('masters_prgms.txt', 'r')
    masters_prgms = f.readline().strip('\n').split(', ')

    f = open('ms_prgms.txt')
    ms_prgms = f.readline().strip('\n').split(', ')

    students = pd.DataFrame()
    students['student_id'] = gen_student_ids(n)
    students['degree'] = np.random.choice(['PhD', 'MS'], n, p = [0.7, 0.3])
    students['funded_position'] = True
    students['fulltime'] = True
    
    for i in range(n):
        # PhD and masters students are hanlded quite differently, with 
        # more nuances in the masters students data
        if students.loc[i, 'degree'] == 'PhD':
            students.loc[i, 'program'] = np.random.choice(phd_prgms)
            students.loc[i, 'year_in_prgm'] = np.random.choice(range(1,8))
        else:
            students.loc[i, 'program'] = np.random.choice(masters_prgms)
            if students.loc[i, 'program'] not in ms_prgms:
                students.loc[i, 'degree'] = 'MA'
            students.loc[i, 'year_in_prgm'] = np.random.choice([1, 2, 3])
            
            # For masters students, we probabilistically assign 
            # funded and fulltime status (both are all TRUE for PhD)
            students.loc[i, 'fulltime'] = np.random.choice([True, False], p = [0.8, 0.2])
            if not students.loc[i, 'fulltime']:
                students.loc[i, 'funded_position'] = False 
            else: 
                students.loc[i, 'funded_position'] = np.random.choice([True, False], p = [0.3, 0.7])
    
    col_order = ['student_id', 'program', 'degree', 'fulltime', 'funded_position', 'year_in_prgm']
    
    students = students[col_order]
    return students

# student_prgm = sim_students_prgrm_table(1000)


def sim_student_demographics_table(student_prgm_df, cities_df):
    '''
    This function simulates student's demographic data. We use the student's 
    program data to help inform a few decisions (e.g., `year_in_prgm` and `age`).
    '''
    students = student_prgm_df.copy()
    n = students.shape[0]
    
    f = open('stem_prgms.txt')
    stem_prgms = f.readline().strip('\n').split(', ')


    # Here we create `marital_status` and `has_children` variables based on the 
    # crosstabs we specify in the dataframe below. This allows us to perfectly 
    # control the relation between these categorical variables. 
    married_children_xtab = pd.DataFrame()
    married_children_xtab['Married'] = [0.1, 0.18]
    married_children_xtab['Single'] = [0.02, 0.70]
    married_children_xtab.index = ['has children', 'no children']

    xkids, xmarried = xtabs2vars(married_children_xtab, n)
    students['has_children'] = xkids
    students['marital_status'] = xmarried 

    ethn_dat = pd.read_csv('ethnicities.csv')
    students['has_children'] = students['has_children'] == 'has children'
    students['sex'] = np.random.choice(['Male', 'Female'], n)
    students['ethnicity'] = np.random.choice(ethn_dat['ethnicity'], n, p = ethn_dat['proportion'])
    students['hometown'] = np.random.choice(cities_df['citystate'], n, p = cities_df['probability'])
    
    for i in range(n):
        students.loc[i, 'age'] = gen_age(students.loc[i, 'year_in_prgm'])
        if students.loc[i, 'program'] not in stem_prgms:
            students.loc[i, 'sex'] = np.random.choice(['Male', 'Female'])
        else:
            students.loc[i, 'sex'] = np.random.choice(['Male', 'Female'], p = [0.6, 0.4])

    col_subset = ['student_id', 'age', 'sex', 'ethnicity', 'hometown', 'marital_status', 'has_children']
    students = students[col_subset]

    return shuffle_rows(students) 

# student_dem = sim_student_demographics_table(student_prgm, cities)


def sim_grants_table(student_ids):
    n = round(len(student_ids)/4)

    agencies = ['NIH', 'NSF', 'NIMH', 'DoD']
    amounts =  [x for x in range(8000, 15000, 500)] + [15000, 20000, \
     25000, 30000, 35000, 40000]
    
    # Assign probabilities for grant amounts such 
    # that larger grants are much less probable.
    probs = [0.13, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05,   \
     0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.019, 0.015, 0.01, 0.005, 0.001]

    df_out = pd.DataFrame()
    df_out['grant_id'] = ['F' + str(x) for x in range(50100, 50100 + n)]
    df_out['student_id'] = np.random.choice(student_ids, n, replace = True)
    df_out['amount'] = np.random.choice(amounts, n, p = probs)
    df_out['agency'] = np.random.choice(agencies, n, p = [0.45, 0.35, 0.1, 0.1])
    return df_out


# grants = sim_grants_table(student_prgm['student_id'])



# This gets our course data from the JSON
def get_json_data(filename):
    with open(filename) as data_file:    
        course_data = json.load(data_file)
    return course_data

# course_data = get_json_data('../input_data/dept_courses.json')

def gen_courses(prgm, n_courses, course_data):
    if n_courses > len(course_data[prgm]):
        courses = np.random.choice(course_data[prgm], n_courses, replace = True)       
    else:
        courses = np.random.choice(course_data[prgm], n_courses, replace = False)
    return courses 

# gen_courses("Applied Mathematics", 5, course_data)


def sim_student_courses(row, course_data):
    '''
    This function generates a single student's course history using `year_in_prgm` to 
    determine the number of courses taken by the student. Grades are assigned at random.
    '''
    yrs = row.loc['year_in_prgm']
    n_courses = int(4 * np.log(5*yrs) + np.random.choice([-1, 0, 1, 2]))
    df = pd.DataFrame()

    df['student_id'] = [row.loc['student_id'] for i in range(n_courses)]
    df['course_num'] = gen_courses(row.loc['program'], n_courses, course_data)
    df['section'] = np.random.choice([1, 2, 3], n_courses, p = [0.8, 0.15, 0.05])
    df['term'] = np.random.choice([1, 2], n_courses, p = [0.6, 0.4])
    df['final_grade'] = np.random.choice(['A', 'B', 'C'], n_courses, p = [0.56, 0.30, 0.14])
    return df

# sim_student_courses(student_prgm.loc[0, :], course_data)

def sim_courses_table(student_df, course_data):
    '''
    This function simulates the course history for all students in `student_df`
    ''' 
    n = student_df.shape[0]
    df = sim_student_courses(student_df.loc[0, :], course_data)

    for i in range(1, n):
        newdat = sim_student_courses(student_df.loc[i, :], course_data)
        df = df.append(newdat, ignore_index = True, verify_integrity = True)
    n_row = df.shape[0]
    df = shuffle_rows(df)
    df['id'] = np.arange(n_row)
    col_order = ['id', 'student_id', 'course_num', 'section', 'term', 'final_grade']
    return df[col_order] 

# student_courses = sim_courses_table(student_prgm, course_data)


def gen_gre_quant(degree_prgm):
    ''' 
    This function simulates quant GRE scores, different means SDs 
    are used for STEM and non-STEM students.
    '''
    f = open('stem_prgms.txt')
    stem_prgms = f.readline().strip('\n').split(', ')
    if degree_prgm in stem_prgms:
        # mean of 2.8 and SD 0.3 give a resulting mean percentile = 94%
        eta = np.random.normal(2.8, 0.3)
        gre = inverse_logit(eta)
    else:
        # mean of 1.7 and SD 0.7 give a resulting mean percentile = 82%
        eta = np.random.normal(1.7, 0.6)
        gre = inverse_logit(eta)
    return gre 

# gen_gre_quant('Appled Mathematics')

def gen_prior_degree(degree_prgm, degree):
    f = open('stem_prgms.txt')
    stem_prgms = f.readline().strip('\n').split(', ')
    if degree == 'PhD':
        if degree_prgm in stem_prgms:
            out = np.random.choice(['MS', 'BS', 'BA'], p = [0.15, 0.6, 0.25])
        else: 
            out = np.random.choice(['MA', 'BS', 'BA'], p = [0.15, 0.1, 0.75])
    elif degree != 'PhD':
        out = np.random.choice(['BA', 'BS'], p = [0.7, 0.3])

    return out 

def gen_gre_verbal(gre_quant):
    eta = 1.7 * gre_quant + np.random.normal(0.0, 0.4)
    gre = inverse_logit(eta)
    return gre


def gen_gre_write(gre_verbal, gre_quant):
    eta = 1.5 * gre_verbal + 0.6 * gre_quant + np.random.normal(0.0, 0.7)
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


def sim_applications_table(student_df, universities):
    n = student_df.shape[0]
    df = pd.DataFrame()
    df['student_id'] = student_df['student_id']
    df['research_experience'] = np.random.choice([True, False], n, p = [0.4, 0.6])
    df['university'] = np.random.choice(universities, n)

    for i in range(n):
        df.loc[i, 'gre_quant_pct'] = gen_gre_quant(student_df.loc[i, 'program'])
        df.loc[i, 'gre_verbal_pct'] = gen_gre_verbal(df.loc[i, 'gre_quant_pct'])
        df.loc[i, 'gre_writing_pct'] = gen_gre_write(df.loc[i, 'gre_verbal_pct'], df.loc[i, 'gre_quant_pct'])
        df.loc[i, 'prior_degree'] = gen_prior_degree(student_df.loc[i, 'program'], \
                                                     student_df.loc[i, 'degree'])
        df.loc[i, 'gpa'] = gen_gpa(df.loc[i, 'gre_verbal_pct'], \
                                   df.loc[i, 'gre_quant_pct'],  \
                                   df.loc[i, 'gre_writing_pct'])
    df = shuffle_rows(df)
    df['id'] = np.arange(n)
    col_order = ['id', 'student_id', 'university', 'prior_degree', 'gre_quant_pct', \
      'gre_verbal_pct', 'gre_writing_pct', 'gpa', 'research_experience', ]
    return df[col_order]

# student_apps = sim_applications_table(student_prgm, undergrad_inst['institution_name'])
# student_apps.head()
# student_apps.describe()
# student_apps


# journal_data = get_json_data('../input_data/journals_info.json')

def get_journal_data(prgm, journal_dict):
    journals = journal_dict[prgm]
    journal, impact_factor =  np.random.choice(journals).split(", ")
    return (journal, float(impact_factor))


def gen_publications(student_prgm_row, journal_data):
    prgm = student_prgm_row.loc['program']
    n_pubs = np.random.choice(range(int(student_prgm_row.loc['year_in_prgm'])))

    pubs = pd.DataFrame()
    pubs['student_id'] = [student_prgm_row.loc['student_id'] for x in range(n_pubs)]
    pubs['journal'] = np.zeros(n_pubs, str)
    pubs['impact_factor'] = np.zeros(n_pubs)

    # Here we choose a random integer that allows us to restrict a given student's 
    # publications to all come from within a given range of the journal's issues.
    issue_range = np.random.choice(range(10, 50))

    for i in range(n_pubs):
        pubs.loc[i, 'journal'], pubs.loc[i, 'impact_factor'] = get_journal_data(prgm, journal_data)
        
        # add random issue and volume numbers
        pubs.loc[i, 'issue'] = np.random.choice(range(1, issue_range))
        pubs.loc[i, 'volume'] = np.random.choice(range(12))
    return pubs 

# gen_publications(student_prgm.loc[0, :], journal_data)

def sim_publications_table(student_prgm, journal_data):
    n = student_prgm.shape[0]
    df = gen_publications(student_prgm.loc[0, :], journal_data)
    for i in range(n):
        newdat = gen_publications(student_prgm.loc[i, :], journal_data)
        df = df.append(newdat, ignore_index = True, verify_integrity = True)

    df = shuffle_rows(df)
    df['id'] = np.arange(df.shape[0])
    col_order = ['id', 'student_id', 'journal', 'issue', 'volume', 'impact_factor'] 
    return df[col_order]

# sim_publications_table(student_prgm, journal_data)


if __name__ == '__main__':
    np.random.seed(11111)
    n_students = 1000

    course_data = get_json_data('../input_data/dept_courses.json')
    journal_data = get_json_data('../input_data/journals_info.json')

    student_prgm = sim_students_prgrm_table(n_students)
    student_dem = sim_student_demographics_table(student_prgm, cities)
    grants = sim_grants_table(student_prgm['student_id'])
    courses = sim_courses_table(student_prgm, course_data)
    apps = sim_applications_table(student_prgm, undergrad_inst['institution_name'])
    pubs = sim_publications_table(student_prgm, journal_data)

    student_prgm.to_csv('../simdata/student_prgm.csv', index = False)
    student_dem.to_csv('../simdata/student_dem.csv', index = False)
    grants.to_csv('../simdata/grants.csv', index = False)
    courses.to_csv('../simdata/courses.csv', index = False)
    apps.to_csv('../simdata/apps.csv', index = False)
    pubs.to_csv('../simdata/pubs.csv', index = False)

