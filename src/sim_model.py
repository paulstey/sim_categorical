import pandas as pd
import numpy as np

student_prgm = pd.read_csv('../simdata/student_prgm.csv')
student_dem = pd.read_csv('../simdata/student_dem.csv')
grants = pd.read_csv('../simdata/grants.csv')
courses = pd.read_csv('../simdata/courses.csv')
apps = pd.read_csv('../simdata/apps.csv')
pubs = pd.read_csv('../simdata/pubs.csv')

student_prgm.head()
student_dem.head()
grants.head()                   # needs aggregation
courses.head()                  # needs aggregation
apps.head()
pubs.head()                     # needs aggregation


pubs_grp = pubs.groupby('student_id')
pubs_count = pubs_grp['journal'].count().reset_index()
pubs_max = pubs_grp['impact_factor'].max().reset_index()
pubs_mean = pubs_grp['impact_factor'].mean().reset_index()


def recode(x, old_new_dict):
    var_type = type(list(old_new_dict.values())[0])        # get type of dictionaries value
    n = len(x)
    out = np.zeros(n, var_type)
    for i in range(n):
        out[i] = old_new_dict[x[i]]
    return out


courses['grade_num'] = recode(courses['final_grade'], {'A': 4, 'B': 3, 'C': 2})

courses_grp = courses.groupby('student_id')
courses_count = courses_grp['course_num'].count().reset_index()
courses_gpa = courses_grp['grade_num'].mean().reset_index()


grants_grp = grants.groupby('student_id')
grants_count = grants_grp['amount'].count().reset_index()
grants_mean = grants_grp['amount'].mean().reset_index()
grants_max = grants_grp['amount'].max().reset_index()
