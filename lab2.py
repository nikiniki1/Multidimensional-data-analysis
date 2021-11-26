import pandas as pd
import numpy as np
import seaborn as sns
from solve import lab_2
table = pd.read_excel("owid-covid-latest.xlsx",index_col=0, na_values='NA')

if __name__ == "__main__":
        table = pd.read_excel("owid-covid-latest.xlsx",index_col=0, na_values='NA')


        obj = lab_2(table)
        for key in ['total_cases', 'total_deaths', 'total_vaccinations', 'total_tests']:
            obj.dt = obj.dataset[key]
            # obj.task1(obj.dt, True)
            # obj.matr()
            # obj.task2(key)
            # obj.task3(key)
            # obj.task4(key)
            # obj.task5()
        obj.task6_7()