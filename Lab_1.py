from solve import lab_1
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

if __name__ == "__main__":
        table = pd.read_excel("owid-covid-latest.xlsx",index_col=0, na_values='NA')

        obj = lab_1(table)
        obj.distribution = scipy.stats.lognorm
        # obj.task4('lognorm')
        
        for key in ['total_cases', 'total_deaths', 'total_vaccinations', 'total_tests']:

                obj.dt = obj.dataset[key]
                # obj.task2(obj.dt, True)
                # obj.task3(obj.dt,key)
                

                # x = np.linspace(0, obj.dt.max(), len(obj.dt))
                # obj.task5(x)
                # obj.task6(x)
                obj.task7('lognorm',key)