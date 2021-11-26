import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as m
from sklearn.metrics import roc_auc_score, roc_curve



class lab_1:
    def __init__(self, dataset):
        self.dataset = self.task1(dataset)
        self.dt = None
        self.distribution = None
        self.MLE_params = None
        self.LS_params = None


    # subsample of whole dataset 
    def task1(self, table):
        table_europe = table[table.continent == 'Europe']
        table_europe = table_europe[['location', 'total_cases', 'total_deaths', 'total_vaccinations', 'total_tests']]
        values = {'total_cases': table_europe.total_cases.mean(), 'total_deaths': table_europe.total_deaths.mean(), 'total_vaccinations': table_europe.total_vaccinations.mean(),"total_tests": table_europe.total_tests.mean()}
        table_europe.fillna(value=values, inplace=True)
        # dt = table_europe.total_cases
        return table_europe

    # PDF estimation hist + kde
    def task2(self, x, cond): 
        sns.histplot(x, stat = 'density',bins=15,kde = cond)
        if cond:
            plt.show()

    # order statistic estimation + boxplot
    def task3(self, x,title):

        box = plt.boxplot(x)
        plt.ylabel(f'{title}')
        plt.show()
        # q_lower = box['boxes'][0].get_ydata()[1]
        # q_upper = box['boxes'][0].get_ydata()[2]
        Median = float(box['medians'][0].get_ydata()[1])
        maxValue = x.max()
        minValue = x.min()
        print(f'Max value - {maxValue}', f'Median - {Median}', f'Min value - {minValue}', sep='\n')
        print()
        plt.clf()

    # selected distibution based on nonparametric analysis 
    def task4(self, var):
        print(f'Selected distribution - {var}')

    def task5(self, x):
 

        # ax = plt.axes()
        self.task2(self.dt,False)
        self.MLE_params = self.distribution.fit(self.dt)
        # LS
        x0 = np.random.rand(len(self.MLE_params))
        r = scipy.optimize.least_squares(self.LS, x0)
        self.LS_params = r.x
        # sns.histplot(x, stat = 'density')
        plt.plot(x, self.distribution.pdf(x, *self.LS_params), label='Least Squares')

        ## MLE
        plt.plot(x, self.distribution.pdf(x, *self.MLE_params), label='MLE')
        plt.legend()
        # plt.ylim(0,1e-6)
        plt.show()


    # Estimation distribution parameters via MLE and LS methods
    def LS(self, params):

        # global dt, distribution
        percs = list(range(10, 75, 5))
        percs = np.array(percs)
        qn_first = np.percentile(self.dt, percs)
        qn_norm = self.distribution.ppf(percs/100, *params)

        return qn_first - qn_norm

    # Verification of the estimated parameters with QQ-plot
    def task6(self, x):

        # Calculation of quantiles
        percs = np.linspace(0, 99, 21)
        qn_first = np.percentile(self.dt, percs)
        qn_lognorm = self.distribution.ppf(percs/100, *self.MLE_params)

        # Building a quantile biplot
        plt.figure(figsize=(10, 10))
        plt.plot(qn_first, qn_lognorm, ls="", marker="o", markersize=6)
        plt.plot(x, x, color="k", ls="--")
        plt.xlim(np.min(self.dt), np.max(self.dt))
        plt.ylim(np.min(self.dt), np.max(self.dt))
        plt.xlabel(f'Empirical distribution for Density')
        plt.ylabel('Theoretical distribution for Density')
        plt.show()

# Stat tests. Kolmogorov-Smirnov, Chi-squared
    def task7(self, dist,title):
        self.MLE_params = self.distribution.fit(self.dt)
        ks = scipy.stats.kstest(self.dt, dist, self.MLE_params, N=100)
        chi2 = scipy.stats.chisquare(self.dt)
        print(f'Тест Колмогорова - Смирнова для {title}:',ks,sep='\n')
        print(f'Тест Пирсона для {title}:',chi2,sep='\n')
        print()


class lab_2:
    def __init__(self, dataset):
        self.dataset = self.task0(dataset)
        self.dt = None
        self.distribution = None
        self.MLE_params = None
        self.LS_params = None


    # subsample of whole dataset 
    def task0(self, table):
        table_europe = table[table.continent == 'Europe']
        table_europe = table_europe[['location', 'total_cases', 'total_deaths', 'total_vaccinations', 'total_tests', 'population']]
        values = {'total_cases': table_europe.total_cases.mean(), 'total_deaths': table_europe.total_deaths.mean(), 'total_vaccinations': table_europe.total_vaccinations.mean(),"total_tests": table_europe.total_tests.mean(), 'population': table_europe.population.mean()}
        table_europe.fillna(value=values, inplace=True)
        # dt = table_europe.total_cases
        return table_europe
    # matrix
    def matr(self):
        pd.plotting.scatter_matrix(self.dataset.loc[:, "total_cases":"population"], diagonal="kde")
        plt.tight_layout()
        plt.show()

    # PDF estimation hist + kde 
    def task1(self,x, cond): 
        sns.histplot(x, stat = 'density',bins=15,kde = cond)
        if cond:
            plt.show()

    # mean, var estimation of margin distribution
    def task2(self,title):
        print(f'Мат. ожидание для {title} - {self.dt.mean()}', f'Дисперсия для {title}- {self.dt.var()}',sep='\n')
        print()

    # nonparametric estimation of conditional distributions
    def task3(self,title):
        # test = sns.kdeplot(data = self.dataset, x ='total_tests',y= 'population')

        x = np.array(self.dt).reshape(-1,1) 
        y = np.array(self.dataset['population']) # target var

        model = LinearRegression()
        model.fit(x,y)
        k = model.coef_
        b = model.intercept_
        print(f'y={float(k)}x{b}')
        plt.plot(x,float(k)*x+b)
        plt.scatter(self.dt,self.dataset['population'])
        plt.xlabel(f'{title}')
        plt.ylabel('population')
        plt.show()
        print()

        # for path in test.collections[0].get_paths():

        #     MX, MY = path.vertices.mean(axis=0)
        #     plt.scatter(MX, MY)
        #     ex = np.sqrt(MX ** 2 + MY ** 2)
        # дисперсии
        covar = self.dataset.cov()
        print(covar)
        np.savetxt('covar.csv',covar)
        # return ex


    # defining intervals
    def r_to_z(self, r):
        return math.log((1 + r) / (1 - r)) / 2.0

    def z_to_r(self, z):
        e = math.exp(2 * z)
        return((e - 1) / (e + 1))

    def r_confidence_interval(self, r, alpha, n):
        z = self.r_to_z(r)
        se = 1.0 / math.sqrt(n - 3)
        z_crit = scipy.stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value

        lo = z - z_crit * se
        hi = z + z_crit * se

        # Return a sequence
        return (self.z_to_r(lo), self.z_to_r(hi))

    # correlation, intervals, p-value

    def task4(self,title):

        corr = scipy.stats.pearsonr(self.dt, self.dataset['population'])
        print(f'Значение p-value = {corr[1]} и correlation = {corr[0]} для population | {title}')
        print('Доверительный интервал при alpha = 0.05',self.r_confidence_interval(corr[0],0.05, len(self.dt)),sep='\n')
        print()

    # multidimensional correlation
    def task5(self):
        corrmat = self.dataset.corr()
        sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()
        plt.show()
        print('Предсказать количество вакцинированных людей от количества смертей','лучше коррелируют смерность к популяцией, что логично. Поэтому мы взяли метрики total_death & total_vaccinations',sep='\n')
        print()

    # linear regression. ROC-AUC
    def task6_7(self):
        # Highlight predictors
        X = self.dataset[['total_cases', 'total_tests', 'total_deaths']]
        # X = self.dataset['total_deaths']
        # Allocate the target variable
        y = self.dataset['total_vaccinations'].values
        y = np.array([int(i) for i in y])
        # Division into training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        x = []
        for i in range(len(y_test)):
            x.append(i)
        # Create a linear regression model
        reg = LinearRegression()
        # Train a linear regression model
        reg.fit(X_train, y_train)
        # Forecast on a test sample

        y_out = reg.predict(X_test)
        k = reg.coef_
        b = reg.intercept_
        # print(k,b,sep='\n')
        # print()
        # R2
        R2 = reg.score(X_test,y_test)
        
        vif = 1/(1-R2) # оценка на мультиколлинеарность
        print(f'Коэффициент детерминации R2: {R2}',f'Коэффициент инфляции дисперсии VIF: {vif}',sep='\n')
        print()

        print('Mean Absolute Error:', m.mean_absolute_error(y_test, y_out))
        print('Mean Squared Error:', m.mean_squared_error(y_test, y_out)) 
        print('Root Mean Squared Error:', np.sqrt(m.mean_squared_error(y_test, y_out)))
        print()
        
        balance = y_out - y_test 
        sns.histplot(balance, kde = True, stat= 'density',bins = 15)
        plt.show()
