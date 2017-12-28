"""
World Bank Group Pension Department
Author:  Natan Goldberger
email: ngoldberger@worldbank.org
"""

import data_management
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.stats import linregress
from functools import reduce


class AnalysisTools:
    def __init__(self):
        self.x = 1

    @staticmethod
    def monthly_return(portfolio_level):
        m_returns = portfolio_level.pct_change()[1:]
        return m_returns

    @staticmethod
    def monthly_excess_return(portfolio_performance, benchmark_performance, single=True):
        if single:
            monthly_xs = portfolio_performance.sub(benchmark_performance[benchmark_performance.columns[0]], axis=0)
        else:
            monthly_xs = portfolio_performance.sub(benchmark_performance, axis=0)
        return monthly_xs

    def month_to_date_excess_return(self, portfolio_performance, benchmark_performance, single=True):
        mtd = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single).tail(1)
        mtd.index = ['Excess Return: MTD']
        return mtd

    def best_worst_period_return(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        data = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        roll_data = self.rolling_excess_returns(portfolio_performance, benchmark_performance)
        best = pd.DataFrame(data=[data[-per:].max().values for per in periods],
                            columns=data.columns, index=['Best: ' + str(per) for per in periods])
        worst = pd.DataFrame(data=[data[-per:].min().values for per in periods],
                             columns=data.columns, index=['Worst: ' + str(per) for per in periods])
        average_gain = pd.DataFrame(data=[data[-per:][data > 0].mean().values for per in periods],
                                    columns=data.columns, index=['Average Gain: ' + str(per) for per in periods])
        average_loss = pd.DataFrame(data=[data[-per:][data < 0].mean().values for per in periods],
                                    columns=data.columns, index=['Average Loss: ' + str(per) for per in periods])
        best_roll = pd.DataFrame(data=[roll_data[i].max().values * 100 for i in range(0, len(roll_data))],
                                 columns=roll_data[0].columns, index=['Best Run-up: 12', 'Best Run-up: 36'])
        worst_roll = pd.DataFrame(data=[roll_data[i].min().values * 100 for i in range(0, len(roll_data))],
                                  columns=roll_data[0].columns, index=['Worst Drawdown: 12', 'Worst Drawdown: 36'])
        hit_ratio = pd.DataFrame(data=[data[-per:][data > 0].count().values / data[-per:].count().values
                                       for per in periods], columns=data.columns,
                                 index=['Hit Ratio: ' + str(per) for per in periods])
        table = pd.concat([best, worst, average_gain, average_loss, hit_ratio, best_roll, worst_roll])
        return table

    @staticmethod
    def rolling_returns(portfolio_performance, window=12):
        rolling_portfolio = portfolio_performance.rolling(window)
        rolling = rolling_portfolio.apply(lambda x: np.prod(1 + x / 100) ** (12 / window) - 1)
        return rolling

    def rolling_excess_returns(self, portfolio_performance, benchmark_performance, windows=None, single=True):
        if windows is None:
            windows = [12, 36]
        roll = []
        if single:
            benchmark_name = benchmark_performance.columns[0]
            for window in windows:
                rolling_excess = self.rolling_returns(portfolio_performance, window=window).sub(
                    self.rolling_returns(benchmark_performance, window=window)[benchmark_name], axis=0)
                roll.append(rolling_excess)
        else:
            for window in windows:
                rolling_excess = self.rolling_returns(portfolio_performance, window=window).sub(
                    self.rolling_returns(benchmark_performance, window=window), axis=0)
                roll.append(rolling_excess)
        return roll

    def rolling_tracking_error(self, portfolio_performance, benchmark_performance, window=12, single=True):
        xs_return = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        rolling_te = xs_return.rolling(window)
        rolling = rolling_te.apply(lambda x: np.std(x) * np.sqrt(12) / 100)
        return rolling

    def best_worst_annual_return(self, portfolio_performance, benchmark_performance, single=True):
        data = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        data['Year'] = data.index.year
        annual_data = data.pivot(columns='Year')
        best = annual_data.max().unstack(level=0)
        best.index = ['Best: ' + str(year) for year in best.index]
        worst = annual_data.min().unstack(level=0)
        worst.index = ['Worst: ' + str(year) for year in worst.index]
        average_gain = annual_data[annual_data > 0].mean().unstack(level=0)
        average_gain.index = ['Average gain: ' + str(year) for year in average_gain.index]
        average_loss = annual_data[annual_data < 0].mean().unstack(level=0)
        average_loss.index = ['Average loss: ' + str(year) for year in average_loss.index]
        hit_ratio = (annual_data[annual_data > 0].count() / annual_data.count()).unstack(level=0)
        hit_ratio.index = ['Hit Ratio: ' + str(year) for year in hit_ratio.index]
        table = pd.concat([best, worst, average_gain, average_loss, hit_ratio])
        return table

    @staticmethod
    def annual_return(performance):
        performance['Year'] = performance.index.year
        annual_data = performance.pivot(columns='Year')
        YTD = (((1 + annual_data / 100).prod() - 1) * 100).unstack(level=0)
        performance.drop('Year', axis=1, inplace=True)
        return YTD

    def annual_excess_return(self, portfolio_performance, benchmark_performance, single=True):
        if single:
            annual_xs = (self.annual_return(portfolio_performance)).sub(self.annual_return(benchmark_performance)
                                                                        [benchmark_performance.columns[0]], axis=0)
            annual_xs.index = ['Excess Return: ' + str(annual_xs.index[i]) for i in range(0, len(annual_xs.index))]
        else:
            annual_xs = self.annual_return(portfolio_performance) - self.annual_return(benchmark_performance)
            annual_xs.index = ['Excess Return: ' + str(annual_xs.index[i]) for i in range(0, len(annual_xs.index))]
        return annual_xs

    @staticmethod
    def period_return(performance, periods=None):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        returns = [(((1 + performance[-period - 1:] / 100).prod()) ** (12 / period) - 1) * 100 for period in periods]
        period_returns = pd.DataFrame.from_records(returns, index=periods)
        return period_returns

    def period_excess_return(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        if single:
            period_xs = (self.period_return(portfolio_performance,
                                            periods=periods)).sub(self.period_return(benchmark_performance,
                                                                                     periods=periods)
                                                                  [benchmark_performance.columns[0]], axis=0)
        else:
            period_xs = (self.period_return(portfolio_performance,
                                            periods=periods)).sub(self.period_return(benchmark_performance,
                                                                                     periods=periods), axis=0)
        period_xs.index = ['Excess Return: ' + str(period_xs.index[i]) for i in range(0, len(period_xs.index))]
        return period_xs

    def tracking_error(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        xs_return = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        tracking_err = [((xs_return[-per:] / 100).std()) * (12 ** (1 / 2)) * 100 for per in periods]
        te = pd.DataFrame.from_records(tracking_err, index=periods)
        te.index = ['Tracking Error: ' + str(te.index[i]) for i in range(0, len(te.index))]
        return te

    def downside_tracking_error(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        xs_return = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        neg_xs_return = xs_return[xs_return < 0]
        tracking_err = [((neg_xs_return[-per:] / 100).std()) * (12 ** (1 / 2)) * 100 for per in periods]
        ds_te = pd.DataFrame.from_records(tracking_err, index=periods)
        ds_te.index = ['Downside Tracking Error: ' + str(ds_te.index[i]) for i in range(0, len(ds_te.index))]
        return ds_te

    def upside_tracking_error(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        xs_return = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        pos_xs_return = xs_return[xs_return > 0]
        tracking_err = [((pos_xs_return[-per:] / 100).std()) * (12 ** (1 / 2)) * 100 for per in periods]
        up_te = pd.DataFrame.from_records(tracking_err, index=periods)
        up_te.index = ['Upside Tracking Error: ' + str(up_te.index[i]) for i in range(0, len(up_te.index))]
        return up_te

    @staticmethod
    def moments(portfolio_performance, periods=None):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        volatility = [((portfolio_performance[-per:] / 100).std()) * (12 ** (1 / 2)) * 100 for per in periods]
        skewnewss = [((portfolio_performance[-per:] / 100).skew()) for per in periods]
        kurtosis = [((portfolio_performance[-per:] / 100).kurt()) for per in periods]
        vol = pd.DataFrame.from_records(volatility, index=periods)
        ske = pd.DataFrame.from_records(skewnewss, index=periods)
        kur = pd.DataFrame.from_records(kurtosis, index=periods)
        vol.index = ['Volatility: ' + str(vol.index[i]) for i in range(0, len(vol.index))]
        ske.index = ['Skewness: ' + str(ske.index[i]) for i in range(0, len(ske.index))]
        kur.index = ['Kurtosis: ' + str(kur.index[i]) for i in range(0, len(kur.index))]
        moments = pd.concat([vol, ske, kur])
        return moments

    @staticmethod
    def downside_volatility(portfolio_performance, periods=None):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        portfolio_performance = portfolio_performance[portfolio_performance < 0]
        volatility = [((portfolio_performance[-per:] / 100).std()) * (12 ** (1 / 2)) * 100 for per in periods]
        downside_vol = pd.DataFrame.from_records(volatility, index=periods)
        downside_vol.index = ['Volatility: ' + str(downside_vol.index[i]) for i in range(0, len(downside_vol.index))]
        return downside_vol

    @staticmethod
    def get_kde(series):
        """
        This method will get the KDE estimate based on the desired
        kernel and return the support, PDF, and CDF.
        :param series:  Variable to estimate KDE
        :return: PDF, CDF, and Support
        """
        # Estimate KDE
        estimate = sm.nonparametric.KDEUnivariate(series)
        estimate.fit(kernel='gau', fft=True)
        # Get variables
        x_axis = pd.DataFrame(estimate.support, columns=['Support'])
        density = pd.DataFrame(estimate.density, columns=['PDF'])
        cumulative_density = pd.DataFrame(density.cumsum().values / density.sum().values, columns=['CDF'])
        dat_to_return = pd.concat([x_axis, density, cumulative_density], axis=1)
        return dat_to_return

    def expected_shortfall(self, portfolio_performance, benchmark_performance, alpha=0.01, periods=None,
                           single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        data = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        expected_shortfall = dict()
        for fund in data.columns:
            expected_shortfall[fund] = dict()
            for per in periods:
                data_distribution = self.get_kde(data[fund][-per:])
                support = data_distribution['Support']
                CDF = data_distribution['CDF']
                calculation = support[CDF < alpha].mean()
                expected_shortfall[fund].update({per: calculation})
        expected_shortfall = pd.DataFrame.from_dict(expected_shortfall, orient='index').transpose()
        expected_shortfall.index = ['ES: ' + str(per) for per in periods]
        return expected_shortfall

    def omega_ratio(self, portfolio_performance, benchmark_performance, center=0, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        data = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        Omega = dict()
        for fund in data.columns:
            Omega[fund] = dict()
            for per in periods:
                data_distribution = self.get_kde(data[fund][-per:])
                support = data_distribution['Support']
                CDF = data_distribution['CDF']
                probability_down = CDF[support < center].max()
                probability_up = 1 - probability_down
                calculation = probability_up / probability_down - 1
                Omega[fund].update({per: calculation})
        Omega = pd.DataFrame.from_dict(Omega, orient='index').transpose()
        Omega.index = ['Omega: ' + str(per) for per in periods]
        return Omega

    @staticmethod
    def r_squared(portfolio_performance, indices, periods=36):
        indices = sm.add_constant(indices)
        data = pd.concat([portfolio_performance, indices], axis=1)[-periods:].dropna()
        r_sqr = pd.DataFrame(data=[sm.OLS(data[fund], data[indices.columns], missing='drop').fit().rsquared for fund in
                                   portfolio_performance.columns], index=portfolio_performance.columns,
                             columns=['R2']).transpose()
        return r_sqr

    @staticmethod
    def beta(portfolio_performance, indices, periods=36):
        data = pd.concat([portfolio_performance, indices], axis=1)[-periods:].dropna()
        beta = pd.DataFrame(data=[sm.OLS(data[fund], data[indices.columns], missing='drop').fit().params for fund in
                                  portfolio_performance.columns], index=portfolio_performance.columns).transpose()
        beta.index = ['Beta: ' + str(periods) + 'M']
        return beta

    def gini(self, portfolio_performance, periods=None):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        gin = pd.DataFrame()
        for per in periods:
            data = (portfolio_performance[-per:]).dropna()
            gi = [self.gini_coef(data[[fund]].as_matrix()) for fund in data.columns]
            gin['Gini: ' + str(per)] = gi
        gin.index = portfolio_performance.columns
        gini = gin.transpose()
        return gini

    @staticmethod
    def gini_coef(array):
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    def consecutive_periods(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120, 1000]
        exs_return = self.monthly_excess_return(portfolio_performance, benchmark_performance, single=single)
        consecutive = pd.DataFrame()
        for per in periods:
            xs_return = exs_return[-per - 1:]
            pos = xs_return > 0
            neg = xs_return < 0
            above = pos.expanding().apply(lambda r: reduce(lambda x, y: x + 1 if y else x * y, r)).max()
            beyond = neg.expanding().apply(lambda r: reduce(lambda x, y: x + 1 if y else x * y, r)).max()
            # cons = pd.concat([above, beyond], axis=1)
            consecutive['Longest Outperformance: ' + str(per)] = above
            consecutive['Longest Underperformance: ' + str(per)] = beyond
        consecutive.index = xs_return.columns
        consecutive = consecutive.transpose()
        return consecutive

    @staticmethod
    def since_invested_date(data):
        si = [data[fund].first_valid_index() for fund in data.columns]
        si_dates = pd.DataFrame(data=si, index=data.columns).transpose()
        return si_dates

    def since_invested_periods(self, data):
        si_dates = self.since_invested_date(data)
        sip = [len(data[data[fund].index > si_dates[fund][0]][fund]) for fund in data.columns]
        si_periods = pd.DataFrame(data=sip, index=data.columns, columns=['Periods']).transpose()
        return si_periods

    def since_invested_return(self, performance, periods=None):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(performance)
        all_periods = periods + ['SI']
        si_return = pd.DataFrame(columns=performance.columns, index=all_periods)
        for fund in performance:
            si_return[fund] = pd.DataFrame(
                data=self.period_return(performance[[fund]],
                                        periods=periods + [si_period[fund][0]]).values,
                columns=[fund], index=all_periods)
        return si_return

    def since_invested_excess_return(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(portfolio_performance)
        all_periods = periods + ['SI']
        idx = ['Excess Return: ' + str(per) for per in all_periods]
        si_xs_return = pd.DataFrame(columns=portfolio_performance.columns, index=idx)
        for fund in portfolio_performance:
            si_xs_return[fund] = pd.DataFrame(
                data=self.period_excess_return(portfolio_performance[[fund]], benchmark_performance[[fund]],
                                               periods=periods + [si_period[fund][0]], single=single).values,
                columns=[fund], index=idx)
        return si_xs_return

    def since_invested_te(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(portfolio_performance)
        all_periods = periods + ['SI']
        idx = ['Tracking Error: ' + str(per) for per in all_periods]
        si_te = pd.DataFrame(columns=portfolio_performance.columns, index=idx)
        for fund in portfolio_performance:
            si_te[fund] = pd.DataFrame(
                data=self.tracking_error(portfolio_performance[[fund]], benchmark_performance[[fund]],
                                         periods=periods + [si_period[fund][0]], single=single).values,
                columns=[fund], index=idx)
        return si_te

    def since_invested_up_te(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(portfolio_performance)
        all_periods = periods + ['SI']
        idx = ['Upside Tracking Error: ' + str(per) for per in all_periods]
        si_te = pd.DataFrame(columns=portfolio_performance.columns, index=idx)
        for fund in portfolio_performance:
            si_te[fund] = pd.DataFrame(
                data=self.upside_tracking_error(portfolio_performance[[fund]], benchmark_performance[[fund]],
                                                periods=periods + [si_period[fund][0]], single=single).values,
                columns=[fund], index=idx)
        return si_te

    def since_invested_down_te(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(portfolio_performance)
        all_periods = periods + ['SI']
        idx = ['Downside Tracking Error: ' + str(per) for per in all_periods]
        si_te = pd.DataFrame(columns=portfolio_performance.columns, index=idx)
        for fund in portfolio_performance:
            si_te[fund] = pd.DataFrame(
                data=self.downside_tracking_error(portfolio_performance[[fund]], benchmark_performance[[fund]],
                                                  periods=periods + [si_period[fund][0]], single=single).values,
                columns=[fund], index=idx)
        return si_te

    def since_invested_moments(self, performance, periods=None):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(performance)
        all_periods = periods + ['SI']
        vol = ['Volatility: ' + str(i) for i in all_periods]
        ske = ['Skewness: ' + str(i) for i in all_periods]
        kur = ['Kurtosis: ' + str(i) for i in all_periods]
        idx = vol + ske + kur
        si_moments = pd.DataFrame(columns=performance.columns, index=idx)
        for fund in performance:
            si_moments[fund] = pd.DataFrame(
                data=self.moments(performance[[fund]],
                                  periods=periods + [si_period[fund][0]]).values,
                columns=[fund], index=idx)
        return si_moments

    def since_invested_best_worst_period(self, portfolio_performance, benchmark_performance, periods=None, single=True):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(portfolio_performance)
        all_periods = periods + ['SI']
        best = ['Best: ' + str(i) for i in all_periods]
        worst = ['Worst: ' + str(i) for i in all_periods]
        avgg = ['Average Gain: ' + str(i) for i in all_periods]
        avgl = ['Average Loss: ' + str(i) for i in all_periods]
        hitr = ['Hit Ratio: ' + str(i) for i in all_periods]
        bestru = ['Best Run-up: 12', 'Best Run-up: 36']
        worsdd = ['Worst Drawdown: 12', 'Worst Drawdown: 36']
        idx = best + worst + avgg + avgl + hitr + bestru + worsdd
        si_bwp = pd.DataFrame(columns=portfolio_performance.columns, index=idx)
        for fund in portfolio_performance:
            si_bwp[fund] = pd.DataFrame(
                data=self.best_worst_period_return(portfolio_performance[[fund]], benchmark_performance[[fund]],
                                                   periods=periods + [si_period[fund][0]], single=single).values,
                columns=[fund], index=idx)
        return si_bwp

    def since_invested_consecutive_periods(self, portfolio_performance, benchmark_performance, periods=None, single=False):
        if periods is None:
            periods = [12, 24, 36, 48, 60, 120]
        si_period = self.since_invested_periods(portfolio_performance)
        all_periods = periods + ['SI']
        items = ['Longest Outperformance: ', 'Longest Underperformance: ']
        idx = [item + str(per) for per in all_periods for item in items]
        si_cons = pd.DataFrame(columns=portfolio_performance.columns, index=idx)
        for fund in portfolio_performance:
            si_cons[fund] = pd.DataFrame(
                data=self.consecutive_periods(portfolio_performance[[fund]], benchmark_performance[[fund]],
                                              periods=periods + [si_period[fund][0]], single=single).values,
                columns=[fund], index=idx)
        return si_cons

    def performance_attribution(self, portfolio_weights, excess_returns):
        expected_performance = []
        for port in portfolio_weights:
            w = portfolio_weights[port]
            expected_return = (w.transpose().dot(excess_returns.transpose())).transpose().mean()*12
            expected_performance.append(expected_return)
        port_performance = pd.DataFrame(data=expected_performance, index=portfolio_weights.columns)
        return port_performance

    @staticmethod
    def risk_attribution(portfolio_weights, covariance):
        r_weights = pd.DataFrame()
        total_risk = []
        covariance = covariance * 12
        for port in portfolio_weights:
            w = portfolio_weights[port]
            tot_risk = (w.transpose().dot(covariance).dot(w)) ** 0.5
            risk_attribution = w.transpose().dot(covariance) / tot_risk
            te_contribution = risk_attribution * w.transpose()
            risk_weight = te_contribution / tot_risk
            r_weights = pd.concat([r_weights, risk_weight], axis=1)
            total_risk.append(tot_risk)
        trisk = pd.DataFrame(data=total_risk, index=portfolio_weights.columns)
        return trisk, r_weights

    def single_factor_regression(self, funds, factors, window=30):
        fund_factor = [fund + " " + factor for fund in funds for factor in factors]
        exposures = pd.DataFrame(columns=fund_factor, index=['Lin Corr', 'Rank Corr', 'Beta', 'P-Val', 'R2'])
        for fund in funds:
            for factor in factors:
                x = factors[factor][-window:]
                y = funds[fund][-window:]
                z = pd.concat([x, y], axis=1).dropna()
                if len(z) < 30:
                    print('Warning: The regression of ' + factor + ' on ' + fund + " has less than 30 observations")
                regres = linregress(z)
                bet_param = regres[0]
                pval = regres[3]
                r2 = regres[4]
                lin_corr = regres[2]
                ran_corr = spearmanr(x, y, nan_policy='omit')[0]
                exposures[fund + " " + factor] = [lin_corr, ran_corr, bet_param, pval, r2]
        exposures = exposures.transpose()
        return exposures

if __name__ == '__main__':
    dc = data_management.DataConnect(path='C:\\Users\\wb514964\\Code\\em-equity\\database\\', database='EMEQ.db')
    at = AnalysisTools()
    portfolio = dc.download_data('performance', columns=['Date', 'Product_ID', 'Return'],
                                 filter_attribute=['Product_ID']
                                 , filter_values=[["'GWBPFS1N22702'", "'GWBPFS1N23802'", "'GWBPFS2NN0602'",
                                                   "'GWBPFS1N24802'", "'GWBPFS1N25002'", "'GWBPFS2NN0302'",
                                                   "'GWBPFS1N24702'", "'GWBPFS1N25202'", "'GWBPFS1N17702'"]])
    benchmark = dc.download_data('performance', columns=['Date', 'Product_ID', 'Return'],
                                 filter_attribute=['Product_ID']
                                 , filter_values=[['37']])
    weights = pd.read_excel('C:\\Users\\wb514964\\Code\\em-equity\\templates\\Risk Attribution.xlsx',
                            sheetname='Weights')
    xs = at.monthly_excess_return(portfolio, benchmark)
    total_risk, risk_weights = at.risk_attribution(weights, xs.cov())
    r2 = at.r_squared(portfolio, benchmark)
    best_worst = at.best_worst_period_return(portfolio, benchmark)
    bwest = at.best_worst_annual_return(portfolio, benchmark)
    cvar = at.expected_shortfall(portfolio, benchmark)
    omega = at.omega_ratio(portfolio, benchmark)
    annual_excess = at.annual_excess_return(portfolio, benchmark)
    period = at.period_excess_return(portfolio, benchmark)
    tracking_error = at.tracking_error(portfolio, benchmark)

    evestment = dc.download_data('performance', columns=['Date', 'Product_ID', 'Return'],
                                 filter_attribute=['Product_ID']
                                 , filter_values=[[671479, 816487, 816889, 816880, 681807,
                                                   838395, 775026, 847572, 800109]], local_names=True)
    portfolio = dc.download_data('performance', columns=['Date', 'Product_ID', 'Return'],
                                 filter_attribute=['Product_ID']
                                 , filter_values=[
            ["'GWBPFS1N22702'", "'GWBPFS1N23802'", "'GWBPFS2NN0602'", "'GWBPFS1N24802'",
             "'GWBPFS1N25002'", "'GWBPFS2NN0302'", "'GWBPFS1N24702'", "'GWBPFS1N25202'",
             "'GWBPFS1N17702'"]])
