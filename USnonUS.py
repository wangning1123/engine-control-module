#
# test
import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
import math
from collections import Counter
import codecs
import matplotlib.pyplot as plt
import seaborn as sns

# test push

# ====================
#       PARAMETERS
# ====================

# month for rolling calculation 12 or 24 or 36
rollingWindows = 36

# percentile calculation for MacroBond charts
pctile = [0.15, 0.25, 0.5, 0.75, 0.9]

# categorize into 4 groups
cquartile = 4


## for consistency charts
# start and end dates for quartile charts, first date of each month, as early as 2000/6/1 for 36 month rolling, last date 2017/6/1

quarStartMth = pd.Timestamp ("2014-01-01 00:00:00")
quarEndMth   = pd.Timestamp ("2017-06-01 00:00:00")

epsilon = 0.00001

## calculate distribution year

distYr = pd.Timestamp ("2014-01-01 00:00:00")


## vintage year
vinYrList = [pd.Timestamp ("2014-01-01 00:00:00"),pd.Timestamp ("1996-01-01 00:00:00"), pd.Timestamp ("2007-01-01 00:00:00")]


# ===================================
#           READ IN DATA
# ===================================

# file location

#InputPath = 'M:\\PID\\Public Equity\\Research\\Universe research\\US-Non-US vs Global managers\\Input'
#OutputPath = 'M:\\PID\\Public Equity\\Research\\Universe research\\US-Non-US vs Global managers\\Output'
#FuncTool = 'M:\\PID\\Public Equity\\Research\\Codes\\FuncTool'

InputPath = r'C:\Users\wb522266\Desktop\NingW\Project\Universe research\Input'
OutputPath = r'C:\Users\wb522266\Desktop\NingW\Project\Universe research\Output'
FuncTool = r'C:\Users\wb522266\Desktop\NingW\Project\Universe research\FuncTool'

ManagersFile_us    = "%s%s" % (InputPath, '\\US Core.xlsx')
QualitativeFile_us = "%s%s" % (InputPath, '\\qualitative-US.xlsx')

ManagersFile_nonus    = "%s%s" % (InputPath, '\\EAFE Core - June.xlsx')
QualitativeFile_nonus = "%s%s" % (InputPath, '\\qualitative-eafe.xlsx')

# change directory
os.chdir(InputPath)
sys.path.append(FuncTool)

# import self-defined functions
from calc_tool import *



# import US
Raw_us            = pd.read_excel(ManagersFile_us, sheetname='Main Data', index_col=0, na_values='---', header=4)
Rawqualitative_us = pd.read_excel(QualitativeFile_us, sheetname='General Info', index_col=0, na_values='---')

# import non-US
Raw_nonus            = pd.read_excel(ManagersFile_nonus, sheetname='General Info', index_col=0, na_values='---', header=4)
Rawqualitative_nonus = pd.read_excel(QualitativeFile_nonus, sheetname='General Info', index_col=0, na_values='---')


# clean NA, remove funds with less than 24/36 months data

Raw_us_36    = Raw_us.dropna(thresh=36, axis=1)
Raw_nonus_24 = Raw_nonus.dropna(thresh=24, axis=1)

# df_glob_36 = df_glob_raw.dropna(thresh=36, axis=1)


# setup type

mgrtype   = list(set(Rawqualitative_us['managertype']))
benchtype = [mgrtype[mgrtype.index('Benchmark')]]
mgrtype.remove(mgrtype[mgrtype.index('Benchmark')])

captype_nonus = list(set(Rawqualitative_nonus['Cap']))
captype_us    = list(set(Rawqualitative_us['Cap']))

# remove small, mid funds in Non-US rawdata

nonus_smallcaps = Rawqualitative_nonus.loc[Rawqualitative_nonus['Cap'].isin(['SC', 'Small-Mid Cap'])]
Raw_nonus_24.drop(list(nonus_smallcaps.index), axis=1, inplace = True, errors= 'ignore')

captype_nonus.remove(captype_nonus[captype_nonus.index('SC')])
captype_nonus.remove(captype_nonus[captype_nonus.index('Small-Mid Cap')])


# # US and Non-US list by manager type
# fundmgrs_list_us  = set(Rawqualitative_us[Rawqualitative_us['managertype'] == 'Fundamental'].index)
# quantmgrs_list_us = list(set(Rawqualitative_us[Rawqualitative_us['managertype'] == 'Quantitative'].index))
# combmgrs_list_us  = list(set(Rawqualitative_us[Rawqualitative_us['managertype'] == 'Combined'].index))

bench_list_us = list(set(Rawqualitative_us[Rawqualitative_us['managertype'] == 'Benchmark'].index))
benchmark_us  = Raw_us_36[bench_list_us]


# fundmgrs_list_nonus  = set(Rawqualitative_nonus[Rawqualitative_nonus['managertype'] == 'Fundamental'].index)
# quantmgrs_list_nonus = list(set(Rawqualitative_nonus[Rawqualitative_nonus['managertype'] == 'Quantitative'].index))
# combmgrs_list_nonus  = list(set(Rawqualitative_nonus[Rawqualitative_nonus['managertype'] == 'Combined'].index))

bench_list_nonus = list(set(Rawqualitative_nonus[Rawqualitative_nonus['managertype'] == 'Benchmark'].index))
benchmark_nonus  = Raw_nonus_24[bench_list_nonus]


# delete columns that dont have latest data (remove funds already closed
for company_name in Raw_nonus_24.columns:
    if pd.isnull (Raw_nonus_24[company_name][-1]):
        del Raw_nonus_24[company_name]

# remove bmk type and data
captype_nonus.remove(captype_nonus[captype_nonus.index('Benchmark')])
captype_us.remove(captype_us[captype_us.index('Benchmark')])

Raw_us_36.drop(bench_list_us,inplace=True,axis=1)
Raw_nonus_24.drop(bench_list_nonus,inplace=True,axis=1)

df_us = Raw_us_36
df_nonus = Raw_nonus_24

    # ===================================
    #           COMBINE US & Non-US
    # ===================================


#create fund ID and fund names
fundID_csv = 'combUSnonUS_Name_CSV.csv'

if not os.path.exists(fundID_csv):
    fund_ID = pd.DataFrame(columns=['combinedName','USname','nonUSname','fundID','USid','NonUSid'])

    for idx, us_name in enumerate(df_us.columns):
        #print("Generating fundID for usFund: {0}".format(us_name))
        for idy, nonus_name in enumerate(df_nonus.columns):
            appendDict = {
                'combinedName': us_name + "x" + nonus_name,
                'USname': us_name,
                'nonUSname': nonus_name,
                'fundID': "US{0}-NonUS{1}".format(idx, idy),
                'USid':idx,
                'NonUSid':idy
            }
            fund_ID = fund_ID.append(appendDict, ignore_index=True)

    fund_ID.to_csv(fundID_csv)


sys.exit(0)


# calculated alpha assume 50/50 weight
# COMB_US_NONUS_FILE = r'result_small.pickle'
TYPE_FILE = r'typeUSnonUS.pickle'

if (not os.path.exists(COMB_US_NONUS_FILE)) or (not os.path.exists(TYPE_FILE)):
    new_index = df_us.index.copy()
    df_combUSnonUS = pd.DataFrame(index=new_index, columns=column_list_name)
    typeUSnonUS = dict()

    for idx, us_name in enumerate(df_us.columns):

        for idy, nonus_name in enumerate(df_nonus.columns):

            combined_name = us_name + "x" + nonus_name
            typeUSnonUS[combined_name] = "US-" + Rawqualitative_us ['managertype'][us_name] + "x Non_US-" + Rawqualitative_nonus ['managertype'][nonus_name]


            for cdate in df_us.index:
                df_combUSnonUS[combined_name][cdate] = (df_us[us_name][cdate] + df_nonus[nonus_name][cdate]) / 2.0

    df_combUSnonUS.to_pickle(COMB_US_NONUS_FILE)
    ftype = open(TYPE_FILE, 'wb')
    pickle.dump(typeUSnonUS, ftype)
    ftype.close()
else:
    ###============
    ## read in data
    ###============
    # df_combUSnonUS = pd.read_pickle (r'M:\PID\Public Equity\Research\Universe research\US-Non-US vs Global managers\Input\result_small.pickle')
    # ftype = open(r'M:\PID\Public Equity\Research\Universe research\US-Non-US vs Global managers\Input\typeUSnonUS.pickle', 'rb')
    df_combUSnonUS = pd.read_pickle(COMB_US_NONUS_FILE)
    ftype = open(TYPE_FILE, 'rb')
    print('Loading content from file ... ')
    #df_combUSnonUS = pd.read_pickle(r'result_small.pickle')

# cut small columns to eliminate running time
#     df_combUSnonUS = df_combUSnonUS[df_combUSnonUS.columns[0:5]]

    # ftype = open(r'typeUSnonUS.pickle', 'rb')
    typeUSnonUS = pickle.load(ftype)
    ftype.close()
    print('Read Data Done.')


# create summary table
TYPE_COUNTER_FILE = 'typeCounter.pickle'
if not os.path.exists(TYPE_COUNTER_FILE):
    typeCounter = dict()
    for cname in typeUSnonUS:
        currType = typeUSnonUS[cname]
        if currType not in typeCounter:
            typeCounter[currType] = 0
        typeCounter[currType] += 1
    ftc = open(TYPE_COUNTER_FILE, 'wb')
    pickle.dump(typeCounter, ftc)
    ftc.close()
else:
    ftc = open(TYPE_COUNTER_FILE, 'rb')
    typeCounter = pickle.load(ftc)
    ftc.close()
for key in typeCounter:
    print("{0}\t{1}".format(key, typeCounter[key]))


    # ===============================================================
    #   CALCULATE ROLLING ALPHA, TRACKING ERROR, INFORMATION RATIO
    # ===============================================================

df_bmk = pd.concat([benchmark_nonus,benchmark_us], axis = 1)
df_combBmk = pd.DataFrame( data = df_bmk.mean(axis=1), columns = ['USnonUSavgBmk'])


COMBINED_FILE = 'combinedDict_rolling_{0}.pickle'.format(rollingWindows)
if not os.path.exists(COMBINED_FILE):
    combinedDict = dict()
    print("Generating combinedDict ... ")
    for i in df_combUSnonUS.columns:
        df_input = pd.concat([df_combUSnonUS[i], df_combBmk], axis=1).dropna()


        output = {
            'name':'',
            'type':'',
            'excRet':[],
            'TE':[],
            'IR':[]
        }

        output['type'] = typeUSnonUS[i]
        output['name'] = i

        # calculate RollingReturn, TE, and IR

        excRet, TE, IR = calcRollingExcTEIR(df_input, rollingWindows)
        output['excRet'] = excRet
        output['TE'] = TE
        output['IR'] = IR
        combinedDict[i] = output

    foutput = open(COMBINED_FILE, 'wb')
    pickle.dump(combinedDict, foutput)
    foutput.close()
else:
    print('Loading combinedDict from file ... ')
    fcmb = open(COMBINED_FILE, 'rb')
    combinedDict = pickle.load(fcmb)
    fcmb.close()


    print('Calculate Alpha, TE and IR Done.')


# =============================================
#           CALCULATE PERCENTILE
# =============================================



def calcPercentile(combinedDict, target, cdate, pctile, ctype=None):
    if ctype is None:
        tmpDict = { target: [combinedDict[cname][target][cdate] for cname in combinedDict if not np.isnan(combinedDict[cname][target][cdate])] }
    else:
        tmpDict = { target: [combinedDict[cname][target][cdate] for cname in combinedDict if (not np.isnan(combinedDict[cname][target][cdate]) and combinedDict[cname]['type'] == ctype)] }
    tdf = pd.DataFrame(tmpDict)
    return tdf.quantile(pctile)



PERCENTILE_FILE = 'percentile_rolling_{0}.pickle'.format(rollingWindows)
if not os.path.exists(PERCENTILE_FILE):
    percentileDict = dict()
    for ctype in ['All', 'US-Quantitativex Non_US-Quantitative', 'US-Combinedx Non_US-Combined', 'US-Fundamentalx Non_US-Fundamental']:
        percentileDict[ctype] = dict()
        for target in ['excRet', 'TE', 'IR']:
            print(u"Processing {0} {1} ... ".format(ctype, target))
            percentileDict[ctype][target] = dict()
            for cdate in df_combUSnonUS.index:
                if ctype == 'All':
                    ctypeActual = None
                else:
                    ctypeActual = ctype
                currQuantile = calcPercentile(combinedDict, target, cdate, pctile, ctype=ctypeActual)
                percentileDict[ctype][target][cdate] = currQuantile


    fperc = open(PERCENTILE_FILE, 'wb')
    pickle.dump(percentileDict, fperc)
    fperc.close()
else:
    fperc = open(PERCENTILE_FILE, 'rb')
    percentileDict = pickle.load(fperc)
    fperc.close()

# Write CSV file
for ctype in ['All', 'US-Quantitativex Non_US-Quantitative', 'US-Combinedx Non_US-Combined', 'US-Fundamentalx Non_US-Fundamental']:
    for target in ['excRet', 'TE', 'IR']:
        resultDF = pd.DataFrame(index=df_combUSnonUS.index, columns=pctile)
        for cdate in df_combUSnonUS.index:
            resultDF.loc[cdate] = percentileDict[ctype][target][cdate].transpose().values[0]
        outCSV = os.path.join(OutputPath, "out_{0}_{1}_rolling_{2}.csv".format(ctype, target, rollingWindows))
        resultDF.to_csv(outCSV)



# ============================================
#            CONSISTENCY
# ============================================

## define calculation of quartile


# rank all from top to down and divide into 4(=quartile) groups
def rankQuartile(combinedDict, columns, target, cdate, cquartile, ctype=None):
    # take out NaN
    tmpDict = dict()
    if ctype is None:
        for qname in combinedDict:
            if not np.isnan(combinedDict[qname][target][qdate]):
                tmpDict[qname] = combinedDict[qname][target][cdate]

    else:
        for qname in combinedDict:
            if not np.isnan(combinedDict[qname][target][qdate]) and combinedDict[qname]['type'] == ctype:
                tmpDict[qname] = combinedDict[qname][target][cdate]

    # change to list
    rankList = [(x,tmpDict[x]) for x in tmpDict]

    # rank based on tmpDict[x]
    if target == 'TE':
        rankList.sort(key = lambda y: y[1], reverse= False)
    else:
        rankList.sort(key=lambda y: y[1], reverse= True)

    # calc

    span = len(tmpDict) / (cquartile)

    quarterDict = dict()
    for i in range(len(rankList)):
        quarterDict[rankList[i][0]] = math.floor((i+1-epsilon)/span) + 1
    rankDf = pd.DataFrame(data=quarterDict, columns=columns, index = [cdate])

        # rankDf = pd.DataFrame( data=[x[1] for x in quarterList], columns=[x[0] for x in quarterList], index = [cdate])

    return rankDf

## main loop script

QUARTILE_FILE = '{0}_quartile_rolling_{1}.pickle'.format(cquartile,rollingWindows)
if not os.path.exists(QUARTILE_FILE):
    columns = df_combUSnonUS.columns.tolist()
    quartileResult = dict()
    for target in ['excRet', 'TE', 'IR']:
        print(u"Processing {0} quartile calculation rolling_{1}months ... ".format(target, rollingWindows))
        finalDF = pd.DataFrame(columns=columns)
        for qdate in df_combUSnonUS.index:
            currQuantile = rankQuartile(combinedDict, columns, target, qdate, cquartile, ctype = None)
            # quartileDict[target][qdate] = currQuantile
            finalDF = finalDF.append(currQuantile)
        quartileResult[target] = finalDF
    fout = open(QUARTILE_FILE, 'wb')
    pickle.dump(quartileResult, fout)
    fout.close()
else:
    print("Loading from {0} ... ".format(QUARTILE_FILE))
    fin = open(QUARTILE_FILE, 'rb')
    quartileResult = pickle.load(fin)
    fin.close()

# present in CSV
for target in ['excRet', 'TE', 'IR']:
    quartileResult_csv = '{0}_quartile_result_{1}_{2}.csv'.format(cquartile, target, rollingWindows)
    quartileResult[target].to_csv(quartileResult_csv)


# # Sanity Check
# row = quartileResult['IR'].loc[pd.Timestamp('2010-07-01 00:00:00')]
#
# tcount = pd.value_counts(row.values)
# print (tcount)

# calculate quartile change



for target in ['excRet', 'TE', 'IR']:
    print("Start quartile changes for {0} ...".format(target))

    # start quartile 1,2,3,4, not larger than cquartlile
    for startQuar in range(1, cquartile+1):

        quartile_FILE_csv = '{0}_{1}quartile{2}_month{3}-{4}_CSV.csv'.format(target, cquartile, startQuar,quarStartMth.year, quarStartMth.month)
        if not os.path.exists(quartile_FILE_csv):

            startRow = quartileResult[target].loc[quarStartMth]
            selectFund = startRow.where(startRow ==startQuar).dropna().index.tolist()

            selectDate = quartileResult[target][quartileResult[target].index >= quarStartMth]
            selectDate = selectDate[selectDate.index <= quarEndMth]

            quarResultDf = pd.DataFrame(columns = list(range(1,cquartile+1)))

            for pdate in selectDate.index:
                tmpCount = quartileResult[target][selectFund].loc[pdate].value_counts()
                tmpPer = tmpCount / tmpCount.sum()
                # print ('number of funds at {0} is {1}'.format(pdate, tmpCount.sum()))

                # tmpPer = tmpCount / quartileResult[target][selectFund].loc[pdate].dropna().sum()
                quarResultDf = quarResultDf.append(tmpPer)


           #quartile_FILE_pickle = '{0}_quartile{1}_month{2}.pickle'.format(target, startQuar,quarStartMth)

            #quarResultDf.to_pickle(quartile_FILE_pickle)
            quarResultDf.to_csv(quartile_FILE_csv)

# =================================
#   QUARTILE PERCENTAGE TABLE
# =================================

# count number of times for a fund in quartile 1 and 2

for target in ['excRet', 'TE', 'IR']:
    quartResult_FILE_pickle = 'quartPercResult_{0}_rolling_{1}.pickle'.format(target, rollingWindows)

    if not os.path.exists(quartResult_FILE_pickle):

        quartCount = quartileResult[target].apply(pd.value_counts).fillna(0)
        quartPerc = (quartCount/quartCount.sum()).T
        quartPerc['1OR2'] = quartPerc.loc[:,1]+ quartPerc.loc[:,2]
        quartPerc['3OR4'] = quartPerc.loc[:,3]+ quartPerc.loc[:,4]
        quartPerc['Number of month'] = (quartCount.sum()).T

    # combined with fund type

        typeDf = (pd.DataFrame([typeUSnonUS])).T
        typeDf.columns = (['fundType'])

        quartResultTab = pd.concat([typeDf, quartPerc], axis = 1)


    # write to CSV and Pickle
        quartResult_FILE_csv = 'quartPercTab_{0}_CSV.csv'.format(target)

        quartResultTab.to_pickle(quartResult_FILE_pickle)
        quartResultTab.to_csv(quartResult_FILE_csv)
    else:
        print("Loading from {0} ...".format(quartResult_FILE_pickle))
        quartResultTab = pd.read_pickle(quartResult_FILE_pickle)





# =====================================
#   QUARTILE PERCENTAGE DISTRIBUTION
# =====================================

# count number of times for a fund in quartile 1 and 2

for target in ['excRet', 'TE', 'IR']:
    quartResultDist_FILE_pickle = '{0}_quartPercResultDist_{1}_since_{2}yr_rolling_{3}.pickle'.format(cquartile, target, distYr.year, rollingWindows)
    print('Calculating Quartile Percentage Distribution_{0}_since{1}yr_rolling_{2}month'.format(target, distYr.year, rollingWindows))

    if not os.path.exists(quartResultDist_FILE_pickle):

        tmpFundQuart = quartileResult[target][quartileResult[target].index >= distYr]

        startRow = tmpFundQuart.loc[distYr]
        selectFund = startRow.where(startRow == 1).dropna().index.tolist()


        quartCount = tmpFundQuart[selectFund].apply(pd.value_counts).fillna(0)
        quartPerc = (quartCount/quartCount.sum()).T.dropna(axis=0, how='all')
        quartPerc['1OR2'] = quartPerc.loc[:,1]+ quartPerc.loc[:,2]

        if cquartile>= 4:
            quartPerc['3OR4'] = quartPerc.loc[:,3]+ quartPerc.loc[:,4]

        quartPerc['Number of month'] = (quartCount.sum()).T

    # combined with fund type

        typeDf = (pd.DataFrame([typeUSnonUS])).T
        typeDf.columns = (['fundType'])

        quartResultTabDist = pd.concat([typeDf, quartPerc], axis = 1)


    # write to CSV and Pickle
        quartResultDist_FILE_csv = '{0}_quartPercTabDist_{1}_since{2}yr_CSV.csv'.format(cquartile,target,distYr.year)

        quartResultTabDist.to_pickle(quartResultDist_FILE_pickle)
        quartResultTabDist.to_csv(quartResultDist_FILE_csv)

    else:
        print("Loading from {0} ...".format(quartResultDist_FILE_pickle))
        quartResultTab = pd.read_pickle(quartResultDist_FILE_pickle)



# =================================
#   DRAW CHART FOR SPECIFIC FUND
# =================================

drawQuartile = [0, 0.25, 0.5, 0.75, 1]
DrawQUARTILE_FILE = 'draw_percentile_rolling_{0}.pickle'.format(rollingWindows)

if not os.path.exists(DrawQUARTILE_FILE):
    drawQuartileDict = dict()
    for ctype in ['All']:  # , 'US-Quantitativex Non_US-Quantitative', 'US-Combinedx Non_US-Combined','US-Fundamentalx Non_US-Fundamental'
        drawQuartileDict[ctype] = dict()
        for target in ['excRet']:
            drawQuartileDict[ctype][target] = dict()
            for cdate in df_combUSnonUS.index:
                if ctype == 'All':
                    ctypeActual = None
                else:
                    ctypeActual = ctype
                currQuantile = calcPercentile(combinedDict, target, cdate, drawQuartile, ctype=ctypeActual)
                drawQuartileDict[ctype][target][cdate] = currQuantile

                print(u'{0} {1}:'.format(cdate, target))
                    # print(currQuantile)


    fperc = open(DrawQUARTILE_FILE, 'wb')
    pickle.dump(drawQuartileDict, fperc)
    fperc.close()
else:
    fperc = open(DrawQUARTILE_FILE, 'rb')
    drawQuartileDict = pickle.load(fperc)
    fperc.close()




# calculate annualized return for this specific fund
# annualized_excess = {}
# te = {}
# ir = {}
#
# Rawmgrs_tup = tuple(Rawmgrs_list)
#
# for item in Rawmgrs_tup:
#
#     annualized_excess_temp = []
#     te_temp = []
#     ir_temp = []
#
#     for periods in [12, 36, 60, 84, 120, -1]:
#
#         if periods != -1:
#
#             truncate = Raw.tail(periods)
#             retseries = truncate[item]
#             Bench_truncate = Benchmark.tail(periods)
#         else:
#             truncate = Raw
#             retseries = truncate[item].dropna()
#             Bench_truncate = Benchmark.tail(retseries.count())
#             Bench_truncate = Bench_truncate
#
#         #            if item == 'DekaBank-Deka Global Equity SRI' and periods == 84:
#         #                item = item
#
#         if retseries.count() == periods or periods == -1:
#
#             excseries = retseries.sub(Bench_truncate[bench_list[0]], axis=0)
#
#             cumul = (1 + retseries / 100).cumprod() ** (12 / retseries.count()) - 1
#             cumul_bench = (1 + Bench_truncate / 100).cumprod() ** (12 / retseries.count()) - 1
#             cumul_last = cumul.tail(1) * 100
#             cumul_bench_last = cumul_bench.tail(1) * 100
#
#             te_temp = te_temp + [excseries.std() * pd.np.sqrt(12)]
#             annualized_excess_temp = annualized_excess_temp + [cumul_last - cumul_bench_last[bench_list[0]]]
#             ir_temp = ir_temp + [
#                 (cumul_last - cumul_bench_last[bench_list[0]]) / (excseries.std() * pd.np.sqrt(12))]
#
#             te[item] = te_temp
#             annualized_excess[item] = annualized_excess_temp
#             ir[item] = ir_temp
#         #                    annualized_excess[item].append(cumul_last-cumul_bench_last)
#         #                    ir[item].append(annualized_excess[item]/te[item])
#
#         else:
#             te_temp = te_temp + [None]
#             annualized_excess_temp = annualized_excess_temp + [None]
#             ir_temp = ir_temp + [None]
#
#             te[item] = te_temp
#             annualized_excess[item] = annualized_excess_temp
#             ir[item] = ir_temp

# Convert the data structure
drawType = 'All'
drawTarget = 'excRet'
pct_data = pd.DataFrame(columns = drawQuartile)
for ddate in df_combUSnonUS.index:
    tmpQuantile = drawQuartileDict[drawType][drawTarget][ddate].T
    tmpQuantile.index = [ddate]
    pct_data = pct_data.append( tmpQuantile )

# Start drawing

nameList = ['AllianceBernstein L.P.-AB US Core OpportunitiesxCI Institutional Asset Management, CI Investme-Black Creek International Equity Fund', \
            'Bristol Gate Capital Partners Inc-Bristol Gate US Equity StrategyxAQR Capital Management LLC-International Equity : EAFE', \
            'Columbia Management Investment Advisers, LLC-Columbia Disciplined Large CorexWellington Management Company LLP-International Quantitative Equity', \
            'Eagle Asset Management, Inc.-Large Cap Core RetailxT. Rowe Price Group, Inc.-International Concentrated Equity Strategy', \
            'Fiera Capital Corporation-Fiera Capital US EquityxCI Institutional Asset Management, CI Investme-Black Creek International Equity Fund']
for nName in nameList:

    nType = typeUSnonUS[nName]

    nRet = combinedDict[nName]['excRet']

    nRet.name = nName

    # print("nRet.name: ", nRet.name)
    #print("nRet.name: ", nRet.name)
    #print("pct_data.index: ", pct_data.index.tolist())
    #print("nRet.index: ", nRet.index.tolist())
    alternateAxis = pct_data.loc[nRet.index[0]:nRet.index[-1]]
    fundExcRet = nRet.dropna()

    sampleFund = fundExcRet.to_frame()
    sampleFund['zero'] = 0

    fig, ax = plt.subplots(figsize=[10, 7])
    ax.plot(sampleFund[nName], color='b', alpha=0.8, linewidth=2)
    ax.plot(sampleFund['zero'], color='k', alpha=0.5, linewidth=1)

            #    ax.plot(bmk_cum_ret)
            #    ax.plot(bmk_em_cum_ret)
    ax.fill_between(pct_data.index, pct_data[0.25], pct_data[0], alpha=0.35, facecolors='r')
    plt.text(pct_data.index[-1], pct_data[0.25][-1] - 0.05, '$4Q$', color='r', size=10)

    ax.fill_between(pct_data.index, pct_data[0.5], pct_data[0.25], alpha=0.2, facecolors='r')
    plt.text(pct_data.index[-1], pct_data[0.5][-1] - 0.05, '$3Q$', color='r', size=10)

    ax.fill_between(pct_data.index, pct_data[0.75], pct_data[0.5], alpha=0.2, facecolors='b')
    plt.text(pct_data.index[-1], pct_data[0.75][-1] - 0.05, '$2Q$', color='b', size=10)

    ax.fill_between(pct_data.index, pct_data[1], pct_data[0.75], alpha=0.35, facecolors='b')
    plt.text(pct_data.index[-1], pct_data[1][-1] - 0.05, '$1Q$', color='b', size=10)

    plt.ylabel("%s%s%s" % ('Rolling ', rollingWindows//12, 'yr, Excess Returns (%) - Gross of fees'), fontsize=12)

    plt.suptitle("%s%s%s%s" % (nName, '\n[' ,nType , ']'), fontsize=13)
    # plt.title("%s%.01f%s%.01f%s%.01f%s%.01f%s%.01f%s%s%s%.01f%s%.01f%s%.01f%s%.01f%s%.01f%s" % ('[ α: (1yr ', annualized_excess[name][0][0], ') (3yr ', annualized_excess[name][1][0], \
    #         ') (5yr ', annualized_excess[name][2][0], ') (10yr ', annualized_excess[name][4][0], \
    #         ') (SinInc. ', annualized_excess[name][5][0], ') ]', \
    #         '\n', '[ IR: (1yr ', ir[name][0][0], ') (3yr ', ir[name][1][0], \
    #         ') (5yr ', ir[name][2][0], ') (10yr ', ir[name][4][0], \
    #         ') (SinInc. ', ir[name][5][0], ') ]'), fontsize=12)

    ax.set_xlim([fundExcRet.index[0], fundExcRet.index[-1]])
    ax.set_ylim([alternateAxis.min().min(), alternateAxis.max().max()])

    location = "%s%s%s%s%s%s" % (OutputPath, '\\', nName,  '_rolling', rollingWindows, '.pdf')
    figure = ax.get_figure()

    figure.savefig(location)




# =============================================
#           HIGH VS LOW TRACKING
# =============================================

# split into ALL, Before2010, After2010



groupList = [ 'After 2010','Before 2010', 'All']
for group in groupList:

    highTE_FILE_pickle = 'highTE_{0}.pickle'.format(group)
    lowTE_FILE_pickle = 'lowTE_{0}.pickle'.format(group)

    if not os.path.exists(highTE_FILE_pickle):

        if group == 'After 2010':
    #       df_funds = df_combUSnonUS.getRowsBeforeLoc(pd.Timestamp('2010-01-01 00:00:00'))
            df_funds = df_combUSnonUS[df_combUSnonUS.index> pd.Timestamp('2009-12-31')]
        elif group == 'Before 2010':
            df_funds = df_combUSnonUS[df_combUSnonUS.index <= pd.Timestamp('2009-12-31')]
        else:
            df_funds = df_combUSnonUS

        highTE_FILE = dict()
        lowTE_FILE  = dict()

        print(u"Processing {0}... ".format(group))
        combinedDictTE = dict()
        for i in df_funds.columns:
            df_input = pd.concat([df_funds[i], df_combBmk], axis=1).dropna()

            combinedDictTE[i] = tr(df_input)



        # Remove NaN
        tmpTElist = [ combinedDictTE[tname] for tname in combinedDictTE if not np.isnan(combinedDictTE[tname]) ]

        # Convert to Series
        tmpDictTE = pd.Series(tmpTElist)

        # Compute median
        mediumTE = tmpDictTE.median()

        # Get fund names below or above median
        highTEnames = [ tname for tname in combinedDictTE if combinedDictTE[tname] >= mediumTE ]
        lowTEnames = [ tname for tname in combinedDictTE if combinedDictTE[tname] < mediumTE ]

        highTEfunds = df_funds[highTEnames]
        lowTEfunds = df_funds[lowTEnames]

        # highTE_FILE = tmpDictTE[tmpDictTE['tname'] > mediumTE]
        # lowTE_FILE = tmpDictTE[tmpDictTE['tname'] <= mediumTE]

        highTE_FILE_csv = 'highTE_{0}_CSV.csv'.format(group)
        lowTE_FILE_csv = 'lowTE_{0}_CSV.csv'.format(group)

        highTEfunds.to_pickle(highTE_FILE_pickle)
        lowTEfunds.to_pickle(lowTE_FILE_pickle)
        highTEfunds.to_csv(highTE_FILE_csv)
        lowTEfunds.to_csv(lowTE_FILE_csv)
    else:
        print("Loading from {0}, {1} ... ".format(highTE_FILE_pickle,lowTE_FILE_pickle ))
        highTEResult = pd.read_pickle(highTE_FILE_pickle)
        lowTEResult = pd.read_pickle(lowTE_FILE_pickle)


# ========================================================================
#     CALCULATE EXCESS RETURN TE IR SINCE VINTAGE YEAR or SINCE INCEPTION
# ========================================================================

df_combBmk = pd.DataFrame( data = df_bmk.mean(axis=1), columns = ['USnonUSavgBmk'])
df_bmk = pd.concat([benchmark_nonus,benchmark_us], axis = 1)

# df_combUSnonUS = df_combUSnonUS.iloc[:,0:5]

for vinYr in vinYrList:

    perfStat_FILE = 'perfStat_sinceYr{0}_Mth{1}.pickle'.format(vinYr.year, vinYr.month)

    if not os.path.exists(perfStat_FILE):

        print("Calculating Excess Return TE IR  ... ")


        df_vinYrfunds = df_combUSnonUS[df_combUSnonUS.index >= pd.Timestamp(vinYr)]
        print ('since vintage year_{0}'.format(vinYr.year))

        tcolumns = ['type', 'excRet', 'TE', 'IR'] #, 'VintageYr'
        df_perfStatData = pd.DataFrame(index =df_vinYrfunds.columns,  columns= tcolumns)

        totalVinYrFunds = len(df_vinYrfunds.columns)
        for idx, i in enumerate(df_vinYrfunds.columns):
            if idx % 1000 == 0 :
                print("Working on vinYrfunds {0}, {1} of {2}".format(i, idx, totalVinYrFunds))
            df_input = pd.concat([df_vinYrfunds[i], df_combBmk], axis=1).dropna()
            # print (df_input)

            df_perfStatData.loc[i,'TE']      = tr(df_input)
            df_perfStatData.loc[i, 'excRet'] = excRetAnn(df_input)
            df_perfStatData.loc[i, 'IR']     = excRetAnn(df_input)/tr(df_input)
            df_perfStatData.loc[i, 'type']   = typeUSnonUS[i]
            # df_perfStatData.loc[i, 'VintageYr'] = df_vinYrfunds.first_valid_index().year


    # write to CSV, and pickle
        calcOut = open(perfStat_FILE, 'wb')
        pickle.dump(df_perfStatData, calcOut)

        perfStat_csv = 'perfStat_yr{0}mth{1}_CSV.csv'.format(vinYr.year, vinYr.month)
        df_perfStatData.to_csv(perfStat_csv)


    else:
        print("Loading from {0} ... ".format(perfStat_FILE))
        calcIn = open(perfStat_FILE, 'rb')
        df_perfStatData = pickle.load(calcIn)
        calcIn.close()



sys.exit(0)



