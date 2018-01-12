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

# ===================================
#           CLEAN DATA
# ===================================

# import US
Raw_us            = pd.read_excel(ManagersFile_us, sheetname='Main Data', index_col=0, na_values='---', header=4)
Rawqualitative_us = pd.read_excel(QualitativeFile_us, sheetname='General Info', index_col=0, na_values='---')

# import non-US
Raw_nonus            = pd.read_excel(ManagersFile_nonus, sheetname='General Info', index_col=0, na_values='---', header=4)
Rawqualitative_nonus = pd.read_excel(QualitativeFile_nonus, sheetname='General Info', index_col=0, na_values='---')

# clean NA, remove funds with less than 24/36 months data
Raw_us_36    = Raw_us.dropna(thresh=36, axis=1)
Raw_nonus_24 = Raw_nonus.dropna(thresh=24, axis=1)

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

bench_list_us = list(set(Rawqualitative_us[Rawqualitative_us['managertype'] == 'Benchmark'].index))
benchmark_us  = Raw_us_36[bench_list_us]

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


# read combined US-NonUS ExcRet

print('Loading combinedDict from file ... ')
COMBINED_FILE = 'combinedDict_rolling_{0}.pickle'.format(rollingWindows)

fcmb = open(COMBINED_FILE, 'rb')
combinedDict = pickle.load(fcmb)
fcmb.close()

# ======================================
#  calc US, Non-US exc return separately
# ======================================

# US Excess Return
usRet_FILE = 'usRetDict_rolling_{0}.pickle'.format(rollingWindows)
    if not os.path.exists(usRet_FILE):
        usRetDict = dict()
        print("Calc US fund Exc Ret... ")
        for i in df_us.columns:
            df_input = pd.concat([df_us[i], benchmark_us], axis=1).dropna()

            output = {
                'name': '',
                'type': '',
                'excRet': [],
                'TE': [],
                'IR': []
            }

            output['type'] = Rawqualitative_us ['managertype'][i]
            output['name'] = i

            # calculate RollingReturn, TE, and IR

            excRet, TE, IR = calcRollingExcTEIR(df_input, rollingWindows)
            output['excRet'] = excRet
            output['TE'] = TE
            output['IR'] = IR
            usRetDict[i] = output

        foutput = open(usRet_FILE, 'wb')
        pickle.dump(usRetDict, foutput)
        foutput.close()
    else:
        print('Loading usRet from file ... ')
        fUSret = open(usRet_FILE, 'rb')
        usRetDict = pickle.load(fUSret)
        fUSret.close()

        print('Calculate US Alpha, TE and IR Done.')

# Non-US excess return
nonusRet_FILE = 'nonusRetDict_rolling_{0}.pickle'.format(rollingWindows)
    if not os.path.exists(nonusRet_FILE):
        nonusRetDict = dict()
        print("Calc Non-US fund Exc Ret... ")
        for i in df_nonus.columns:
            df_input = pd.concat([df_nonus[i], benchmark_nonus], axis=1).dropna()

            output = {
                'name': '',
                'type': '',
                'excRet': [],
                'TE': [],
                'IR': []
            }

            output['type'] = Rawqualitative_nonus ['managertype'][i]
            output['name'] = i

            # calculate RollingReturn, TE, and IR

            excRet, TE, IR = calcRollingExcTEIR(df_input, rollingWindows)
            output['excRet'] = excRet
            output['TE'] = TE
            output['IR'] = IR
            nonusRetDict[i] = output

        foutput = open(nonusRet_FILE, 'wb')
        pickle.dump(nonusRetDict, foutput)
        foutput.close()
    else:
        print('Loading non-us Return from file ... ')
        fnonUSret = open(nonusRet_FILE, 'rb')
        nonusRetDict = pickle.load(fnonUSret)
        fnonUSret.close()

        print('Calculate non-US Alpha, TE and IR Done.')

# ==================
#   Output table
# ==================



    # create fund ID and fund names
    fundID_csv = 'combUSnonUS_RetAttribution_CSV.csv'

    if not os.path.exists(fundID_csv):
        fund_ID = pd.DataFrame(columns=['combinedName', 'USname', 'nonUSname', 'fundID',  \
                                       'USid', 'NonUSid','usRet', 'nonUSRet','combUSnonUSret'])

        for idx, us_name in enumerate(df_us.columns):
            # print("Generating fundID for usFund: {0}".format(us_name))
            for idy, nonus_name in enumerate(df_nonus.columns):
                appendDict = {
                    'combinedName': us_name + "x" + nonus_name,
                    'USname': us_name,
                    'nonUSname': nonus_name,
                    'fundID': "US{0}-NonUS{1}".format(idx, idy),
                    'USid': idx,
                    'NonUSid': idy,

                    'usRet':[],
                    'nonUSRet':[],
                    'combUSnonUSret':[],
                    'usRetSign':[],
                    'nonUSRetSign':[],
                    'combUSnonUSretSign':[]

                }
                appendDict['usRet']          = usRetDict['excRet'][us_name]
                appendDict['nonusRet']       = nonusRetDict['excRet'][nonus_name]
                appendDict['combUSnonUSret'] = combinedDict['excRet'][combinedName]

                if usRetDict['excRet'][us_name] >= 0:
                    appendDict['usRetSign'] = 'Pos'
                else:
                    appendDict['usRetSign'] = 'Neg'


                if nonusRetDict['excRet'][nonus_name] >= 0:
                    appendDict['nonusRetSign'] = 'Pos'
                else:
                    appendDict['nonusRetSign'] = 'Neg'

                if combinedDict['excRet'][combinedName] >= 0:
                    appendDict['combUSnonUSretSign'] = 'Pos'
                else:
                    appendDict['combUSnonUSretSign'] = 'Neg'


                fund_ID = fund_ID.append(appendDict, ignore_index=True)


        fund_ID.to_pickle(lowTE_FILE_pickle)

        #example_Print = fund_ID[]
        #selectDate = fund_ID[target][quartileResult[target].index >= quarStartMth]

        #fund_ID.to_csv(fundID_csv)




