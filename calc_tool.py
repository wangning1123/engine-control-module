# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:27:48 2017

@author: WB522266
"""



    # ==========================================================================
    #           CALCULATE ROLLING RETURN, TRACKING ERROR, AND INFORMATION RATIO
    # ==========================================================================
    
# Goal: calculate rolling return
# Input: dataframe (monthly return in percentage) and rolling window (number of month, default is 12)



def calcRollingExcTEIR(inputData, window=12):
    import numpy as np
    import pandas as pd

    fundRet = inputData[inputData.columns[0]]
    bmkRet  = inputData[inputData.columns[-1]]
    excessRet = fundRet - bmkRet

    rollingFund = fundRet.rolling(window=window, center=False).apply(lambda x: (np.prod(1+x/100)**(12/window)-1)*100)
    rollingBmk = bmkRet.rolling(window=window, center=False).apply(lambda x: (np.prod(1+x/100)**(12/window)-1)*100)
    excRollRet = rollingFund - rollingBmk

    retMonthDiff     = fundRet.sub(bmkRet, axis = 0)
    trRolling = pd.rolling_std(retMonthDiff,window = window) * pd.np.sqrt(12)

    irRolling = excRollRet / trRolling

    return excRollRet, trRolling, irRolling




    # ================================================
    #  CALCULATE TRANKING ERROR
    # ================================================

def tr(inputData):
    import numpy as np
    import pandas as pd

    fundRet       = inputData[inputData.columns[0]]
    bmkRet        = inputData[inputData.columns[-1]]
    retMonthDiff  = fundRet.sub(bmkRet, axis = 0)

    tr = np.std(retMonthDiff) * np.sqrt(12)

    return tr



    # ================================================
    #  CALCULATE EXCESS RETURN
    # ================================================

def excRetAnn(inputData):
    import numpy as np
    import pandas as pd

    fundRet       = inputData[inputData.columns[0]]
    bmkRet        = inputData[inputData.columns[-1]]

    countMon = len(fundRet.index)

    compFundRet = (np.prod(1+fundRet/100)**(12/countMon)-1)*100
    compBmkRet  = (np.prod(1+bmkRet/100)**(12/countMon)-1)*100

    excRet  = compFundRet - compBmkRet


    return excRet



    
    
    
    
    
    
    
    
    
    
    
    
    