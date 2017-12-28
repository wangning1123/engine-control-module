"""
World Bank Group Pension Department
Author:  Natan Goldberger
email: ngoldberger@worldbank.org
"""

import sqlite3
import pandas as pd
import numpy as np
import datetime as dt


class DataConnect:
    def __init__(self, path='C:\\Users\\wb514964\\Code\\em-equity\\database\\', database='EMEQ.db'):
        self.path = path
        self.database = database

    def download_data(self, table_name, columns=None, filter_attribute=None, filter_values=None, init=None, end=None,
                      local_names=False):
        """ Function to retrieve tables from database and filter by columns
        :table_name: table to download from the database
        :columns: columns from table_name to include in the Dataframe
        :filter: attributes to use for filtering the table
        :filter_values: values of the attributes used to filter
        """
        conn = sqlite3.connect(self.path + self.database)
        if table_name == 'performance':
            funds_list = [str(fund) for fund in filter_values[0]]
            funds_list = ', '.join(funds_list)
            if local_names:
                query = ("SELECT performance.*, identifiers.Local_Name FROM performance INNER JOIN identifiers ON"
                         " performance.Product_ID = identifiers.Product_ID WHERE performance.Product_ID IN (" +
                         funds_list + ")")
            else:
                query = ("SELECT performance.*, identifiers.Product_Name FROM performance INNER JOIN identifiers ON"
                         " performance.Product_ID = identifiers.Product_ID WHERE performance.Product_ID IN (" +
                         funds_list + ")")
            returns_data = pd.read_sql_query(query, conn)
            conn.close()
            table = self.clean_performance_data(returns_data, init=init, end=end, local_names=local_names)
        else:
            if filter_attribute is not None:
                filter_query = " WHERE " + ' AND '.join(
                    # [filter_attribute[i] + " IN ('" + "', '".join([values for values in [filter_values[i]]][0]) + "')"
                    [filter_attribute[i] + " IN ('" + "', '".join([values for values in [filter_values]][0]) + "')"
                     for i in range(0, len(filter_attribute))])
            else:
                filter_query = ""

            if columns is not None:
                attributes = ', '.join(columns)
            else:
                attributes = '*'

            query = ("SELECT " + attributes + " FROM " + table_name + filter_query)
            table = pd.read_sql_query(query, conn)

        conn.close()
        return table

    @staticmethod
    def clean_performance_data(performance_data, init=None, end=None, local_names=False):
        """ Function to convert the date from the database to datetime and pivot the table
        :performance_data: table with monthly performance data
        :init: initial date for the final table
        :end: end date for the final table
        :local_names: If True, it will use the local names on the database as columns
        """
        if local_names:
            returns_data = performance_data.pivot(index='Date', columns='Local_Name').xs('Return', axis=1,
                                                                                        drop_level=True)
        else:
            returns_data = performance_data.pivot(index='Date', columns='Product_Name').xs('Return', axis=1,
                                                                                        drop_level=True)
        funds = returns_data.columns
        returns_data = returns_data[funds].replace('---', np.nan)
        returns_data.index = [dt.datetime.strptime(returns_data.index[i], '%Y-%m-%d %H:%M:%S') for i in
                              range(0, len(returns_data.index))]
        if init is not None:
            returns_data = returns_data.loc[init:]
        if end is not None:
            returns_data = returns_data.loc[:end]

        return returns_data.groupby(pd.TimeGrouper(freq='M')).first()

    @staticmethod
    def clean_index_data(index_data, init=None, end=None):
        """ Function to convert the date from the database to datetime and pivot the table
        :index_data: price level of indices
        :init: initial date for the final index table
        :end: end date for the final index table
        """
        index_data = index_data.pivot(index='Date', columns='Index_Name').xs('Level', axis=1,
                                                                                       drop_level=True)
        indices = index_data.columns
        index_data = index_data[indices].replace('---', np.nan)
        index_data.index = [dt.datetime.strptime(index_data.index[i], '%Y-%m-%d %H:%M:%S') for i in
                              range(0, len(index_data.index))]
        if init is not None:
            index_data = index_data.loc[init:]
        if end is not None:
            index_data = index_data.loc[:end]

        return index_data.groupby(pd.TimeGrouper(freq='M')).sum()

    def upload_data(self, table_name, excel_name, sheetname='', skip_rows=0,
                    stack=False, new_column_names=None):
        """ Function to assist with uploading information from an Excel spreadsheet to a SQL database. This function
        avoids generating duplicates in the database in the process of uploading a DataFrame to the table.
        :table_name: name of the table to work with
        :excel_name: excel document name
        :sheetname: (Optional) name of the sheet to read
        :skip_rows: (Optional) number of rows to skip on the Excel document
        """
        if new_column_names is None:
            new_column_names = []
        excel_data = pd.read_excel(excel_name, sheetname=sheetname, skiprows=skip_rows)
        sql_file = self.path + self.database
        conn = sqlite3.connect(sql_file)
        if stack:
            excel_data = excel_data.stack(0)
            # excel_data.columns = new_column_names
            cols = [excel_data.index.values[i] for i in range(0, len(excel_data))]
            df = pd.DataFrame(cols)
            df[2] = excel_data.values
            df.columns = new_column_names
            df.to_sql(table_name + '_temp', conn, if_exists='replace', index=False)
        else:
            excel_data.to_sql(table_name + '_temp', conn, if_exists='replace', index=False)

        c = conn.cursor()
        query = ("CREATE TABLE new_table AS SELECT * FROM " + table_name + "_temp UNION SELECT * FROM " + table_name)
        c.execute(query)
        query = ("DROP TABLE " + table_name + "_temp")
        c.execute(query)
        query = ("DROP TABLE " + table_name)
        c.execute(query)
        query = ("ALTER TABLE new_table RENAME TO " + table_name)
        c.execute(query)
        conn.commit()
        conn.close()


if __name__ == '__main__':
    dc = DataConnect()
    #sample = dc.download_data('general_info', ['Product_ID'], filter_attribute=['Asset_Class', 'Investment_Approach'],
    #                          filter_values=[['Equity'], ['Quantitative', 'Fundamental']])
    #performance = dc.download_data('performance', filter_values=[['37','303']], init='2005-01-01', end='2017-08-01')
    # dc = DataConnect(path='C:\\Users\\wb514964\\Code\\dm-fixedincome\\database\\', database='DMFI.db')
    # dc.upload_data('performance', 'C:\\Users\\wb514964\\Desktop\\zeal.xlsx', 'Sheet1')
    # dc.upload_data('identifiers', 'C:\\Users\\wb514964\\Desktop\\upload.xlsx', 'Sheet2')
    # dc.upload_data('performance', 'C:\\Users\\wb514964\\Desktop\\msci.xlsx', 'Sheet1', stack=True,
    #                new_column_names=['Date', 'Product_ID', 'Return'])
    dc.upload_data('MSCI_indices', 'C:\\Users\\wb514964\\Desktop\\China factors.xlsx', 'up2db', stack=True,
                   new_column_names=['Date', 'Index_Name', 'Level'])
    # x = dc.download_data('MSCI_indices',['Date','Index_Name','Level'],
    #                  filter_attribute=['Index_Name'],
    #                  filter_values=['china', 'korea', 'brazil', 'india', 'russia', 'frontier'])
    # dc.upload_data('performance','C:\\Users\\wb514964\\Desktop\\pepe.xlsx', sheetname='Sheet1')