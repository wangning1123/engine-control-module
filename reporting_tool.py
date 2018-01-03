"""
World Bank Group Pension Department
Author:  Natan Goldberger, Joel Niamien
email: ngoldberger@worldbank.org, jniamien@worldbank.org
"""

#from port_fund_analysis import AnalysisTools
#from data_airport import DataConnect
import win32com.client as win32
import win32clipboard
#import pandas as pd


class ReportingTool():

    def export_report(self, table, excel_location, sheet_name, cell_location):
        excel = win32.DispatchEx('Excel.Application')
        excel.Visible = False
        check = False
        
              
        try:
            f = open(excel_location)
            if f:
                check = True
                f.close()

        except IOError:
            check = False
                
                
        if check == True:
            workbook = excel.Workbooks.Open(excel_location)
            if sheet_name not in [workbook.Sheets(i).Name for i in range(1,workbook.Sheets.Count+1)]:
                #print ('adding1')
                worksheet = workbook.Worksheets.Add()
                worksheet.Name = sheet_name
            workbook.Close(SaveChanges=True)
            excel.Application.Quit()

        else:
            workbook = excel.Workbooks.Add()
            workbook.SaveAs(excel_location)
            workbook = excel.Workbooks.Open(excel_location)
            if sheet_name not in [workbook.Sheets(i).Name for i in range(1,workbook.Sheets.Count+1)]:
                #print('adding2')
                worksheet = workbook.Worksheets.Add()
                worksheet.Name = sheet_name
            workbook.Close(SaveChanges=True)
            excel.Application.Quit()

       
        workbook = excel.Workbooks.Open(excel_location)
       
        sheet_to_paste = workbook.Sheets(sheet_name)
        table.to_clipboard()
        sheet_to_paste.Range(cell_location).PasteSpecial()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.CloseClipboard()
        
        workbook.Close(SaveChanges=True)
        excel.Application.Quit()

        return None

