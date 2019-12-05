import pandas as pd
from pandas_datareader import data
import csv


file = open('brazil_tickers.csv','r')
reader = csv.reader(file)
rownum = 0

error_log = open('tickers_nao_baixados','w')
download_log = open('tickers_baixados','w')

close_values = pd.Series()

rownum = 0
all_data = pd.Series()

start_date = '2010-01-01'
end_date = '2019-07-09'


#Download the data from yahoo and save into a csv file
for row in reader:
    try:
        if rownum == 0:
            all_data = data.DataReader(row[0], 'yahoo', start_date, end_date)['Close']
            all_data = all_data.rename(row[0])
            print("baixou " + row[0] + '\n')
        if rownum != 0: 
            result = data.DataReader(row[0], 'yahoo', start_date, end_date)['Close']
            result = result.rename(row[0])
            print("baixou " + row[0] + '\n')
            all_data = pd.merge(all_data,result,on='Date',how='outer')
        download_log.write(row[0] + '\n')
        rownum += 1
    except Exception as e:
        error_log.write(row[0] + '\n')
        print(e)
        print("nao baixou " + row[0] + '\n')

all_data = all_data.sort_index()  
all_data.to_csv('data_stocks.csv')  

file.close()
error_log.close()
download_log.close()
