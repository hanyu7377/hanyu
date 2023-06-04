import yfinance as yf
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import getpass
password = getpass.getpass("please input your password: ")
def connect_to_sql():
    cnx = None
    try:
        cnx = mysql.connector.connect(
            host='localhost',
            user='root',
            password=password,
            database='stock_database'
        )
        print("MySQL database connection successful")
    except Error as err:
        print(f"Error: '{err}'")    
    return cnx   
def get_data_and_store_tosql(company_name):
    data = yf.download(company_name, datetime.now() - timedelta(days=365), datetime.now())
    cnx = connect_to_sql()
    cursor = cnx.cursor()
    table_name = f'{company_name}_stock_prices'
    

    drop_table_query = f"DROP TABLE IF EXISTS `{table_name}`"
    cursor.execute(drop_table_query)
    cnx.commit()
    
    create_table_query = f'''
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            Date DATE,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume BIGINT,
            Adj_Close FLOAT,
            PRIMARY KEY (Date)
        )
    '''

    cursor.execute(create_table_query)

    # 插入資料
    insert_query = f'''
        INSERT INTO `{table_name}` (Date, Open, High, Low, Close, Volume, Adj_Close)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    '''

    for index, row in data.iterrows():
        date = index.date()
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        volume = row['Volume']
        adj_close = row['Adj Close']
        values = (date, open_price, high_price, low_price, close_price, volume, adj_close)
        cursor.execute(insert_query, values)
    cnx.commit()
    cursor.close()
    cnx.close()

    print(f"Get data and store them at `{table_name}` table successfully")
get_data_and_store_tosql("MSFT")
get_data_and_store_tosql("GOOGL")
get_data_and_store_tosql("AMZN")
get_data_and_store_tosql("TSLA")