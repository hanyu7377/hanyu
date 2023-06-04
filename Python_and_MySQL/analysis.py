import mysql.connector
import matplotlib.pyplot as plt
import getpass
password = getpass.getpass("please input your password: ")
def connect_to_sql():
    cnx = mysql.connector.connect(
        host='localhost',
        user='root',
        password=password,
        database='stock_database'
    )
    return cnx  
def company(table):
    
    cnx = connect_to_sql()
    
    cursor = cnx.cursor()
    command_in_query = f'SELECT * FROM {table}_stock_prices;'
    cursor.execute(command_in_query)
    data = cursor.fetchall()
    price = []
    date = []
    for row in data:
        row =list(row)
        price.append(row[4])
        date.append(row[0])
    return date ,price

company_list = ["GOOGL","AMZN","MSFT",'TSLA']

for ele, index in enumerate(company_list):
    plt.plot((company(company_list[ele]))[0] , (company(company_list[ele]))[1] , marker = "o",markersize = 3,label = company_list[ele])
plt.xlabel("Date",fontsize = 20)
plt.ylabel("Prices (USD)",fontsize = 20)
plt.title("The stock prices variety of 4 big company",fontsize = 24)
plt.tight_layout
plt.legend(shadow = True,loc = "best",fontsize = 12)
plt.show()
    