import openpyxl

# 打開 Excel 檔案
workbook = openpyxl.load_workbook('bank.xlsx')

# 選擇指定的工作表
worksheet = workbook['Account_information']

# 遍歷每一列
for row in worksheet.iter_rows():
    # 獲取第三和第四個元素
    element_3 = row[2].value
    element_4 = row[3].value
    
    # 根據條件更改最後一欄的值
    if element_3 == "yes" and element_4 == 'yes':
        row[-1].value = 4
    if element_3 == "yes" and element_4 == 'no':
        row[-1].value = 3
    if element_3 == "no" and element_4 == 'yes':
        row[-1].value = 2
    if element_3 == "no" and element_4 == 'no':
        row[-1].value = 1

# 儲存 Excel 檔案
workbook.save('bank.xlsx')