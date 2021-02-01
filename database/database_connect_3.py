import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymysql

def sql2df(name_cur, name_tab_str):
    sql_data='''select * from %s; '''%name_tab_str
    name_cur.execute(sql_data)
    data = name_cur.fetchall()
    cols = [i[0] for i in name_cur.description]
    df = pd.DataFrame(np.array(data), columns=cols)
    return df

conn = pymysql.connect(host='localhost', port=3306, user="root", passwd="root", db="db")
cursor = conn.cursor()
df_read = sql2df(cursor, 'meter')

# 按行遍历
flag = True
length = df_read.shape[0]
for index, row in df_read.iterrows():
    if row['code'] is None:
        (df_read.fillna(method='ffill', limit=1) + df_read.fillna(method='bfill', limit=1)) / 2
        value = df_read.loc[index]['code']
        row_id = row['id']
        sql_update = 'update db set code = {} where id = {}'.format(value, row_id)
        cursor.execute(sql_update)
        flag = False

cursor.colse()
conn.commit()



df_null = df_read[df_read.isnull().T.any()]
# 所有空值的位置 [行号， 列号]
df_null = df_read.isnull().stack()[lambda x : x].index.tolist()


for i in df_null:
    if i[1] == 'code':
        sql_update = 'update db set {} = {} where id = {}'.format(i[1], value, i[0])

df_read['code'].fillna(method='bfill', inplace=True)

df_read.loc[3, 'code'] = '123'



df_read.to_sql('meter', con=engine, if_exists='replace', index=False)

print(df_read)
