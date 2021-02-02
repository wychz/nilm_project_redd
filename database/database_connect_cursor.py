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
        row_id = row['id']
        if index < length - 1:
            value_before = df_read.loc[index - 1]['code']
            value_after = df_read.loc[index + 1]['code']
            value = None
            sql_update = ''
            if value_before is not None and value_after is not None:
                value = (value_before + value_after) / 2
                sql_update = 'update db set code = {} where id = {}'.format(value, row_id)
                flag = True
            elif value_before is None and value_after is None:
                continue
            elif flag is True:
                if value_before is None or value_after is None:
                    if value_before is None:
                        value = value_after
                    elif value_after is None:
                        value = value_before
                    sql_update = "update meter set code = {} where id = {}".format(value, row_id)
                    df_read.loc[index]['code'] = value
                    cursor.execute(sql_update)
                    flag = False
        else:
            value_before = df_read.loc[index - 1]['code']
            if value_before is not None:
                sql_update = "update meter set code = {} where id = {}".format(value_before, row_id)
                df_read.loc[index]['code'] = value_before
                cursor.execute(sql_update)

cursor.close()
conn.commit()

