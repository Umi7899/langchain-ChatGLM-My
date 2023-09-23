import pandas as pd
import numpy as np
import tensorflow as tf
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import TensorBoard
import datetime
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt
import os

# 顺位生成七码
def complete_combination(combination):
    unique_combination = sorted(list(set(combination)))  # 去重并排序
    max_value = max(unique_combination)  # 最大值

    while len(unique_combination) < 7:  # 直到组合长度为7
        max_value = (max_value + 1) % 10  # 下一个补充的数字
        if max_value not in unique_combination:  # 确保补充的数字不重复
            unique_combination.append(max_value)
            unique_combination = sorted(unique_combination)  # 重新排序

    return unique_combination

# 从数据库获得数据
def getdatafrommysql(table):
    cnx = pymysql.connect(
        host="124.221.25.6", 
        port=3306,
        user="dev", 
        password="?7h3{<+2{47!..H", 
        database="ticket_crawler"
    )

    # 创建游标对象
    cursor = cnx.cursor()

    # 执行查询语句
    query = "SELECT * FROM {}".format(table)
    cursor.execute(query)

    # 提取查询结果
    rows = cursor.fetchall()
    
    # 获取列名列表
    column_names = [desc[0] for desc in cursor.description]

    # 关闭游标和连接
    cursor.close()
    cnx.close()
    return pd.DataFrame(rows, columns=column_names)

# 将数据处理为可用数据
def seperatedata(data):
    # 创建新的列  
    # data[['num1', 'num2', 'num3', 'num4', 'num5']] = []
    # 将 num 列的值分配给新列  
    data[['num1', 'num2', 'num3', 'num4', 'num5']] = data['num'].str.split(',').tolist()
    # 删除原始的 num 列  
    data = data.drop('num', axis=1)
    # 设置新的列顺序  
    new_order = ['id', 'no_id', 'kj_date', 'num1', 'num2', 'num3', 'num4', 'num5', 'create_time', 'update_time', 'del_flag']
    # 更换列位置   
    data = data[new_order]
    data = data.rename(columns={'kj_date': 'date'}) 
    return data

# 向数据库写入数据
def writedatatomysql(sd, ed, pred10, top3nums):
    # 建立数据库连接
    db = pymysql.connect(
        host="124.221.25.6", 
        port=3306,
        user="dev", 
        password="?7h3{<+2{47!..H", 
        database="ticket_crawler"
    )

    # 创建游标对象
    cursor = db.cursor()

    # 定义要插入的数据
    data = {
        "start_date": sd,
        "end_date": ed
    }
    for i, prediction in enumerate(pred10):
        group_key = f"group_{i + 1}"
        prediction_str = str(prediction.tolist()).replace("[", "").replace("]", "").replace(",", " ")
        data[group_key] = prediction_str

    for i, t3n in enumerate(top3nums):
        key = f"digit_{i + 1}"
        t3n_str = str(t3n).replace("[", "").replace("]", "").replace(",", " ")
        data[key] = t3n_str
        
    # 检查是否存在相同的数据
    query = "SELECT * FROM ai_pl WHERE start_date = %(start_date)s AND end_date = %(end_date)s"
    cursor.execute(query, data)
    existing_data = cursor.fetchall()

    if len(existing_data) == 0:
        # 定义插入数据的SQL语句
        sql = """
        INSERT INTO ai_pl (start_date, end_date, group_1, group_2, group_3, group_4, group_5,
                                group_6, group_7, group_8, group_9, group_10, digit_1, digit_2, digit_3, digit_4, digit_5)
        VALUES (%(start_date)s, %(end_date)s, %(group_1)s, %(group_2)s, %(group_3)s, %(group_4)s,
                %(group_5)s, %(group_6)s, %(group_7)s, %(group_8)s, %(group_9)s, %(group_10)s,
                %(digit_1)s, %(digit_2)s, %(digit_3)s, %(digit_4)s, %(digit_5)s)
        """

        # 执行插入操作
        cursor.execute(sql, data)

        # 提交更改
        db.commit()
        print("数据添加成功")
    else:
        print("数据已存在，未存储")
    # 关闭数据库连接
    db.close()

# 匹配计数器（已弃用）
def direct_compare(unique_predictions, unique_given_data, name):
    # 完全匹配的数据计数器
    exact_matches = 0
    # 不考虑顺序的数据计数器
    unordered_matches = 0
    for prediction in unique_predictions:
        # 如果预测数据与给定的数据完全对应，计数器加一
        if prediction in unique_given_data:
            exact_matches += 1
            print("{}直选匹配的数据为：{}".format(name, prediction))
        # 如果预测数据与给定的数据不考虑顺序后能够对应，计数器加一
        else:
            if any(set(prediction) == set(given) for given in unique_given_data):
                unordered_matches += 1
                print("{}组选匹配的数据为：{}".format(name, prediction))
    return exact_matches, unordered_matches

# 将正确答案与指定七码比较是否命中
def transresultto7code(orinums, seven_num):
    # seven_num = [[0,1,2,3,4,8,9], [0,3,4,5,6,7,9], [0,1,5,6,7,8,9], [1,2,3,4,5,6,7], [0,2,4,5,6,7,8], [2,3,5,6,7,8,9]]
    array_7code = []
    for i, orinum in enumerate(orinums):
        seven_count_orinums = [0] * len(seven_num)
        for j, num_list in enumerate(seven_num):
            if all(num in num_list for num in orinum):
                seven_count_orinums[j] = 1
        array_7code.append(seven_count_orinums)
    return np.array(array_7code)

# 绘制表格
def drawtable(column_names, data, mon, tod, res, lastname):
    # 定义列名
    row_names = ["MON", "TUE", "THU", "WED", "FRI", "SAT", "SUN"]
    new_column_names = ['res']
    new_res = []
    for col_name in column_names:  
        col_name = str(col_name).replace('[', '')
        col_name = col_name.replace(']', '')
        col_name = col_name.replace(', ', '')
        new_parts = col_name[:3] + "\n" + col_name[3:]
        new_column_names.append(''.join(new_parts))

    for re in res:  
        re = str(re).replace('[', '')
        re = re.replace(']', '')
        re = re.replace(' ', '')
        new_res.append(''.join(re))
    # 创建空的DataFrame表格
    df = pd.DataFrame(index=row_names, columns=new_column_names)

    # 填充表格
    for i, row in enumerate(row_names):
        if i < len(data):
            df.loc[row] = [new_res[i]] + ['√' if val == 1 else '' for val in data[i]]
        else:
            df.loc[row] = '-'

    # 绘制表格
    plt.figure(figsize=(10, 8))  # 调整图表大小

    # 调整行高度
    cell_height = 0.05  # 行高度
    table = plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center', cellColours=[['w']*len(new_column_names)]*len(row_names))
    for i, key in enumerate(table.get_celld().keys()):
        if key[0] == 0:
            table[key].set_height(cell_height)

    plt.axis('off')
    if not os.path.exists('./result/7code'): 
        os.makedirs('./result/7code') 
    plt.savefig('./result/7code/{}-{}-{}.png'.format(mon, tod, lastname), bbox_inches='tight', pad_inches=0)
    from PIL import Image

    # 打开原始图像
    original_image = Image.open('./result/7code/{}-{}-{}.png'.format(mon, tod, lastname))

    # 获取原始图像的尺寸
    original_width, original_height = original_image.size

    # 设置裁切后的目标尺寸
    target_width = original_width
    target_height = 160

    # 计算裁切的左上角和右下角坐标
    left = 0
    top = (original_height - target_height) // 2
    right = original_width
    bottom = top + target_height

    # 裁切图像
    cropped_image = original_image.crop((left, top, right, bottom))

    # 调整图像大小
    resized_image = cropped_image.resize((target_width, target_height))

    # 保存裁切并调整大小后的图像
    resized_image.save('./result/7code/{}-{}-{}.png'.format(mon, tod, lastname))
    # plt.show()

#寻找周一
def findmonday(data):
    for index, row in data.iterrows():
        date_str = row['date']
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        if date.weekday() == 0:  # 0表示周一
            return index

# data = getdatafrommysql('digit_pl')
data = pd.read_csv("plw_data.csv")
if 'num' in data.columns:  
   data = seperatedata(data)
   print(data)

tod = 0
mon = findmonday(data)
# use = str(input("是否使用已有权重(y/n)"))
use = 'n'
# datanum = len(data)-mon-8
datanum = 20
ep = 100
nump = 1  #应对不同的数据来源。1是本地

# 读取彩票数据
X = data.iloc[mon+8:mon+8+datanum, nump:nump+3].values.astype(int) # 训练数据，延后七天
y = data.iloc[mon+1:mon+1+datanum, nump:nump+3].values.astype(int) # 标签
val_features = data.iloc[mon+1:mon+21, nump:nump+3].values.astype(int) # 预测数据
given_data_3 = data.iloc[tod:mon+1, nump:nump+3].values.astype(int)
given_data_3 = given_data_3[::-1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定义自编码模型
input_dim = X_train.shape[1]
encoding_dim = 5
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)

# 编译自编码模型
autoencoder.compile(optimizer='adam', loss='mse')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 训练自编码模型
if use == 'y':
    autoencoder.load_weights('./weights/7code/autoencoder.h5')
else:
    autoencoder.fit(X_train, X_train, epochs=ep, batch_size=32)


# 提取特征
encoder_model = tf.keras.models.Model(inputs=input_layer, outputs=encoder)
encoded_X_train = encoder_model.predict(X_train)
# encoded_test = encoder_model.predict(test)
encoded_X_test = encoder_model.predict(X_test)

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(encoding_dim,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(input_dim, activation='linear')
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
if use == 'y':
    model.load_weights('./weights/7code/model.h5')
else:
    model.fit(encoded_X_train, y_train, epochs=ep, batch_size=32, callbacks=[tensorboard_callback])
# print(test_features)

# 预测下一期彩票号码
val_features_encoded = encoder_model.predict(val_features)
next_lottery_numbers = model.predict(val_features_encoded)
nn_predictions = model.predict(encoded_X_test)
x_train_pred = model.predict(encoded_X_train)

# 使用决策树模型预测下一期彩票号码
tree = DecisionTreeClassifier()
if use == 'y':
    tree = joblib.load('./weights/7code/tree.pkl')
else:
    tree.fit(x_train_pred, y_train)
predictions_5 = tree.predict(next_lottery_numbers)
predictions_3 = predictions_5[:, :3]
dt_predictions = tree.predict(nn_predictions)

print("预测周次：从{}到{}".format(data.iloc[mon, nump-1], data.iloc[tod, nump-1]))

# 存入数据
# writedatatomysql(start_date, end_date, predictions_5, top_three_nums)

# 将七天预测转换为七码
completed_combinations = []
for combination in predictions_3:
    completed_combination = complete_combination(combination)
    completed_combinations.append(completed_combination)

unique_cc = list(set(map(tuple, completed_combinations)))  # 使用set去重，并将元素转换为元组以便进行比较
unique_cc = [list(item) for item in unique_cc]  # 将元组转换回列表
'''
unique_cc为预测结果七码，送入大模型
'''
unique_cc_str = ', '.join(map(str, unique_cc[0:6]))

print(unique_cc_str)

# print(unique_cc[0:6])
# print(transresultto7code(given_data_3, unique_cc[0:6]))
seven_num = [[0,1,2,3,4,8,9], [0,3,4,5,6,7,9], [0,1,5,6,7,8,9], [1,2,3,4,5,6,7], [0,2,4,5,6,7,8], [2,3,5,6,7,8,9]]
drawtable(unique_cc[0:6], transresultto7code(given_data_3, unique_cc[0:6]), str(data.iloc[mon, nump-1]).replace('-', ''), str(data.iloc[tod, nump-1]).replace('-', ''), given_data_3, "pred")
drawtable(seven_num, transresultto7code(given_data_3, seven_num), str(data.iloc[mon, nump-1]).replace('-', ''), str(data.iloc[tod, nump-1]).replace('-', ''), given_data_3, "ori")

if use != 'y':
    # save = str(input("是否保存模型(y/n)"))
    save = 'y'
    if save == 'y':
        if not os.path.exists('./weights/7code'): 
            os.makedirs('./weights/7code') 
        autoencoder.save_weights('./weights/7code/autoencoder.h5')
        model.save_weights('./weights/7code/model.h5')
        joblib.dump(tree, './weights/7code/tree.pkl')
