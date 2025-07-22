# 基本语句
## 判断
## 循环
正负下标
```python
cn_list = ['Apple', 'Facebook', 'Alibaba', 'Bilibili', 'Tesla', 'Tencent']
print(cn_list[-1])
print(cn_list[:3])
print(cn_list[2:4]) # 2<=index<4
print(cn_list[-3:])
```
## 数据结构
### list
```python
cn_list.append('Amazon')
cn_list = cn_list + ['Luckin', 'Starbucks']
cn_list.remove(cn_list[i])
cn_list.insert(4, 'KFC')
```
### dictionary
没有优先级 = map
```python
files = {"ID": 111, "passport": "my passport", "books": [1,2,3]}
print('files["ID"]:', files["ID"])
print(files.get("nonexist", "默认值"))  # 输出 默认值（因为 key 不存在）
files.update({"files": ["1", "2"]})
popped = files.pop("ID") # 弹出并返回值
for key, value in files.items():
    print("key:", key, ", value:", value) # 遍历
```
### tuple
特点：里面的元素不可变
```python
files = ("file1", "file2", "file3")
```
### set
避免重复元素，无序
```python
my_files = set(["file1", "file2", "file3"])
my_files.add("file3")
my_files.remove("file3")
print("交集 ", your_files.intersection(my_files))
print("并集 ", your_files.union(my_files))
print("补集 ", your_files.difference(my_files))
```
# 库
## numpy
```python
import numpy as np
my_array = np.array([1,2,3])
```
```python
cars = np.array([5, 10, 12, 6])
print("数据：", cars, "\n维度：", cars.ndim)
```
```python
cars1 = np.array([5, 10, 12, 6])
cars2 = np.array([5.2, 4.2])
cars = np.concatenate([cars1, cars2])  # 添加数据
print(cars)
```
合并数据
```python
test1 = np.array([5, 10, 12, 6])
test2 = np.array([5.1, 8.2, 11, 6.3])
# 首先需要把它们都变成二维，下面这两种方法都可以加维度
test1 = np.expand_dims(test1, 0)
test2 = test2[np.newaxis, :]
print("test1加维度后 ", test1)
print("test2加维度后 ", test2)
# 然后再在第一个维度上叠加
all_tests = np.concatenate([test1, test2])
print("括展后\n", all_tests)
print("第一维度叠加：\n", np.concatenate([all_tests, all_tests], axis=0))
print("第二维度叠加：\n", np.concatenate([all_tests, all_tests], axis=1))
# 竖直水平合并
a = np.array([
    [1,2],
    [3,4]
])
b = np.array([
    [5,6],
    [7,8]
])
print("竖直合并\n", np.vstack([a, b]))
print("水平合并\n", np.hstack([a, b]))
```
```python
# 筛选数据
condition = a > 7
print(np.where(condition, -1, a))
```

# 画图
## 环境配置
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
```
## 散点图
```python
tips = sns.load_dataset("tips")
sns.relplot(data=tips, x="total_bill", y="tip") # 2-dimensional
sns.relplot(data=tips, x="total_bill", y="tip", hue="smoker") # 3-dimensional
```
第三个维度:颜色，float类型渐变
## 折线图
```python
dowjones = sns.load_dataset("dowjones")
sns.relplot(data=dowjones, x="Date", y="Price", kind="line")
```
显示95%误差
```python
fmri = sns.load_dataset("fmri")
sns.relplot(data=fmri, x="timepoint", y="signal", kind="line")
```