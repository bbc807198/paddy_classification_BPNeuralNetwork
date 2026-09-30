#导包
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary
# ========================================================
# 1: 数据读取与数据清洗
def clean_dataset():
    # 读取CSV数据
    # 使用pandas读取数据
    df = pd.read_csv("./水稻试验/paddydataset.csv")
    
    # 数据分析发现以下变量无额外信息:
    # Seedrate,LP_Mainfield, Nursery area, LP_nurseryarea, DAP, 
    # Weed, Urea, Potassh, Micronutrients, Pest, 因此删除
    redundant_area_features = [
        "Seedrate(in Kg)",
        "LP_Mainfield(in Tonnes)",
        "Nursery area (Cents)",
        "LP_nurseryarea(in Tonnes)",
        "DAP_20days",
        "Weed28D_thiobencarb",
        "Urea_40Days",
        "Potassh_50Days",
        "Micronutrients_70Days",
        "Pest_60Day(in ml)"
    ]
    # 删除
    df = df.drop(
        columns=redundant_area_features
    )
    # 计算单位面积产量
    df["Yield_per_hectare"] = (
        df["Paddy yield(in Kg)"]
        /
        df["Hectares "]
    )
    # 删除目标泄漏变量Paddy yield,Trash
    leakage_features = [

        "Paddy yield(in Kg)",

        "Trash(in bundles)"
    ]
    existing_leakage = [
        col for col in leakage_features
        if col in df.columns
    ]
    df = df.drop(
        columns=existing_leakage
    )
    """
        # 删除完全重复的数据
        print("\n" + "=" * 60)
        print("开始删除重复样本")
        print("=" * 60)
        # 统计重复数量
        duplicate_num = df.duplicated().sum()
        print("重复样本数量:", duplicate_num)
        # 删除完全一样的行
        df = df.drop_duplicates()
        print("删除重复后数据规模:")
        print(df.shape)"""
    # 保存清洗后的数据
    output_file = "rice_cleaned_dataset_including_repeated.csv"
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

# ========================================================
# 2: 定义函数,构建数据集
def create_dataset():
    #加载CSV文件数据集,本次采用含重复样本
    data=pd.read_csv("rice_cleaned_dataset_including_repeated.csv")
    # 根据三分位数划分Low / Medium / High 0 -> Low 1 -> Medium 2 -> High
    low_threshold = data["Yield_per_hectare"].quantile(1/3)
    high_threshold = data["Yield_per_hectare"].quantile(2/3)
    def classify_yield(value):
        if value <= low_threshold:
            return 0
        elif value <= high_threshold:
            return 1
        else:
            return 2
    data["Yield_Level"] = (data["Yield_per_hectare"].apply(classify_yield))
    # 将标签加入已经清洗好的data
    """
    # 如果Step1中删除了重复数据,
    # 这里应该使用同样的drop_duplicates()
    data = data.drop_duplicates()"""
    # 得到最终预测标签y
    y = data["Yield_Level"]
    # 创建特征变量
    X = data.iloc[:,:-2]
    # 类别变量
    categorical_features = ["Agriblock",
        "Variety",
        "Soil Types",
        "Nursery",
        "Wind Direction_D1_D30",
        "Wind Direction_D31_D60",
        "Wind Direction_D61_D90",
        "Wind Direction_D91_D120"]
    # 数值变量
    numerical_features = [
        col
        for col in X.columns
        if col not in categorical_features
    ]   
    # 类别变量One-Hot编码
    # Soil:clay alluvial
    # 转换:
    # clay       1 0
    # alluvial   0 1
    X_encoded = pd.get_dummies(X,
        columns=categorical_features,
        dtype=float)
    # 所有数据转换float
    X_encoded = (X_encoded.astype(float) )
    # 转回DataFrame
    x = pd.DataFrame(
        X_encoded,
        columns=X_encoded.columns
    )
    #保存数据
    x.to_csv(
        "./水稻试验/x.csv",
        index=False
    )
    y.to_csv(
        "./水稻试验/y.csv",
        index=False
    )
    #切分训练集与测试集
    #参考y的类别分布,分割后两个数据集都保持原有比例
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,stratify=y)
    #数据标准化
    scaler = StandardScaler()
    x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
    x_test[numerical_features] = scaler.transform(x_test[numerical_features])
    #将数据集封装为张量数据集 数据->张量Tensor->数据集TensorDataSet->数据加载器DataLoader
    train_dataset=TensorDataset(torch.tensor(x_train.values).float(),torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test.values).float(),torch.tensor(y_test.values))
    #返回结果,x_train.shape[1]充当特征数,len(np.unique(y)充当标签数
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

# ========================================================
# 3:搭建神经网络
class PaddyYieldModel(nn.Module):
    #在init魔法方法中,初始化父类成员以及搭建神经网络
    def __init__(self,input_dim,output_dim):
        #初始化父类成员
        super().__init__()
        #搭建神经网络,采用两个隐藏层加输出层
        self.linear1=nn.Linear(in_features=input_dim,out_features=128)
        self.linear2=nn.Linear(in_features=128,out_features=64)
        self.output=nn.Linear(in_features=64,out_features=output_dim)
   
    #定义前向传播方法(forward)
    def forward(self,x):
        #隐藏层1:加权求和+激活函数(relu)
        x=torch.relu(self.linear1(x))
        #隐藏层2:加权求和+激活函数(relu)
        x=torch.relu(self.linear2(x))
        #输出层:加权求和+激活函数(softmax)
        #x=torch.softmax(self.output(x),dim=1)
        #由于后续采用多分类交叉熵损失函数,这里只做加权求和
        x=self.output(x)
        return x

# ========================================================
# 4.模型训练
def train(train_dataset,input_dim,output_dim):
    #创建数据加载器(参数1:数据集对象2789*0.8,参数2:每批次训练条数,参数3:是否打乱)
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #创建神经网络模型
    model=PaddyYieldModel(input_dim,output_dim)
    #定义损失函数,使用多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #创建优化器对象
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #定义轮数变量
    epochs = 50
    #开始每轮训练
    for epoch in range(epochs):
        #定义损失值变量和训练批次数变量
        total_loss,batch_num=0.0,0
        #定义训练开始的时间变量
        start=time.time()
        #开始每批次训练
        for x,y in train_loader:
            #切换至训练模型状态(训练模式或者测试模式)
            model.train()
            #计算预测值
            y_pred = model(x)
            #计算损失
            loss = criterion(y_pred,y)
            #梯度清零,反向传播,更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #累加损失值(各批次的平均损失累加)
            total_loss += loss.item()
            batch_num += 1
        #本轮训练结束,打印训练信息
        print(f"epoch:{epoch+1},loss:{total_loss / batch_num:.4f},time:{time.time()-start:.2f}s")
    #各轮训练结束,保存模型(参数)
    #print(f"\n\n模型的参数信息:{model.state_dict()}\n\n")
    torch.save(model.state_dict(),f="./水稻试验/model/modeldata.pth")

# ========================================================
# 5.模型测试
def evaluate(train_dataset,input_dim,output_dim):
    #创建神经网络分类对象
    model= PaddyYieldModel(input_dim,output_dim)
    #加载模型参数
    model.load_state_dict(torch.load("./水稻试验/model/modeldata.pth"))
    #创建测试集的数据集加载器对象
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)
    #定义变量,记录预测正确的样本数
    correct=0
    #开始测试
    for x,y in test_loader:
        #切换为测试模式
        model.eval()
        #计算预测值,由于定义输出层时只进行加权求和,没有设置激活函数,因此这里的y_pred还只是一个向量
        y_pred = model(x)
        #转换为类别,用argmax获取最大值对应的下标,dim=1表示行处理
        y_pred = torch.argmax(y_pred,dim=1)
        #print(y_pred)
        #统计预测正确的个数
        #print(y_pred==y)
        #print((y_pred==y).sum())
        correct+=(y_pred==y).sum()
        
    #模型测试结束,打印准确率
    print(f"准确率:{correct / len(test_dataset):.4f}")
        
# 6:测试
if __name__=="__main__":
    #1清洗数据
    #clean_dataset()
    #2准备数据集
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    """print(train_dataset)
    print(test_dataset)
    print(input_dim)
    print(output_dim)"""
    #3构建神经网络模型
    #model=PaddyYieldModel(input_dim,output_dim)
    #计算模型参数(16指每批次十六条数据)
    #summary(model,input_size=(16,input_dim))
    #4模型训练
    train(train_dataset,input_dim,output_dim)
    #5模型测试
    evaluate(train_dataset,input_dim,output_dim)
