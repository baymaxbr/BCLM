import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

# 读取训练集数据，更改为读取 xlsx 文件
train_data = pd.read_csv('训练集.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Subtype', 'Histology', 'T', 'Sentinel', 'Surgery', 'Radiation', 'Chemotherapy', 'liver', 'brain', 'bone']]
y = train_data['lung']

# 创建并训练LR模型
lr_model = LogisticRegression(C=10)
lr_model.fit(X, y)

# 特征映射
feature_order = [
    'Age', 'Subtype', 'Histology', 'T', 'Sentinel', 'Surgery', 'Radiation', 'Chemotherapy', 'liver', 'brain', 'bone']
class_mapping = {0: "No lung metastasis", 1: "Breast cancer lung metastasis"}
Age_mapper = {"＜64 years": 0, "64-74 years": 1, "＞74 years": 2}
Subtype_mapper = {"HR+/HER2-": 0, "HR-/HER2+": 1, "HR+/HER2+": 2, "HR-/HER2-": 3}
Histology_mapper = {"Non special": 0, "Special": 1}
T_mapper = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
Sentinel_mapper = {"Negative": 0, "Positive": 1}
Surgery_mapper = {"NO": 0, "Yes": 1}
Radiation_mapper = {"NO": 0, "Yes": 1}
Chemotherapy_mapper = {"NO": 0, "Yes": 1}
liver_mapper = {"NO": 0, "Yes": 1}
brain_mapper = {"NO": 0, "Yes": 1}
bone_mapper = {"NO": 0, "Yes": 1}


# 预测函数
def predict_lung_metastasis(Age, Subtype, Histology, T, Sentinel, Surgery, Radiation, Chemotherapy, liver, brain, bone):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[Age]],
        'Subtype': [Subtype_mapper[Subtype]],
        'Histology': [Histology_mapper[Histology]],
        'T': [T_mapper[T]],
        'Sentinel': [Sentinel_mapper[Sentinel]],
        'Surgery': [Surgery_mapper[Surgery]],
        'Radiation': [Radiation_mapper[Radiation]],
        'Chemotherapy': [Chemotherapy_mapper[Chemotherapy]],
        'liver': [liver_mapper[liver]],
        'brain': [brain_mapper[brain]],
        'bone': [bone_mapper[bone]],
    }, columns=feature_order)

    prediction = lr_model.predict(input_data)[0]
    probability = lr_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("LR Model Predicting Lung Metastasis of Breast Cancer")
st.sidebar.write("Variables")

Age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
Subtype = st.sidebar.selectbox("Subtype", options=list(Subtype_mapper.keys()))
Histology = st.sidebar.selectbox("Histology", options=list(Histology_mapper.keys()))
T = st.sidebar.selectbox("T", options=list(T_mapper.keys()))
Sentinel = st.sidebar.selectbox("Sentinel", options=list(Sentinel_mapper.keys()))
Surgery = st.sidebar.selectbox("Surgery", options=list(Surgery_mapper.keys()))
Radiation = st.sidebar.selectbox("Radiation", options=list(Radiation_mapper.keys()))
Chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(Chemotherapy_mapper.keys()))
liver = st.sidebar.selectbox("liver", options=list(liver_mapper.keys()))
brain = st.sidebar.selectbox("brain", options=list(brain_mapper.keys()))
bone = st.sidebar.selectbox("bone", options=list(bone_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_lung_metastasis(Age, Subtype, Histology, T, Sentinel, Surgery, Radiation, Chemotherapy, liver, brain, bone)

    st.write("Class Label: ", prediction)  # 结果显示在右侧的列中
    st.write("Probability of developing lung metastasis: ", probability)