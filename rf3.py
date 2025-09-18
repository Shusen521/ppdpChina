import streamlit as st
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import numpy as np
import pandas as pd

# 定义特征名称（中文标签）
feature_names_chinese = [
    "产假期间工作压力",
    "丈夫能否提供足够照顾",
    "产后休养场所",
    "与公婆关系",
    "孕期体重增长对您是否造成困扰",
    "当前您的睡眠状况",
    "社会支持水平"
]

# 定义特征的选项（中文）
options_dict_chinese = {
    "产假期间工作压力": [("无", 0), ("有", 1)],
    "丈夫能否提供足够照顾": [("能", 1), ("不能", 2)],
    "产后休养场所": [("自己家", 1), ("月子中心", 2), ("其他", 3)],
    "与公婆关系": [("极好", 1), ("好", 2), ("一般", 3), ("差", 4), ("极差", 5)],
    "孕期体重增长对您是否造成困扰": [("否", 0), ("是", 1)],
    "当前您的睡眠状况": [("极好", 1), ("好", 2), ("一般", 3), ("差", 4), ("极差", 5)],
    "社会支持水平": [("高", 1), ("一般", 2), ("低", 3)],
}

# 设置页面标题（中文）
st.title("产后抑郁早期风险预测模型")

# 加载训练好的模型
@st.cache_resource
def load_model():
    model = joblib.load("RF.pkl")
    return model

model = load_model()

# 使用 Streamlit 表单，控制运行行为
with st.form("input_form"):
    st.subheader("请输入以下特征值（请根据您的情况选择相应选项）")
    
    # 输入特征（使用中文选项）
    inputs = {}
    for feature in feature_names_chinese:
        options = [option[0] for option in options_dict_chinese[feature]]
        # 添加提示信息（tooltips）
        inputs[feature] = st.selectbox(
            f"{feature}（请选择一个选项）",
            options,
            help=f"选择您的{feature}状态"
        )
    
    # 添加提交按钮（中文）
    submitted = st.form_submit_button("提交")

# 只有在点击按钮后才运行以下代码
if submitted:
    try:
        # 准备输入数据（将选择的中文标签转换为数字）
        input_data = np.array([dict(options_dict_chinese[feature])[inputs[feature]] for feature in feature_names_chinese]).reshape(1, -1)

        # 预测结果和概率
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # 根据概率进行分类
        prob_depression = probability[0][1]  # 有抑郁风险的概率
        risk_level = ""

        if prob_depression <= 0.25:
            risk_level = "低风险"
        elif 0.26 < prob_depression <= 0.63:
            risk_level = "中度风险"
        else:
            risk_level = "高风险"

        # 显示结果（中文）
        st.subheader("预测结果")
        st.write(f"**抑郁风险概率:** {probability[0][1]:.2%}")
        st.write(f"**风险分类:** {risk_level}")
        
        # 获取特征重要性
        feature_importances = model.feature_importances_
        # 创建DataFrame，使用中文特征名
        importance_df = pd.DataFrame({
            '特征': feature_names_chinese,  # 使用中文特征名
            '重要性': feature_importances
        }).sort_values(by='重要性', ascending=False)

        # 创建用户输入数据的DataFrame
        user_input_values = [
            dict(options_dict_chinese[feature])[inputs[feature]]
            for feature in feature_names_chinese
        ]
        user_input_df = pd.DataFrame({
            '特征': feature_names_chinese,  # 使用中文特征名
            '值': user_input_values
        })

        # 上下顺序显示图表（中文标题）
        st.subheader("特征重要性")
        fig_importance = px.bar(
            importance_df,
            x='重要性',
            y='特征',
            orientation='h',
            title='特征重要性',
            labels={'重要性': '重要性', '特征': '特征'},
            color='重要性',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        st.subheader("预测概率分布")
        prob_df = pd.DataFrame({
            '类别': ['低风险', '中度风险', '高风险'],
            '概率': [probability[0][0], 0, prob_depression] if risk_level == "低风险" else
                   [0, probability[0][1], prob_depression] if risk_level == "中度风险" else
                   [0, 0, probability[0][1]]
        })
        fig_probability = px.bar(
            prob_df,
            x='类别',
            y='概率',
            title='预测概率分布',
            labels={'概率': '概率', '类别': '类别'},
            color='类别',
            color_discrete_map={'低风险': 'blue', '中度风险': 'orange', '高风险': 'red'}
        )
        st.plotly_chart(fig_probability, use_container_width=True)

        st.subheader("您的输入数据")
        fig_user_input = px.bar(
            user_input_df,
            x='值',
            y='特征',
            orientation='h',
            title='您的输入数据',
            labels={'值': '值', '特征': '特征'},
            color='值',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_user_input, use_container_width=True)

    except ValueError:
        st.error("请确保所有输入都是有效数字！")
    except Exception as e:
        st.error(f"错误: {e}")
