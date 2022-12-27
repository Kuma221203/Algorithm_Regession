import streamlit as st
import pandas as pd
import numpy as np
from time import sleep
from sklearn import preprocessing
st.set_page_config(
    page_title = 'LoylP',
    page_icon = '🤘',
)
#Sidebar
st.sidebar.success("# Welcome to ...", icon = "👋")
st.sidebar.markdown('----')
st.sidebar.success('# Show dataframe từ file csv', icon = "📄")
st.sidebar.markdown('----')
st.sidebar.success('# Choose input feature', icon = "📩")
st.sidebar.markdown('----')
st.sidebar.success('# Choose Algorithm', icon = "🧮")
st.sidebar.markdown('## Decision tree')
st.sidebar.markdown('## Linear regression')
st.sidebar.markdown('## XGBoost')
st.sidebar.markdown('----')
st.sidebar.success('# Choose ratio of train/test split', icon = "🎚")
st.sidebar.markdown('----')
st.sidebar.success('# Drawexplicitly chart', icon = "📊")
st.sidebar.markdown('----')
#Title
#Tên chương trình
st.title(' :orange[Chương trình] :green[kiểm tra] :violet[MAE và MSE]')
if "name" not in st.session_state:
    st.session_state["name"] = ""
name = st.text_input('Nhập tên bạn vào đây: ', st.session_state["name"])
if name:
    st.write('### Chào mừng ', name, 'đến với chương trình kiểm tra MSE và MAE')
    sleep(1)
#Thêm file csv
    st.title(" :red[Hãy thêm file csv vào đây]")
    sleep(0.5)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        file = pd.read_csv(uploaded_file)
        st.write(file)
#Chọn input cho bài
        st.title(" :blue[Choose input feature]")
        click1 = st.checkbox('R&D Spend')
        click2 = st.checkbox('Administration')
        click3 = st.checkbox('Marketing Spend')
        click4 = st.checkbox('State')
        del file['Profit']
        if not click1:
            del file['R&D Spend']
        if not click2:
            del file['Administration']
        if not click3:
            del file['Marketing Spend']
        if not click4:
            del file['State']
        my_data = pd.DataFrame(file).to_numpy()
        label_encoder = preprocessing.LabelEncoder()
        my_data[:, -1] = label_encoder.fit_transform(my_data[:, -1])
        st.write(my_data)
#Chọn thuật toán
        sleep(2)
        st.title(" :orange[Choose Algorithm]")
        option = st.selectbox(
            'Hãy chọn thuật toán bạn muốn',
            ('Choose','Decision Tree Regression', 'Linear Regression', 'XGBoost'))
        if option == 'Decision Tree Regression':
            sleep(0.5)
            st.caption('## Bạn đã chọn Decision Tree Regression')
        else:
            if option == 'Linear Regression':
                sleep(0.5)
                st.caption('## Bạn đã chọn Linear Regression')
            else:
                if option == 'XGBoost':
                    sleep(0.5)
                    st.caption('## Bạn đã chọn XGBoost')
#Kéo thanh tỉ lệ
        st.title(" :green[Choose ratio of train/test split]")
        ti_le = st.slider('Chọn tỉ lệ', 0.0, 1.0, 0.1)
        st.write("Bạn đã chọn tỉ lệ là: ",ti_le)
#Show biểu đồ cột
        st.title(" :violet[Drawexplicitly chart]")
