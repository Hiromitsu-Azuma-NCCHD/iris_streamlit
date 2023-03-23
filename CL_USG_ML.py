# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
#import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from PIL import Image

# Webサイトのタイトルと表示されるテキストを記入
st.title('CL_USG_classifier')
st.write('超音波画像をアップロードしてください')


# 画像の読み込み
# img = Image.open('iris.jpg')
#st.image(img,caption = 'Iris' , use_column_width = True)
#st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


#アップロードされた超音波写真を用いて推論
def main():
    st.title('CL_USG_ML')
    model = load model('CervixUSGmodel.pt')
    upload_file =st.file_uploader('経腟超音波画像を選んでください',type["jpg","png","jpeg"])
    
    if uploaded_file is None:
        return
    
    img = Image.open(uploaded_file)
    st.image(img)
    tensor = img2tensor(img)
    with torch.no_grad():
        model.eval()
        output = model(tensor)

    pred_df = annotate_labels(output)

    fig, ax = plt.subplots(figsize=(6,6))
    pred_df.head(20).iloc[::-1].plot(kind="barh", ax=ax)
    ax.grid()
    st.markdown("## result")
    st.pyplot(fig)

    st.markdown("## Predict Detail")
    st.dataframe(data=pred_df)

























