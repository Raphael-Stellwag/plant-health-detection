import streamlit as st
from PIL import Image

from model_template import ModelTemplate

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    selected_model = ModelTemplate()

    st.title('Plant Disease Classification')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = selected_model.predict(image)
        st.write(prediction)