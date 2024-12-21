import streamlit as st
from PIL import Image

from model_template import ModelTemplate

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    models = [ModelTemplate()]
    model_names = list()

    for model in models:
        model.load_model()
        model_names.append(model.name)


    st.title('Plant Disease Classification')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    option = st.selectbox(
        "What model do you like to use?",
        (model_names)
    )

    if uploaded_file is not None and option is not None:

        selected_model = models[model_names.index(option)]

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = selected_model.predict(image)
        st.write(prediction)