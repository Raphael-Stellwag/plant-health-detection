import streamlit as st
from PIL import Image

from ResNet9 import ResNet9
from model_ResNet import ResNetModel
from model_cnn import ModelCNN
from model_template import ModelTemplate

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    last_selected_model: ModelTemplate = ModelTemplate()
    last_selected_model.load_model()

    models = [ModelTemplate(), ModelCNN(), ResNetModel()]
    model_names = list()

    for model in models:
        model_names.append(model.name)

    st.title('Plant Disease Classification')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    option = st.selectbox(
        "What model do you like to use?",
        (model_names)
    )

    if uploaded_file is not None and option is not None:

        selected_model = models[model_names.index(option)]

        if last_selected_model != selected_model:
            last_selected_model.unload_model()
            selected_model.load_model()
            last_selected_model = selected_model

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = selected_model.predict(image)
        st.write(prediction)