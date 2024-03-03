import json
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import sklearn
import matplotlib

matplotlib.use("Agg")

class Web:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        st.title("Classifier Ads App")
        st.header("Example Web App with Streamlit")
        st.markdown("""
            #### Description
            + This is an example Data Analysis and Implementation Machine learning of the Ads Dataset 
              depicting the various species built with Streamlit.

            #### Purpose
            + To show an example ML of Ads using the Streamlit framework.
    	""")

    def clean_text(self, string) -> None:
        # Lower case the string
        string = string.lower()

        # Remove undesirable characters
        special_chars = ['-', '/', '*', '_', '(', ')', '!', '|', ',', ';', '+', '@']
        for char in special_chars:
            string = string.replace(char, ' ')

        return string

    def load_and_clean_data(self, data) -> pd.DataFrame:
        content = data.getvalue().decode("utf-8")
        test = [json.loads(line) for i, line in enumerate(content.strip().split('\n')) if i > 0]
        df = pd.DataFrame(test)
        df["text"] = df["section"].map(str) + " " + df["city"].map(str) + " " + df["heading"].map(str)
        df["text"] = df["text"].apply(self.clean_text)
        return df

    def display_data_checks(self, df) -> None:
        # Display data checks (shape, columns, null data, duplicate data, description, value counts)
        if st.checkbox("Show Shape"):
            st.write(df.shape)

        if st.checkbox("Show Columns"):
            st.write(df.columns.to_list())

        if st.checkbox("Show Null Data"):
            st.write(df.isnull().sum())

        if st.checkbox("Show Duplicate Data"):
            st.dataframe(df[df.duplicated()])

        if st.checkbox("Description Data"):
            st.write(df.describe())

        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:, -1].value_counts())

    def display_section_counts_bar_plot(self, df) -> None:
        if st.checkbox("Section Counts Bar Plot"):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.title('Section Count Plot')
            st.write(df['section'].value_counts().plot(kind='bar'))
            st.pyplot()

    def display_result(self, x, model) -> None:
        prediction = model.predict(x)
        prediction_df = pd.DataFrame({'Prediction': prediction})

        st.subheader('Result')
        st.write(prediction_df)

    def ml(self, data) -> None:
        st.header("Data Analysis and Prediction")
        if data is None:
            st.write("Data Not Found or Data Not JSON!")
        else:
            df = self.load_and_clean_data(data)

            st.write(df.head())
            self.display_data_checks(df)
            self.display_section_counts_bar_plot(df)

            st.header("Predictor")
            X = df.text.values
            model = joblib.load("model.sav")
            self.display_result(X, model)

    def main(self) -> None:
        data = st.file_uploader("Upload Data", type=["json"])
        self.ml(data)

if __name__ == '__main__':
    app = Web()
    app.main()
