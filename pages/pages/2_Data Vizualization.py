import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Vizualization", page_icon="📈")

st.markdown("# Data Vizualization")
st.sidebar.header("Data Vizualization")

upload_file=st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None:
    uploaded_filename = upload_file.name
    if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_filename:
        st.success("File uploaded successfully!")
        df = pd.read_csv(upload_file)

        visualize_all = st.checkbox("📈 Visualize All Columns")

        if visualize_all:
            st.subheader("📊 Pairplot (Seaborn)")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for pairplot.")
            else:
                fig = sns.pairplot(df[numeric_cols])
                st.pyplot(fig)
            
            st.subheader("📄 Descriptive Statistics")
            st.dataframe(df.describe())
        else:
            # Select specific columns and chart type
            columns = st.multiselect("Select columns to visualize", df.columns.tolist())
            chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Line", "Bar", "Scatter"])

            if st.button("📊 Show Visualization"):
                if not columns:
                    st.warning("Please select at least one column.")
                else:
                    st.subheader(f"{chart_type} for selected columns")
                    
                    fig, ax = plt.subplots()

                    if chart_type == "Histogram":
                        for col in columns:
                            sns.histplot(df[col], kde=True, ax=ax, label=col)
                        ax.legend()
                    elif chart_type == "Box Plot":
                        sns.boxplot(data=df[columns], ax=ax)
                    elif chart_type == "Line":
                        if len(columns) != 2:
                            st.warning("Scatter plot needs exactly 2 columns (x and y).")
                        else:
                            df[columns].plot(kind='line', ax=ax)
                    elif chart_type == "Bar":
                        df[columns].sum().plot(kind='bar', ax=ax)
                    elif chart_type == "Scatter":
                        if len(columns) != 2:
                            st.warning("Scatter plot needs exactly 2 columns (x and y).")
                        else:
                            sns.scatterplot(data=df, x=columns[0], y=columns[1], ax=ax)

                    st.pyplot(fig)
