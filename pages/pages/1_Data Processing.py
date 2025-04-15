import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st
import numpy as np

st.set_page_config(page_title="Data Processing", page_icon="🔧")
st.markdown("# Data Processing")
st.subheader("Please make sure your dataset do not contain Object and datetime datatype")
st.sidebar.header("Data Processing")

# File uploader
upload_file=st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None:
    uploaded_filename = upload_file.name
    if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_filename:
        st.success("File uploaded successfully!")
        data = pd.read_csv(upload_file)
        
        head=data.head()
        preview=data.describe()
        column_info = pd.DataFrame({

        'Column Name': data.columns,
        'Non-Null Count': data.count(),
        'Data Type': data.dtypes,
        })
# data info    
        with st.expander ("**Data Preview**"):
            st.dataframe(head)
            st.dataframe(preview)
            st.dataframe(column_info,hide_index=True)

#for missing values and outliers
        #types=data.dtypes.astype(str).to_list()
        missing_columns = data.columns[data.isna().any()].tolist()
        count=data.isna().sum().sum()
        cols=data.select_dtypes(include=np.number).astype(str).columns 
        outliers_col_names=[]
        outlierscount=0
        for columns in cols:
            Q1 = data[columns].quantile(0.25)
            Q3 = data[columns].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[columns] < lower_bound) | (data[columns] > upper_bound)]
            if not outliers.empty:
                outliers_col_names.append(columns)
                outlierscount+=1

        cpydata=data.copy()
        if outlierscount>0 and count>0:
            st.warning(f"This dataset contains outliers and missing values. The columns with outliers are {outliers_col_names} and columns with missing values are {missing_columns}")
            if st.button("Treat"):

                for col in missing_columns:
                    imp=SimpleImputer(strategy='Median')
                    if cpydata[col].dtype in ['float64', 'int64']:
                        cpydata[col]=imp.fit_transform(cpydata[col])
                    else:
                        cpydata[col].fillna(cpydata[col].mode()[0], inplace=True)  

                for col in outliers_col_names:
                    q1 = cpydata[col].quantile(0.25)
                    q3 = cpydata[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    cpydata[col] = np.where(cpydata[col] < lower, lower, np.where(cpydata[col] > upper, upper, cpydata[col]))      
                st.success("✅ Data cleaned using median method for missing and flooring and capping method for outliers.")
                csv = cpydata.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
        elif outlierscount>0 and count==0:
            st.warning(f"This dataset contains outlier values. The columns with outliers are {outliers_col_names}")
            if st.button("Treat"):
                for col in outliers_col_names:
                    q1 = cpydata[col].quantile(0.25)
                    q3 = cpydata[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    cpydata[col] = np.where(cpydata[col] < lower, lower, np.where(cpydata[col] > upper, upper, cpydata[col]))      
                st.success("✅ Data cleaned using flooring and capping method for outliers.")
                csv = cpydata.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
        elif outlierscount==0 and count>0:
            st.warning(f"This dataset contains missing values. The columns with missing values are {missing_columns}")
            if st.button("Treat"):
                for col in missing_columns:
                    imp=SimpleImputer(strategy='Median')
                    if cpydata[col].dtype in ['float64', 'int64']:
                        cpydata[col]=imp.fit_transform(cpydata[col])
                    else:
                        cpydata[col].fillna(cpydata[col].mode()[0], inplace=True)
                st.success("✅ Data cleaned using median method(missing).")
                csv = cpydata.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            
        else:
            st.success("This dataset do not need any processing")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="data.csv",
                mime="text/csv")
        

        
        





