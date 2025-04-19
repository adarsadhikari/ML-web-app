import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np

st.set_page_config(page_title="Data Processing", page_icon="🔧")
st.markdown("# Data Processing")
st.sidebar.header("Data Processing")

# File uploader
upload_file=st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None:
    data = pd.read_csv(upload_file)
    uploaded_filename = upload_file.name
    if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_filename:
        st.success("File uploaded successfully!")
        #data = pd.read_csv(upload_file)
        
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

#for object
        cnt=0
        object_cols = data.select_dtypes(include='object').columns.astype(str).tolist()

        if object_cols:
            st.warning(f"The dataset contains object datatype {object_cols}")
            st.subheader("Choose Conversions for object column")
            for col in object_cols:
                st.selectbox(
                    f"Convert '{col}' to:",
                    ["numeric", "datetime", "category"],
                    key=f"objectconvert_{col}"
                )
        else:
            st.info("✅ No object columns detected.")
            cnt+=1
        
        def convert_object_columns(df):
            object_cols = df.select_dtypes(include='object').columns.astype(str).tolist()
            df_copy = df.copy()
            le=LabelEncoder()

            for col in object_cols:
                
                target_type = st.session_state.get(f"objectconvert_{col}", "None")
                try:
                    if target_type == "numeric":
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype("float64")
                    elif target_type == "datetime":
                        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    elif target_type == "category":
                        df_copy[col] = le.fit_transform(df_copy[col])
                except Exception as e:
                    st.warning(f"Could not convert {col} to {target_type}: {e}")

            return df_copy


#for missing values and outliers
        #types=data.dtypes.astype(str).to_list()
        missing_columns = data.select_dtypes(include=['number']).columns[data.select_dtypes(include=['number']).isna().any()].astype(str).tolist()
        if missing_columns:
            st.warning(f"The dataset contains missing values. The columns with missing values are {missing_columns}")
            st.subheader("Choose Conversions for missing values")
            for col in missing_columns:
                st.selectbox(
                    f"Convert '{col}' to:",
                    ["mean", "median", "mode"],
                    key=f"missingconvert_{col}"
                )
        else:
            st.info("✅ No missng values detected.")
            cnt+=1
        
        count=data.isna().sum().sum()
        
        def convert_missing_columns(df):
            missing_columns = df.columns[df.isna().any()].astype(str).tolist()
            df_copy = df.copy()
            s1=SimpleImputer(strategy="median")
            s2=SimpleImputer(strategy="mean")
            s3=SimpleImputer(strategy="most_frequent")

            for col in missing_columns:
                target_type = st.session_state.get(f"missingconvert_{col}", "None")
              
                try:
                    if target_type == "mean":
                        df_copy[col] = s2.fit_transform(df_copy[[col]])
                    elif target_type == "median":
                        df_copy[col] = s1.fit_transform(df_copy[[col]])
                    elif target_type == "mode":
                        df_copy[col] = s3.fit_transform(df_copy[[col]])
                   
                except Exception as e:
                    st.warning(f"Could not convert {col} to {target_type}: {e}")

            return df_copy
        
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
        
        if outliers_col_names:
            st.warning(f"The dataset contains outliers. The columns with outliers are {outliers_col_names}")
            st.subheader("Choose Conversions for outliers values")
            for col in outliers_col_names:
                key = f"outliersconvert_{col}"
                if key not in st.session_state:
                    st.session_state[key] = "None"
                
                st.selectbox(
                    f"Convert '{col}' to:",
                    ["None","IQR"],
                    key=key
                )
        else:
            st.info("✅ No outliers values detected.")
            cnt+=1

        def handle_outliers(df):
            #cols=data.select_dtypes(include=np.number).astype(str).columns 
            df_copy = df.copy() 
          
            for coln in outliers_col_names:
                
                target_type = st.session_state.get(key, "None")
                
                try:
                    if target_type=="IQR":
                        q1 = df_copy[coln].quantile(0.25)
                        q3 = df_copy[coln].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        df_copy[coln] = np.where(df_copy[coln] < lower, lower, np.where(df_copy[coln] > upper, upper, df_copy[coln]))
                except Exception as e:
                    st.warning(f"Could not convert {coln} to {target_type}: {e}")
            
            return df_copy
            
        
        if st.button("🧪 Treat Data"):
            if cnt==3:
                st.success("This dataset do not need any processing")
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name="data.csv",
                    mime="text/csv")
            else:
                df_clean = convert_object_columns(data)
                df_clean = convert_missing_columns(df_clean)
                df_clean = handle_outliers(df_clean)
                

                st.success("✅ Data successfully treated!")
                st.warning("Please recheck this dataset to confirm the dataset is cleaned")

                st.download_button(
                    "📥 Download Cleaned CSV",
                    df_clean.to_csv(index=False),
                    "cleaned_data.csv",
                    "text/csv"
                )
        
       