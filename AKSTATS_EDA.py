import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import describe
# Title
st.title("AKSTATS – Exploratory Data Analysis (EDA) App")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File successfully loaded!")

    # Show DataFrame
    st.subheader("Raw Data")
    st.write(df)
  
    from scipy.stats import skew, kurtosis

    def full_summary(df):
        stats = pd.DataFrame()
        stats['Count'] = df.count()
        stats['Mean'] = df.mean()
        stats['Median'] = df.median()
        stats['Mode'] = df.mode().iloc[0]  # if multiple modes, take the first
        stats['Std Dev'] = df.std()
        stats['Variance'] = df.var()
        stats['Min'] = df.min()
        stats['25%'] = df.quantile(0.25)
        stats['50%'] = df.quantile(0.50)
        stats['75%'] = df.quantile(0.75)
        stats['Max'] = df.max()
        stats['Skewness'] = df.apply(skew)
        stats['Kurtosis'] = df.apply(kurtosis)
        stats['Range'] = df.max() - df.min()
        stats['Sum'] = df.sum()
        stats['Unique'] = df.nunique()
        stats['Missing'] = df.isnull().sum()

        return stats.T  # Transpose for better layout
    
    # Basic Info
    st.subheader("Summary Statistics")
    summary_stats = full_summary(df.select_dtypes(include=[np.number]))  # only numeric columns
    st.dataframe(summary_stats) 

    # Let users pick any column for X and one or more for Y
    st.subheader("Select X and Y axes for line plot")

    # All column names
    all_cols = df.columns.tolist()

    # Dropdown for X-axis
    x_axis_col = st.selectbox("Choose X-axis column", options=all_cols)

    # Multiselect for Y-axis (excluding selected X to prevent redundant plotting)
    y_axis_cols = st.multiselect("Choose Y-axis column(s)", options=[col for col in all_cols if col != x_axis_col])

    # Generate plot
    if x_axis_col and y_axis_cols:
        st.subheader("Custom Line Plot")
        fig, ax = plt.subplots()

        for y_col in y_axis_cols:
            ax.plot(df[x_axis_col], df[y_col], label=f"{y_col} vs {x_axis_col}")

        ax.set_xlabel(x_axis_col)
        ax.set_ylabel("Y-axis Values")
        ax.set_title("Line Plot with Custom Axes")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Please select both X-axis and at least one Y-axis column to display the plot.")

    # Drop the first column from the DataFrame
    df_rest = df.iloc[:, 1:]

    # Columns selection (only numeric columns)
    st.subheader("Histogram - To visualize distribution of a column")
    numeric_cols = df_rest.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col_to_plot = st.selectbox("Select column for histogram", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df_rest[col_to_plot], kde=True, ax=ax)
        st.pyplot(fig)

    # Allow user to select columns for correlation analysis
    st.subheader("Select columns for correlation analysis")
    numeric_cols = df_rest.select_dtypes(include="number").columns.tolist()
    selected_cols = st.multiselect("Choose columns", options=numeric_cols, default=numeric_cols)

    # Correlation heatmap
    if selected_cols:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df_rest[selected_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Please select at least two column for correlation analysis.")

    # Conclusion generation function
    def generate_conclusion(df):
        numeric_df = df.select_dtypes(include='number')
        total_rows = len(df)
        total_cols = df.shape[1]
        col_names = numeric_df.columns.tolist()

        # Defensive checks
        most_skewed = numeric_df.skew().abs().idxmax() if not numeric_df.empty else 'N/A'
        most_kurtotic = numeric_df.kurt().abs().idxmax() if not numeric_df.empty else 'N/A'
    
        corr_matrix = numeric_df.corr().abs().unstack()
        corr_matrix = corr_matrix[corr_matrix < 1]  # Remove self-correlation
        most_correlated = corr_matrix.idxmax() if not corr_matrix.empty else ('N/A', 'N/A')

        conclusion = f"""

        ### Exploratory Data Analysis Summary

        Your dataset consists of **:orange[{total_rows}] rows** and **:orange[{total_cols}] columns**, with **:orange[{len(col_names)}] numeric features** are analyzed.

        Key insights derived from the analysis:

        - Central tendencies measured using **Mean, Median, and Mode**.
        - Spread quantified via **Standard Deviation** and **Variance**.
        - Distribution characteristics explored through **Skewness** and **Kurtosis**.

         **:orange[{most_skewed}]** showed the highest skewness, indicating it may be skewed significantly from the normal distribution.

         **:orange[{most_kurtotic}]** had the most pronounced kurtosis, suggesting heavier tails or outlier sensitivity.

         Correlation analysis found **:orange[{most_correlated[0]}]** and **:orange[{most_correlated[1]}]** to be the most strongly associated features.

         Visual tools like histograms and custom line plots helped surface patterns, trends, and anomalies across the dataset.

        _This EDA dashboard provides a strong foundation to guide feature engineering, modeling decisions, or business interpretations. - AK_
        """
        st.markdown(conclusion)
   
    st.subheader("Conclusion")
    generate_conclusion(df)

else:
    st.warning("Please upload a CSV file to get started.")