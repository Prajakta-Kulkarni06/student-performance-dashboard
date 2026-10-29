import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

st.set_page_config(
    page_title="StudentInsight | Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    bundle = joblib.load(
        os.path.join(BASE, "student_models.joblib")
    )
    neural_net = tf.keras.models.load_model(
        os.path.join(BASE, "student_ann.keras")
    )
    data = pd.read_csv(
        os.path.join(BASE, "student_data.csv")
    )
    metrics = pd.read_csv(
        os.path.join(BASE, "model_results.csv")
    )
    return bundle, neural_net, data, metrics

bundle, ann, df, results = load_assets()
preprocessor = bundle["preprocessor"]
models = bundle["models"]
features = bundle["feature_names"]

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
[data-testid="stSidebar"] {
    background-color: #14243b;
}
[data-testid="stSidebar"] * {
    color: white;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e4eaf2;
}
h1, h2, h3 {
    color: #193452;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🎓 StudentInsight")
st.sidebar.caption("Academic Performance Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Explorer", "Model Evaluation",
     "Predict Performance", "About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("ML + Deep Learning Analytics")

if page == "Dashboard":
    st.title("Student Performance Dashboard")
    st.write(
        "Explore academic patterns and model-based score predictions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student Records", f"{len(df):,}")
    c2.metric("Available Features", len(features))
    c3.metric("Average Final Grade", f"{df['G3'].mean():.2f}/20")
    c4.metric(
        "Pass Rate (G3 ≥ 10)",
        f"{(df['G3'] >= 10).mean() * 100:.1f}%"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Final Grade Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["G3"], bins=15, kde=True, ax=ax)
        ax.set_xlabel("Final grade (0–20)")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Study Time vs Grade")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="studytime", y="G3", ax=ax)
        ax.set_xlabel("Study time category")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.write("Filter records and explore the dataset.")

    if "school" in df.columns:
        schools = ["All"] + sorted(df["school"].dropna().unique().tolist())
        selected_school = st.selectbox("School", schools)
        view = df if selected_school == "All" else df[
            df["school"] == selected_school
        ]
    else:
        view = df

    st.dataframe(view, use_container_width=True)

    st.subheader("Numerical Correlations")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        view.select_dtypes(include="number").corr(),
        cmap="coolwarm", center=0, ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download filtered data as CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="student_data.csv",
        mime="text/csv"
    )

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Metrics computed on the held-out test set in Colab.")

    st.dataframe(
        results.sort_values("RMSE").round(3),
        hide_index=True,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=results.sort_values("RMSE"),
        x="RMSE", y="Model", ax=ax
    )
    ax.set_title("RMSE Comparison — Lower Is Better")
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "MAE measures average absolute error. RMSE penalizes larger errors. "
        "R² compares predictions with a mean-baseline model."
    )

elif page == "Predict Performance":
    st.title("Student Performance Prediction")
    st.write(
        "Enter the available student characteristics to estimate "
        "a final grade on the 0–20 scale."
    )

    with st.form("prediction_form"):
        values = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            series = df[feature]
            parent = col1 if i % 2 == 0 else col2

            with parent:
                if pd.api.types.is_numeric_dtype(series):
                    low = float(series.min())
                    high = float(series.max())
                    default = float(series.median())

                    if low == high:
                        values[feature] = low
                    elif pd.api.types.is_integer_dtype(series):
                        values[feature] = st.number_input(
                            feature,
                            min_value=int(low),
                            max_value=int(high),
                            value=int(round(default)),
                            step=1
                        )
                    else:
                        values[feature] = st.number_input(
                            feature,
                            min_value=low,
                            max_value=high,
                            value=default
                        )
                else:
                    options = sorted(
                        series.dropna().astype(str).unique().tolist()
                    )
                    values[feature] = st.selectbox(feature, options)

        selected_model = st.selectbox(
            "Prediction algorithm",
            list(models.keys()) + ["Deep Learning ANN"]
        )
        submitted = st.form_submit_button(
            "Predict Final Grade",
            type="primary"
        )

    if submitted:
        row = pd.DataFrame([values], columns=features)
        processed = preprocessor.transform(row)

        if selected_model == "Deep Learning ANN":
            processed = np.asarray(
                processed.toarray()
                if hasattr(processed, "toarray")
                else processed,
                dtype=np.float32
            )
            prediction = float(
                ann.predict(processed, verbose=0)[0][0]
            )
        else:
            prediction = float(
                models[selected_model].predict(processed)[0]
            )

        prediction = float(np.clip(prediction, 0, 20))

        st.subheader("Estimated Result")
        st.metric("Predicted Final Grade", f"{prediction:.2f} / 20")

        if prediction < 10:
            st.warning(
                "The estimate is below 10/20. Consider reviewing "
                "the student's learning needs with a teacher."
            )
        else:
            st.success(
                "The estimate is at or above 10/20."
            )

        st.caption(
            "This is an experimental estimate, not an official grade "
            "or a definitive assessment of a student's ability."
        )

elif page == "About":
    st.title("About StudentInsight")
    st.write("""
    StudentInsight is an educational analytics demonstration built
    using Python, Pandas, Scikit-learn, TensorFlow, and Streamlit.

    **Project capabilities**
    - Student dataset exploration
    - Machine Learning regression
    - Artificial Neural Network prediction
    - Model comparison and visualization
    - Interactive score estimation

    **Dataset:** UCI Student Performance dataset.

    **Responsible use:** Predictions may be inaccurate and can reflect
    limitations in the training data. Use them only as supplementary
    information alongside professional educational judgment.
    """)
