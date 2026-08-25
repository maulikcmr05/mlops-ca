import stream as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="*",
    layout="wide",
)

st.title("Flower Speces classifier")
st.write("Enter Flower measurements to predict the data")
st.write(f"Accuracy: {accuracy}")

st.divider()

sepal_height=st.number_input("sepal height (cm)"),min_value=0.0,max_value=10.0,value=5.1
sepal_width=st.number_input("sepal height (cm)"),min_value=0.0,max_value=10.0,value=3.4
petal_height=st.number_input("Enter the petal height"),min_value=0.0,max_value=10.0,value=1.4
petal_width=st.number_input("Enter the petal width"),min_value=0.0,max_value=10.0,value=0.2

if st.button("Predict"):
    prediction = model.predict([[sepal_height, sepal_width, petal_height, petal_width]])
    probability = model.predict_proba([[sepal_height, sepal_width, petal_height, petal_width]])
    species = iris.target_names[prediction][0]
    st.success(f"Predicted : {species.upper()}")
    st.subheader("Prediction Confidence")
    st.write( {
            iris.target_names[i]: f"{probability[0][i]*100:.2f}%"
            for i in range(len(iris.target_names))
        })
    st.progress(float(max(probability[0])))
st.divider()
st.cpation("Devloped Streamlit")