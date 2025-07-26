#  Employee Salary Prediction App

This is a web-based application built using **Streamlit** for predicting whether an individual's income exceeds \$50K/year based on various demographic attributes.  


🔗 **Live Demo**: [Click here to use the app](https://ibm---employeesalaryprediction-vgvey7sp2kwrxxutgjrowy.streamlit.app/)  


---

## 🚀 Overview

The app collects user inputs through an interactive sidebar and uses those inputs to predict the likelihood of an individual earning more than \$50K per year.  
It supports real-time interaction and is accessible through any web browser — no installations required.

---

## 🧠 Features

### 📝 Input Fields

The app provides the following fields in the sidebar:

#### 🔹 Categorical Fields
- **Workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked  
- **Education**: Bachelors, HS-grad, 11th, Masters, 9th, Some-college, Assoc-acdm, Assoc-voc, 7th-8th, Doctorate, 5th-6th, 10th, 1st-4th, Preschool, 12th  
- **Marital Status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse  
- **Occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces  
- **Relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried  
- **Race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black  
- **Sex**: Male, Female  
- **Native Country**: United-States, India, Mexico, Philippines, Germany, Canada, England, Cuba, Iran, China, France, Puerto-Rico, Jamaica, Vietnam, Japan, Italy, Greece, Columbia, Thailand, Ecuador, Poland, Honduras, Ireland, Hungary, Scotland, Guatemala, Nicaragua, Trinadad&Tobago, Laos, Taiwan, Haiti, Hong, South, Yugoslavia, El-Salvador, Dominican-Republic, Portugal, Outlying-US(Guam-USVI-etc), Cambodia, Holand-Netherlands, Peru  

#### 🔸 Numerical Fields
- **Age**: 17 – 90 (Slider)
- **Fnlwgt**: 10,000 – 1,000,000 (Input field)
- **Education Number**: 1 – 16 (Slider)
- **Hours per Week**: 1 – 99 (Slider)

---

## 🧾 Output

Once all fields are filled, the app will generate a prediction output such as:

> 🔍 **Prediction: Income > \$50K/year**  
> or  
> 🔍 **Prediction: Income ≤ \$50K/year**

---

## 🛠️ Tech Stack

| Tool         | Purpose                      |
|--------------|------------------------------|
| Streamlit    | Web application framework    |
| Python       | Core language for app logic  |
| scikit-learn / joblib | (Optional) Model loading and prediction |



