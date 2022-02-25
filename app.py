
# coding: utf-8

# In[20]:


from flask import Flask


# In[21]:


app = Flask(__name__)


# In[22]:


from flask import request, render_template

@app.route("/",methods = ["Get","POST"])
def index():
    if request.method == "POST":
        income = float(request.form.get("income"))
        age = float(request.form.get("age"))
        loan = float(request.form.get("loan"))
        
        import joblib
        import numpy as np

        #import different models
        log_model = joblib.load("log_model")
        cart_model = joblib.load("cart_model")
        rf_model = joblib.load("rf_model")
        xgb_model = joblib.load("XGB_model")
        mlp_model = joblib.load("MLP_model")
        
        #Normalize the data for MLP model
        norm_income = (income - 20014.48947)/(69995.685580 - 20014.48947)
        norm_age = (age-18.055189)/(63.971796 - 18.055189)
        norm_loan = (loan - 1.377630)/(13766.051240 - 1.377630)
        
        #Predicted values based on different model
        log_pred = log_model.predict([[income,age,loan]])
        cart_pred = cart_model.predict([[income,age,loan]])
        rf_pred = rf_model.predict([[income,age,loan]])
        xgb_pred = xgb_model.predict([[income,age,loan]])
        mlp_pred = mlp_model.predict([[norm_income,norm_age,norm_loan]])
        
        log_pred1 = 'No' if float(log_pred[0])<0.5 else 'Yes'
        cart_pred1 = 'No' if float(cart_pred[0])<0.5 else 'Yes'
        rf_pred1 = 'No' if float(rf_pred[0])<0.5 else 'Yes'
        xgb_pred1 = 'No' if float(xgb_pred[0])<0.5 else 'Yes'
        mlp_pred1 = 'No' if float(mlp_pred[0])<0.5 else 'Yes'
        
        s = f"""
Will the person Default in payment: \n
1. Log Reg: {log_pred1} \n
2. CART: {cart_pred1} \n
3. Random Forest: {rf_pred1} \n 
4. XGBoost: {xgb_pred1} \n
5. MLP: {mlp_pred1} """
        return (render_template("index.html", result = s))
    else:
        return (render_template("index.html", result = "Please Enter the values"))


# In[23]:


if __name__ == "__main__":
    app.run()

