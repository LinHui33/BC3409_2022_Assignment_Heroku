
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
        
        #Predicted values based on different model
        log_pred = log_model.predict([[income,age,loan]])
        cart_pred = cart_model.predict([[income,age,loan]])
        rf_pred = rf_model.predict([[income,age,loan]])
        xgb_pred = xgb_model.predict([[income,age,loan]])
        
        #Normalize the data for MLP model
        norm_income = (income - 20014.48947)/(69995.685580 - 20014.48947)
        norm_age = (age-18.055189)/(63.971796 - 18.055189)
        norm_loan = (loan - 1.377630)/(13766.051240 - 1.377630)
        
        #Preditions
        log_pred = log_model.predict([[income,age,loan]])
        cart_pred = cart_model.predict([[income,age,loan]])
        rf_pred = rf_model.predict([[income,age,loan]])
        xgb_pred = xgb_model.predict([[income,age,loan]])
        mlp_pred = mlp_model.predict([[norm_income,norm_age,norm_loan]])

        #result1 = pd.DataFrame({'Model':['Log','CART','RF','XGB',"MLP"],'Outcome':[float(log_pred[0]),float(cart_pred[0]),float(rf_pred[0]),float(xgb_pred[0]),float(mlp_pred[0])]})

        #result1['Default?'] = np.where(result1.Outcome>0.5,'Default', 'Not Default')
        s = "Log = " + str(log_pred) + ', CART = ' + str(cart_pred) + ', RF = ' + str(rf_pred) + ', xgb = ' + str(xgb_pred) + ', mlp = ' + str(mlp_pred)
        #s = f"""
 #The predicted Outcome for various models are: 
 #1. The outcome for Log Reg is {result1['Default?'][0]}
# 2. The outcome for CART is {result1['Default?'][1]}
# 3. The outcome for Random Forest is {result1['Default?'][2]}
# 4. The outcome for XGBoost is {result1['Default?'][3]}
# 5. The outcome for MLP is {result1['Default?'][4]} """
        return (render_template("index.html", result = s))
    else:
        return (render_template("index.html", result = "Please Enter the values"))


# In[23]:


if __name__ == "__main__":
    app.run()

