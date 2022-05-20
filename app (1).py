import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


data3 = pd.read_csv('Quotation_dataset3.csv')

def preprocess(data):
    data['Sales order date'] = pd.to_datetime(data['Sales order date'],dayfirst = True)
    data['Date'] = pd.to_datetime(data['Date'],dayfirst = True)
    data['Quote Date']= pd.to_datetime(data['Quote Date'],dayfirst = True)
    data['Quote Date - So Date Difference'] = (data['Sales order date'] - data['Quote Date']).dt.days
    data['Age of Customer(days)'] = (data['Date'] - data['Sales order date']).dt.days
    data['Age of Customer(days)']=data['Age of Customer(days)'].fillna(0)
    for i in data['Customer ID']:
        a=list(data.loc[data['Customer ID']==i,'Age of Customer(days)'].values)
        data.loc[data['Customer ID']==i,'Age of Customer(days)']=(max(a))
    data['Conversion Ratio'] = 0.0
    data['Age of Customer (years)'] = round((data['Age of Customer(days)']/365),2)
    for i in data['Customer ID']:
        a=list(data.loc[data['Customer ID']==i,'Quote Amount'])
        l = list(data.loc[data['Customer ID']==i,'Sales Order Amount'])
        s = 0
        c= 0
        for j in l:

            if pd.isnull(j)==False:
                s = s+j
                c = c+1
        if c==0:
            m = 0 
        else:
            m = s

        d=sum(a)/len(a)
        data.loc[data['Customer ID']==i,'TotalSales']= m 
        rate=c/len(a)
        data.loc[data['Customer ID']==i,'Conversion Ratio']= round((rate*100),2)
    data['Avg Days'] = 0.0
    for i in data['Customer ID']:
        b = list(data.loc[data['Customer ID']==i,'Quote Date - So Date Difference'])
        s = 0
        c= 0
        for j in b:
            if pd.isnull(j)==False:
                s = s+j
                c = c+1
        if c==0:
            m = 0 
        else:
            m = s/c

        data.loc[data['Customer ID']==i,'Avg Days']= m
    data2 = data.drop_duplicates(subset=['Customer ID'],keep = 'last')
    from sklearn.preprocessing import MinMaxScaler
    data2[['Conversion Ratio','Age of Customer (years)','TotalSales','Avg Days']] = MinMaxScaler().fit(data2[['Conversion Ratio','Age of Customer (years)','TotalSales','Avg Days']]).transform(data2[['Conversion Ratio','Age of Customer (years)','TotalSales','Avg Days']])
    return data2


    

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    customers = [x for x in request.form.values()]
    df=pd.read_csv(request.form['csvfile'])
    #df=pd.DataFrame(customers,columns=['Customer ID'])
    data5=data3.merge(df, left_on='Customer ID', right_on='Customer ID')
    data5=preprocess(data5)
    X_R1=data5[['Age of Customer (years)','Conversion Ratio', 'TotalSales', 'Avg Days']]
    data5['Rating']=model.predict(X_R1)
    df_out=data5.sort_values(['Rating'],ascending=False)
    df_out=df_out[["Customer ID","Rating"]]
    df_out = df_out.reset_index()
    df_out.drop(columns = ['index'],inplace = True)
    df_out ['Rank']= df_out.index+1
    #df_out['Rating']=df_out['Rating']+df_out['Request Age']
    
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    
    

    #return render_template('index.html', prediction_text='Customer Ranking {}'.format(output))
    #return render_template('index.html', prediction_text='Customer Ranking {}'.format(out1))
    return df_out.to_html()


if __name__ == "__main__":
    app.run(debug=True)