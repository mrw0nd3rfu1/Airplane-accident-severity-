# Airplane-accident-severity-
This was a problem in hackerearth competition that I have entered
With the above dataset as we can see the only option is to do regression or do Random Forest
So I attempted on random forest as it works better that regression for total accuracy(may take longer to process) but because of the low size of dataset I chose this.
I selected multiple random forest models with different arguments in each of them and trying 1 by 1 to get minimum mae score.
Some models I used are-

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=1)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)


I end up chosing the 'model' as it yeilded minimum mae score and r2 score of 81%.


As the predicted output was categorical I changed the output to numerical with LabelEncoder, there is another option of one hot encoding but I didn't tried it.
I also removed the data if it has any null value in the row to remove error.

In score_dataset method I just fitted the model with all the training and testing data.
I also converted the preds numpy array to list so that I can change the float values to string as the output required was in string.
the code for it-

 preds = preds.tolist()
    for x in range(len(preds)):
        preds[x]= round(int(preds[x]))
        if preds[x]==0:
            preds[x]='Highly_Fatal_And_Damaging'
        elif preds[x]==3:
            preds[x]= 'Significant_Damage_And_Serious_Injuries'
        elif preds[x]==1:
            preds[x]='Minor_Damage_And_Injuries'
        elif preds[x]==2:
            preds[x]='Significant_Damage_And_Fatalities'


Then I just accumulated all the outputs into 'predictions.csv' with Accident_ID and Severity as the features.

For the sources I didn't used any as I did the project from my previous experience.

Tools used-
scikit-sklearn
numpy
pandas


You can check my airport.py file to check the code.
