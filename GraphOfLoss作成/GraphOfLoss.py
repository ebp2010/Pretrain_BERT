import plotly.express as px  
import pandas as pd 
import json

# Opening JSON file
f = open('./trainer_state.json')
# f = open('D:/PYTHON_code/LIBRARY/GraphOfLoss/trainer_state.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)

log_history = data['log_history']

#Type change list to dataframe
df = pd.DataFrame(log_history)

x = df['step']
y1 = df['loss']
y2 = df['learning_rate']

lossDF = pd.concat([x, y1], axis=1)
learning_rateDF = pd.concat([x, y2], axis=1)

# Set figure
fig1 = px.line(lossDF, x="step", y="loss", title="Loss") 
fig1.update_traces(line_color='orange')

fig2 = px.line(learning_rateDF, x="step", y="learning_rate", title="Learning Rate") 
fig2.update_traces(line_color='orange')


# Save figure
fig1.write_image('./Step_Loss.png', width=1000, height=1000)
fig2.write_image('./Step_Learning_rate.png', width=1000, height=1000)


