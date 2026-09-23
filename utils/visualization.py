import plotly.express as px

def risk_distribution(df):
    return px.histogram(df,x="Probability",nbins=20,title="Predicted Risk Distribution")
