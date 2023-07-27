import numpy as np
from scipy.stats import gamma

def normalize(ws, m=0.5):
    # Split Time column into two columns, Date and Time
    ws[['Date', 'Time']] = ws['Time'].str.split(' ', n=1, expand=True)
    # Rename columns
    ws          = ws.rename(columns={'WindSpeed': 'WindSpeed_orig'})
    # Calculate V_prime
    ws['V_prime'] = ws['WindSpeed_orig'] ** m
    # Calculate V_prime_bar and s for each unique hour
    V_prime_bar = ws.groupby('Time')['V_prime'].mean()
    s           = ws.groupby('Time')['V_prime'].std(ddof=0)
    # Calculate NormalizedWindspeed
    ws['NormalizedWindspeed'] = (ws['V_prime'] - V_prime_bar[ws['Time']].values) / s[ws['Time']].values
    return ws

def unnormalize(data, time, refdata, m=0.5):
    str_list    = np.char.split(refdata['Time'], sep=" ", maxsplit=1)
    refdata     = np.concatenate((str_list.tolist(), refdata.iloc[:, 1:3].values), axis=1)
    refdata     = pd.DataFrame(refdata, columns=["Date", "Time", "WindSpeed"])
    days        = len(np.unique(refdata.iloc[:, 0]))
    hour        = np.unique(refdata.iloc[:, 1])
    V_prime     = np.power(refdata.iloc[:, 2], m)
    refdata     = pd.concat([refdata, pd.DataFrame({'V_prime': V_prime})], axis=1)
    V_prime_bar = [np.mean(refdata['V_prime'][np.where(refdata['Time']==h)[0]]) for h in hour]
    s           = [np.std(refdata['V_prime'][np.where(refdata['Time']==h)[0]]) for h in hour]
    Vpred       = []
    for i in range(len(data)):
        idx     = np.where(hour == time[i])[0][0]
        Vpred.append(np.power((data[i] * s[idx]) + V_prime_bar[idx], 1/m))
    return np.array(Vpred)


def getmode(data, bin_size='default'):
    if bin_size == 'default':
        h, bin_edges = np.histogram(data)
    else:
        num_bins        = round((max(data) - min(data)) / bin_size)
        h, bin_edges    = np.histogram(data, bins=num_bins)
    mode = bin_edges[np.argmax(h)]
    return mode


def mean_weibull(scale, shape):
    return scale * gamma(1 + 1/shape)


def median_weibull(scale, shape):
    return scale * np.power(np.log(2), 1/shape)


def mode_weibull(scale, shape):
    return scale * np.power((shape - 1) / shape, 1/shape)


def sd_weibull(scale, shape):
    return np.sqrt(np.power(scale, 2) * (gamma(1 + 2/shape) - np.power(gamma(1 + 1/shape), 2)))

def aicc(aic, n, p, q):
    aicc            = aic + 2 * (((p+q+1)**2 + (p+q+1))/(n-p-q))
    return aicc


import pandas as pd
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
# Importing csv file into ws
ws          = pd.read_csv(r"D:\resources wind data set\Wind_Time_Series_Dataset\Wind Time Series Dataset(hourly).csv")

# Getting Wind speed data(hourly) in to wsdata from ws
wsdata      = ws.iloc[:,1].to_numpy()
# Weibull Function
shape, loc, scale = weibull_min.fit(wsdata, floc=0)
print("Shape: ", shape, "\nScale: ", scale)
plt.hist(wsdata, density=True, bins=50, color="lightgrey", edgecolor="black")
x           = np.linspace(min(wsdata), max(wsdata), 500)
y           = weibull_min.pdf(x, shape, loc=0, scale=scale)
# Plotting for Hourly Wind Speed
plt.plot(x, y, color="red", linewidth=2)
plt.xlabel("Wind Speed (m/s)", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("Hourly wind speed data", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Increasing dpi of the plot
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.show()

#plotting for 10 min Wind Speed
ws_min      = pd.read_csv(r"D:\resources wind data set\Wind_Time_Series_Dataset\Wind Time Series Dataset(10min).csv")
wsdata      = ws_min.iloc[:,1].to_numpy()
shape, loc, scale = weibull_min.fit(wsdata, floc=0)
print("Shape: ", shape, "\nScale: ", scale)
plt.hist(wsdata, density=True, bins=50, color="lightgrey", edgecolor="black")
x           = np.linspace(min(wsdata), max(wsdata), 500)
y           = weibull_min.pdf(x, shape, loc=0, scale=scale)
plt.plot(x, y, color="red", linewidth=2)
plt.xlabel("Wind Speed (m/s)", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("10min wind speed data", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.show()

# Printing the processed wind speed data
ws.head()

# Creatind standard deviation for hourly data
wsda        = (ws['WindSpeed'] - ws['WindSpeed'].min()) / (ws['WindSpeed'].max() - ws['WindSpeed'].min())
wsda

#Plotting the deviation for hourly using Plotly 
import plotly.graph_objs as go
import plotly.io as pio

# Create figure
fig = go.Figure()

# Add trace for wind speed
fig.add_trace(go.Scatter(x=ws['Time'][:1811], y=ws['WindSpeed'][:1811], mode='markers', marker=dict(size=3)))

# Set x-axis and y-axis labels
fig.update_xaxes(title_text="Time (Days)")
fig.update_yaxes(title_text="Wind Speed (m/s)")

# Set x-axis tick labels
fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 300, 600, 900, 1200, 1500, 1800], ticktext=[0, 15, 30, 45, 60, 75, 90]))

# Show the plot
pio.show(fig)

# plotting for 10 min data using plotly

fig_10min = go.Figure()

# Add trace for wind speed
fig_10min.add_trace(go.Scatter(x=ws_min['Time'][:1811], y=ws_min['WindSpeed'][:1811], mode='markers', marker=dict(size=3)))

# Set x-axis and y-axis labels
fig_10min.update_xaxes(title_text="Time")
fig_10min.update_yaxes(title_text="Wind Speed (m/s)")

# Set x-axis tick labels
fig_10min.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 300, 600, 900, 1200, 1500, 1800], ticktext=[0, 15, 30, 45, 60, 75, 90]))

# Show the plot
pio.show(fig_10min)