import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.signal import savgol_filter

from .model import DATA_PATH

def import_notification_target():
    df=pd.read_excel(io=str(DATA_PATH)+'/Notifications_Data_030820-123120.xlsx',index_col=0)
    notification_data = df["NOTIFICATIONS"]

    notification_smoothed = savgol_filter(notification_data, window_length=12, polyorder=2)
    notification_smoothed = pd.DataFrame(notification_smoothed, index=notification_data.keys())
    notification_smoothed = notification_smoothed.rename(columns={0:'smoothed_data'})
    notification_smoothed = notification_smoothed['smoothed_data']

    #fig = px.line(notification_data)
    #fig = fig.add_trace(go.Scatter(x = notification_smoothed.keys(), y=notification_smoothed, name = "smoothed"))
    #fig.show()

    notification_target = notification_smoothed[31::2]
    return notification_target