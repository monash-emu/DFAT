import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from summer2.functions.time import get_piecewise_function as pcwise_fcn
from summer2.functions.time import get_linear_interpolation_function as linear_interp_fcn
from summer2.functions.time import get_sigmoidal_interpolation_function as sigmoidal_interp_fcn
from summer2.functions import time as stf

pd.options.plotting.backend = "plotly"

def get_pcwise_transmission_modifier(param_dict,num_breakpts):
    breakpts = []
    s = param_dict["len_pd"+str(1)]
    breakpts.append(s)
    for i in range(2,num_breakpts+1):
        s += param_dict["len_pd"+str(i)]
        breakpts.append(s)
    rates = []
    for k in range(1,num_breakpts+2):
        rates.append(param_dict["rate"+str(k)])
    f = pcwise_fcn(breakpts, rates)
    F = stf.get_time_callable(f)
    return F

def get_sigmoidal_transmission_modifier(times,param_dict,num_breakpts):
    curr_bp = times[0]
    breakpts = []
    vals = []
    breakpts.append(curr_bp)
    vals.append(param_dict["val0"])
    time_length = times[1] - times[0]
    subinterval_length = time_length / (num_breakpts + 1)
    for i in range(1,num_breakpts+1):
        curr_bp += subinterval_length
        breakpts.append(curr_bp)
        vals.append(param_dict["val"+str(i)])
    breakpts.append(times[1])
    vals.append(param_dict["val"+str(num_breakpts+1)])
    f = sigmoidal_interp_fcn(breakpts, vals)
    F = stf.get_time_callable(f)
    return F

def get_linear_interp_transmission_modifier(times,param_dict,num_breakpts):
    curr_bp = times[0]
    breakpts = []
    vals = []
    breakpts.append(curr_bp)
    vals.append(param_dict["val0"])
    time_length = times[1] - times[0]
    subinterval_length = time_length / (num_breakpts + 1)
    for i in range(1,num_breakpts+1):
        curr_bp += subinterval_length
        breakpts.append(curr_bp)
        vals.append(param_dict["val"+str(i)])
    breakpts.append(times[1])
    vals.append(param_dict["val"+str(num_breakpts+1)])
    f = linear_interp_fcn(breakpts, vals)
    F = stf.get_time_callable(f)
    return F

def plot_transmission_modifier(m,F):
    x_vals = m.times
    y_vals = F(x_vals)
    epoch = m.get_epoch()
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x = epoch.index_to_dti(m.times), y = y_vals, name = "transmission_modifier"))
    fig.show()

