import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from summer2.functions.time import get_piecewise_function as pcwise_fcn
from summer2.functions.time import get_linear_interpolation_function as linear_interp_fcn
from summer2.functions.time import get_sigmoidal_interpolation_function as sigmoidal_interp_fcn
from summer2.functions import time as stf

from estival import targets as est
from estival import priors as esp

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
    vals = []
    if isinstance(num_breakpts, int):
        curr_bp = times[0]
        breakpts = []
        breakpts.append(curr_bp)
        vals.append(param_dict["val0"])
        time_length = times[1] - times[0]
        subinterval_length = np.round(time_length / (num_breakpts + 1))
        for i in range(1,num_breakpts+1):
            curr_bp += subinterval_length
            breakpts.append(curr_bp)
            vals.append(param_dict["val"+str(i)])
        breakpts.append(times[1])
        vals.append(param_dict["val"+str(num_breakpts+1)])
    elif num_breakpts == 'fixed1':
        breakpts = [0, 20, 39, 55, 75, 95, 118, 134, 154, 174, 194, 214, 234, 254, 268]
        num_breakpts = len(breakpts)
        for i in range(0,num_breakpts):
            vals.append(param_dict["val"+str(i)])
    f = sigmoidal_interp_fcn(breakpts, vals)
    F = stf.get_time_callable(f)
    return F

def get_linear_interp_transmission_modifier(times,param_dict,num_breakpts):
    vals = []
    if isinstance(num_breakpts, int):
        curr_bp = times[0]
        breakpts = []
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
    elif num_breakpts == 'fixed1':
        breakpts = [0, 20, 39, 55, 75, 95, 118, 134, 154, 174, 194, 214, 234, 254, 268]
        num_breakpts = len(breakpts)
        for i in range(0,num_breakpts):
            vals.append(param_dict["val"+str(i)])
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

def generate_pcwise_transmission_priors(m,num_breakpts):
    time_length = m.times.max() - m.times.min()
    subinterval_length = time_length / (num_breakpts + 1)
    priors = [esp.UniformPrior("rate1",(0,0.7))]
    for i in range(1,num_breakpts+1):
        priors.append(esp.UniformPrior("len_pd"+str(i),(0.25*subinterval_length,1.5*subinterval_length)))
        #prior for len_pd might need adjustments
        priors.append(esp.UniformPrior("rate"+str(i+1),(0,1)))
    return priors

def generate_transmission_priors(num_breakpts):
    #For sigmoidal or linear_interp
    priors = []
    if num_breakpts == 'fixed1':
        num_breakpts = 13
    for i in range(0,num_breakpts+2):
        priors.append(esp.UniformPrior("val"+str(i),(0,1)))
    return priors