import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from summer2 import CompartmentalModel, Stratification, Multiply
from summer2.parameters import Parameter
from summer2.functions.time import get_piecewise_function as pcwise_fcn
from summer2.functions.time import get_linear_interpolation_function as linear_interp_fcn
from summer2.functions.time import get_sigmoidal_interpolation_function as sigmoidal_interp_fcn

DATA_PATH = Path(__file__).parent.parent / "data"
REF_DATE = datetime(2020,4,7)

def build_model(num_breakpts, transmission_modifier_mode = 'pcwise_constant'):
    m = build_base_model(num_breakpts, transmission_modifier_mode)
    m.stratify_with(generate_age_stratification(m))
    return m

def build_base_model(num_breakpts, transmission_modifier_mode = 'pcwise_constant'):
    #num_breakpts = number of breakpts for pcwise constant effective transmission rate parameter
    times = [0,268]
    m = CompartmentalModel(
    times=times,
    compartments=["S", "E1", "E2", "E3", "E4", "I1", "I2", "I3", "I4", "Q1", "Q2", "Q3", "Q4","R"],
    infectious_compartments=["I1", "I2", "I3", "I4"],
    timestep = 1,
    ref_date=REF_DATE
    )
    m.set_initial_population({"S": 13484462.0, "E1": 90.0, "E2": 90.0, "E3": 90.0, "E4": 90.0 , "I1": 136.0, "I2": 136.0, "I2": 136.0, "I3": 136.0, "I4": 136.0})
    #We set up a time-varying "transmission modifier" that may be piecewise constant, sigmoidal, or linear interpolated
    if transmission_modifier_mode == 'pcwise_constant':
        breakpts, rates = generate_pcwise_transmission_params(num_breakpts)
        transmission_modifier = pcwise_fcn(breakpts, rates)
    else:
        breakpts, vals  = generate_transmission_params(times,num_breakpts)
        if transmission_modifier_mode == 'sigmoidal':
            transmission_modifier = sigmoidal_interp_fcn(breakpts, vals)
        elif transmission_modifier_mode == 'linear_interp':
            transmission_modifier = linear_interp_fcn(breakpts, vals)
    m.add_infection_frequency_flow("infection", transmission_modifier, "S","E1")
    #Progression rate for wild-type strain is 1/6.65 (Yu Wu et al.), which is multiplied by 4 because of the chained E compartmens
    m.add_transition_flow("progression1", 0.6,"E1","E2")
    m.add_transition_flow("progression2", 0.6,"E2","E3")
    m.add_transition_flow("progression3", 0.6,"E3","E4")
    m.add_transition_flow("progression4", 0.6,"E4","I1")
    #Detection can happen from any of the I compartments
    m.add_transition_flow("notification1", Parameter("detection_rate1"),"I1","Q1")
    m.add_transition_flow("notification2", Parameter("detection_rate2"),"I2","Q2")
    m.add_transition_flow("notification3", Parameter("detection_rate3"),"I3","Q3")
    m.add_transition_flow("notification4", Parameter("detection_rate4"),"I4","Q4")
    # The progressions along the I (or Q) compartments are governed (ultimately) by a single parameter tau, which corresponds to the mean no. of days of infectiousness
    # The rate for each transition is 4/tau (the 4 is due to the chained compartments structure).
    # Detection of cases can be done at any stage of the infectiousness period, given by the transition from Ik to Qk, for k=1,2,3,4
    #Action point: Below is the assumption for now but this should be updated after checking relevant lit.
    tau = 14.0
    m.add_transition_flow("I1_to_I2", 4/tau,"I1","I2")
    m.add_transition_flow("I2_to_I3", 4/tau,"I2","I3")
    m.add_transition_flow("I3_to_I4", 4/tau,"I3","I4")
    m.add_transition_flow("I4_to_R", 4/tau,"I4","R")
    m.add_transition_flow("Q1_to_Q2", 4/tau,"Q1","Q2")
    m.add_transition_flow("Q2_to_Q3", 4/tau,"Q2","Q3")
    m.add_transition_flow("Q3_to_Q4", 4/tau, "Q3","Q4")
    m.add_transition_flow("Q4_to_R", 4/tau, "Q4","R")
    #
    #Requesting Outputs
    m.request_output_for_flow("notification1", "notification1")
    m.request_output_for_flow("notification2", "notification2")
    m.request_output_for_flow("notification3", "notification3")
    m.request_output_for_flow("notification4", "notification4")
    m.request_output_for_compartments(name="infectious1", compartments=["I1"], save_results=True)
    m.request_output_for_compartments(name="infectious2", compartments=["I2"], save_results=True)
    m.request_output_for_compartments(name="infectious3", compartments=["I3"], save_results=True)
    m.request_output_for_compartments(name="infectious4", compartments=["I4"], save_results=True)
    m.request_output_for_compartments(name="infectious", compartments=["I1","I2","I3","I4"], save_results=True)
    m.request_output_for_compartments(name="N", compartments=["S", "E1", "E2", "E3", "E4", "I1", "I2", "I3", "I4", "Q1", "Q2", "Q3", "Q4","R"], save_results=True)
    m.request_aggregate_output(name = "notifications", sources=["notification1", "notification2", "notification3", "notification4"], save_results=True)
    return m

def generate_age_mixing_matrix(home_weight=1.0, work_weight=1.0, school_weight=0.0, other_weight=1.0):
    mixing_matrix_home = pd.read_csv(str(DATA_PATH)+"/mixing_matrices/12_year_bands/synthetic_mixing_ncr_home.csv")
    mixing_matrix_work = pd.read_csv(str(DATA_PATH)+"/mixing_matrices/12_year_bands/synthetic_mixing_ncr_work.csv")
    mixing_matrix_school = pd.read_csv(str(DATA_PATH)+"/mixing_matrices/12_year_bands/synthetic_mixing_ncr_school.csv")
    mixing_matrix_other = pd.read_csv(str(DATA_PATH)+"/mixing_matrices/12_year_bands/synthetic_mixing_ncr_other.csv")
    age_mixing_matrix = home_weight * mixing_matrix_home.drop(['Unnamed: 0'],axis=1).to_numpy() + work_weight * mixing_matrix_work.drop(['Unnamed: 0'],axis=1).to_numpy() + school_weight * mixing_matrix_school.drop(['Unnamed: 0'],axis=1).to_numpy() + other_weight * mixing_matrix_other.drop(['Unnamed: 0'],axis=1).to_numpy()
    return age_mixing_matrix

def generate_age_stratification(model, home_weight=1.0, work_weight=1.0, school_weight=0.0, other_weight=1.0):
    age_mixing_matrix = generate_age_mixing_matrix(home_weight=1.0, work_weight=1.0, school_weight=0.0, other_weight=1.0)
    strata = [str(n) for n in range(0,61,12)]
    strat = Stratification(name="age", strata=strata, compartments=model.compartments)
    strat.set_population_split({"0": 0.2215, "12": 0.2108, "24": 0.2192, "36": 0.1598, "48": 0.1099, "60": 0.0788})
    strat.set_mixing_matrix(age_mixing_matrix)
    return strat

def generate_pcwise_transmission_params(num_breakpts):
    breakpts = []
    s = Parameter("len_pd"+str(1))
    breakpts.append(s)
    for i in range(2,num_breakpts+1):
        s += Parameter("len_pd"+str(i))
        breakpts.append(s)
    rates = []
    for k in range(1,num_breakpts+2):
        rates.append(Parameter("rate"+str(k)))
    return breakpts, rates

def generate_transmission_params(times,num_breakpts):
    #This is for sigmoidal and linear interp.
    if isinstance(num_breakpts, int):
        curr_bp = times[0]
        breakpts = []
        vals = []
        breakpts.append(curr_bp)
        vals.append(Parameter("val0"))
        time_length = times[1] - times[0]
        subinterval_length =  time_length / (num_breakpts + 1)
        for i in range(1,num_breakpts+1):
            curr_bp += subinterval_length
            breakpts.append(curr_bp)
            vals.append(Parameter("val"+str(i)))
        breakpts.append(times[1])
        vals.append(Parameter("val"+str(num_breakpts+1)))
    elif num_breakpts == 'fixed1':
        breakpts = [0, 20, 39, 55, 75, 95, 118, 134, 154, 174, 194, 214, 234, 254, 268]
        num_breakpts = len(breakpts)-2
        vals = []
        for i in range(0,num_breakpts+2):
            vals.append(Parameter("val"+str(i)))
    elif num_breakpts == 'fixed2':
        breakpts = [0, 24, 39, 55, 85, 118, 134, 147, 177, 208, 238, 268]
        num_breakpts = len(breakpts)-2
        vals = []
        for i in range(0,num_breakpts+2):
            vals.append(Parameter("val"+str(i)))
    return breakpts, vals

def generate_default_parameters(m,num_breakpts,transmission_parameter_mode='pcwise_constant'):
    if num_breakpts == 'fixed1':
        num_breakpts = 13
    if num_breakpts == 'fixed2':
        num_breakpts = 10
    defp = {}
    if transmission_parameter_mode == 'pcwise_constant':
        time_length = m.times.max() - m.times.min()
        subinterval_length = time_length / (num_breakpts + 1)
        for i in range(1,num_breakpts+2):
            defp["len_pd"+str(i)] = 0.9*subinterval_length
            defp["rate"+str(i)] = 0.1
    else:
        for i in range(0,num_breakpts+2):
            defp["val"+str(i)] = 0.1
    defp["detection_rate1"] = 0.5
    defp["detection_rate2"] = 0.45
    defp["detection_rate3"] = 0.4
    defp["detection_rate4"] = 0.35
    return defp