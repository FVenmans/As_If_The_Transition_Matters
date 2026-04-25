"""
Frank: this is my file adapted from post_process_nolearn.py

"""
import numpy as np  # using float32 to have a compatibility with tensorflow
import pandas as pd
import shutil  
import importlib
import tensorflow as tf
#import seaborn as sns
import Parameters
import matplotlib.pyplot as plt
from matplotlib import rc
import State
import PolicyState
import Definitions
from Graphs import run_episode
Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
# --------------------------------------------------------------------------- #
#Parameters
# --------------------------------------------------------------------------- #
ts_beg = 2025
ts_end = 2100
dt=tf.cast(Parameters.dt , dtype=tf.int32)
ts = range(ts_beg,  ts_end + 1, dt)
N_episode_length = tf.cast(((ts_end - ts_beg)/dt + 1), dtype=tf.int32) #76 elements

ts_end_single = 2225
ts_single = range(ts_beg, ts_end_single + 1, dt)
N_episode_length_single = tf.cast(((ts_end_single - ts_beg)/dt + 1), dtype=tf.int32) 

N_episode_length_euler= min(76, N_episode_length)   
N_irf = min(76, N_episode_length)  # number of years for IRF
N_sim_batch = 10000  #initially 100
Unscaled=1 #set to 0 to multiply capitals and investments and production by AL (hard to see variation)

versionidx = 5 #change here for consequtive versions. Spaces are important for sed command in bash file
versions = {
    1: 'RRA=1.35',
    2: 'Temperature and capital uncertainty',
    3: 'Only temperature uncertainty',
    4: 'Only capital uncertainty',
    5: 'No uncertainty',
    6: 'RRA=3'
}
# --------------------------------------------------------------------------- #
# Plot setting
# --------------------------------------------------------------------------- #
tf.get_logger().setLevel('CRITICAL')
pd.set_option('display.max_columns', None)
# Get the size of the current terminal
terminal_size_col = shutil.get_terminal_size().columns  

rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
# rc('text', usetex=True)

# Font size
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.title_fontsize"] = 14

# Figure size
fsize = (9, 6)
line_args = {'markerfacecolor': 'None', 'color': 'tab:blue', 'marker': None,
             'linestyle': '-'}
distribution_args = {'markerfacecolor': 'None', 'color': 'tab:blue',
                     'marker': '.', 'linestyle': 'None'}
lb_quantiles = [10, 25, 50, 75, 90]

# Error percentiles used to report distributions of Euler discrepancies
err_percentiles = [0.001, 0.25, 0.50, 0.75, 0.999]

# seaborn color ##Frank: I  commented this out, because package seaborn could not be found.
#sns_color_list = sns.color_palette()
#sns_blue, sns_orange, sns_green, sns_red, sns_purple, sns_brown, sns_pink, \
#    sns_gray, sns_yellow, sns_cyan = sns.color_palette()

# --------------------------------------------------------------------------- #
# Economic variables
# --------------------------------------------------------------------------- #

# Defined economic variables (definitions)
econ_defs = ['y', 'c', 'E', 'P', 'P_kd','P_ks','y_E', 'y_T', 'epsilon', 'y_epsilon','SCC','ic','id','lambdakd']  
 
econ_def_labels = {'y':r'$Production$', 
                   'c':r'$Consumption$', 
                   'E':r'$Energy$', 
                   'P':r'GHG Emissions (GtCO2e)', 
                   'P_kd':r'$P_{kd}$',
                   'P_ks':r'$P_{ks}$',
                   'y_E':r'$y_{E}$',
                   'y_T':r'$y_{S}$',   
                   'epsilon':r'$\epsilon$',
                   'y_epsilon':r'$y_{\epsilon}$',
                   'SCC':r'Social Cost of Carbon (USD 2024/tCO2e)',
                   'ic':r'Clean investment',
                   'id':r'Dirty investment',
                   'lambdakd':r'$\lambda_{kd}$'
                  }

# State labels
state_labels = {'kf':r'$kf$',
                'kc': r'Clean capital (per unit of effective labour)',
                'kd': r'$kd$',
                'ks': r'$ks$',
                'T': r'Temperature (°C)',
                'tau': r'$tau$',
                'y_kf': r'$y_{kf}$',
                'y_kc': r'$y_{kc}$',
                'y_kd': r'$y_{kd}$',
                'y_T': r'$y_{T}$'
               }


# Policy labels
policy_state_labels = {'if_': r'$if$',
                       'ic' : r'$ic$',
                       'id' : r'$id$',
                       'lambdaks': r'$\lambda^{ks}$',
                       'lambdaT' : r'$\lambda^T$',
                       'lambdakd': r'$\lambda^{kd}$' ,
                       'V': r'$V$'
                       }
    

# --------------------------------------------------------------------------- #
# Simulation periods and batch size
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Simulate the economy for {} periods".format(N_episode_length_single))

# Import equations
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# Number of state, policy and defined variables
N_state = len(Parameters.states)  # Number of state variables
N_policy_state = len(Parameters.policy_states)  # Number of policy variables
N_defined = len(econ_defs)  # Number of defined variables

# Starting state
#earlier code: starting_state = tf.reshape(tf.constant([Parameters.kf0, Parameters.kc0, Parameters.kd0, Parameters.ks0, Parameters.T0, Parameters.tau0]), shape=(1, N_state))  
initial_values = tf.constant([[
    Parameters.kf0, Parameters.kc0, Parameters.kd0, Parameters.ks0, 
    Parameters.T0, Parameters.tau0, 1, 1, 1, 1
]])  
# Tile to match the batch size (workaround to use the Parameters.starting_state.assign function)
Parameters.starting_state.assign(
    tf.tile(initial_values, [Parameters.N_sim_batch, 1])
)  
Hooks.post_init()
starting_state = Parameters.starting_state[0:1,:]

# Simulate the economy for N_episode_length_single time periods
simulation_starting_state = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length_single, 1, 1]) 
# --------------------------------------------------------------------------- #
#Graphs with one realization
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Simulate the economy for one episode for {} periods".format(
    N_episode_length_single))
# --------------------------------------------------------------------------- #
# Simulate for one state episode 
state_1episode = run_episode(simulation_starting_state) 

# Simulate for one policy episode
policy_state_1episode = np.empty(
    shape=[N_episode_length_single, 1, N_policy_state], dtype=np.float32)  
for tidx in range(N_episode_length_single):  
    policy_state_val = Parameters.policy(state_1episode[tidx, :, :])  #FV: loop needed because Parameters.policy cannot take in 3D tensors, it requires batch size in the first dimension and states in the next. 
    policy_state_1episode[tidx, :, :] = policy_state_val

state_1episode = tf.reshape(state_1episode, shape=[N_episode_length_single, N_state]) 
policy_state_1episode = tf.reshape(
    policy_state_1episode, shape=[N_episode_length_single, N_policy_state])

#FV: make time-dependent tfp * labour 
AL_1episode = 1 if Unscaled == 1 else Definitions.AL(state_1episode, policy_state_1episode)

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Plot one simulated episode for {} periods".format(N_episode_length_single))
# --------------------------------------------------------------------------- #

for sidx, state in enumerate(Parameters.states):
    fig, ax = plt.subplots(figsize=fsize)
    # State variable
    state_val = getattr(State, state)(state_1episode)
    # Adjust state variables
    if state in ['ks']:
        state_val = AL_1episode * state_val * Parameters.ks_scale 
    elif state in ['kf']:
        state_val = AL_1episode * state_val * Parameters.kf_scale
    elif state in ['kc', 'kd']:  
        state_val = AL_1episode * state_val 
    ax.plot(ts_single, state_val.numpy(), **line_args) #FV: .numpy() converts tensor into a numpy array, ** allows to pass a dictionary to the function
    #ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end_single])
    ax.set_ylabel(state_labels[state])

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end_single)
        + '_' + state + '.pdf')
    plt.close()

for pidx, ps in enumerate(Parameters.policy_states):
    fig, ax = plt.subplots(figsize=fsize)
    # policy variable
    ps_val = getattr(PolicyState, ps)(policy_state_1episode)
    if ps in ['id']:
        ps_val = AL_1episode * ps_val * Parameters.id_scale
    elif ps in ['ic']:
        ps_val = AL_1episode * ps_val * Parameters.ic_scale
    elif ps in ['if_']:
        ps_val = AL_1episode * ps_val   
    ax.plot(ts_single, ps_val.numpy(), **line_args)
    #ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end_single])
    ax.set_ylabel(policy_state_labels[ps])

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end_single)
        + '_' + ps + '.pdf')
    plt.close()

for didx, de in enumerate(econ_defs): 
    fig, ax = plt.subplots(figsize=fsize)
    # defined economic variable
    de_val = getattr(Definitions, de)(state_1episode, policy_state_1episode)
    if de in ['y','c','E']:
        de_val= AL_1episode * de_val
    ax.plot(ts_single, de_val.numpy(), **line_args)
    #x.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end_single])
    ax.set_ylabel(econ_def_labels[de])

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end_single)
        + '_' + de + '.pdf')
    plt.close()

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# Make Monte Carlo Graphs
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Simulate the economy for {} periods in {} simulation batch".format(
    N_episode_length, N_sim_batch))
# --------------------------------------------------------------------------- #
    
simulation_starting_state_batch = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, N_sim_batch, 1])

# Simulate the economy for N_sim_batch times to compute the collection of
# state and policy episodes
state_episode_batch = run_episode(simulation_starting_state_batch)

# Policy variables for N_sim_batch simulations (only used in Euler discrepancies)
policy_state_episode_batch = np.empty(
    shape=[N_episode_length, N_sim_batch, N_policy_state], dtype=np.float32)
for tidx in range(N_episode_length):  
    policy_state_batch = Parameters.policy(state_episode_batch[tidx, :, :])
    policy_state_episode_batch[tidx, :, :] = policy_state_batch

# Scaling + calculate policy variables and definitions
state_episode_batch_scaled = np.empty_like(state_episode_batch, dtype=np.float32)
policy_state_episode_batch_scaled = np.empty_like(policy_state_episode_batch, dtype=np.float32)
defined_episode_batch_scaled = np.empty(shape=[N_episode_length, N_sim_batch, N_defined], dtype=np.float32)

for tidx in range(N_episode_length):
    state_batch = state_episode_batch[tidx, :, :]
    policy_state_batch = Parameters.policy(state_episode_batch[tidx, :, :])
    AL_batch = 1 if Unscaled == 1 else Definitions.AL(state_batch, policy_state_batch)
    # State variables
    for sidx, state in enumerate(Parameters.states):
        state_val = getattr(State, state)(state_batch)
        if state in ['ks']:
            state_val = AL_batch * state_val * Parameters.ks_scale 
        elif state in ['kf']:
            state_val = AL_batch * state_val * Parameters.kf_scale
        elif state in ['kc', 'kd']:  
            state_val = AL_batch * state_val 
        state_episode_batch_scaled[tidx, :, sidx] = state_val
    # Policy variables
    for pidx, policy in enumerate(Parameters.policy_states):
        policy_val = getattr(PolicyState, policy)(policy_state_batch)
        if policy in ['if_']:
            policy_val = AL_batch * policy_val  
        policy_state_episode_batch_scaled[tidx, :, pidx] = policy_val
    # Defined economic variables
    for didx, de in enumerate(econ_defs):
        defined_val = getattr(Definitions, de)(state_batch, policy_state_batch)
        if de in ['y','c','E','id','ic']:
            defined_val= AL_batch * defined_val
        defined_episode_batch_scaled[tidx, :, didx] = defined_val

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Plot the distribution of economic variables for {} periods".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
# Compute the quantiles of each variable along with the number of simulations
quantile_state = np.percentile(
    state_episode_batch_scaled, q=lb_quantiles, axis=1)
quantile_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=lb_quantiles, axis=1)
quantile_defined = np.percentile(
    defined_episode_batch_scaled, q=lb_quantiles, axis=1)
# Save quantiles for boxplots
years_to_save = [2030, 2050, 2100]
tidx_to_save = [list(ts).index(y) for y in years_to_save]

for varname in ['T', 'kc', 'kd']:
    idx = list(Parameters.states).index(varname)
    df = pd.DataFrame(
        quantile_state[:, tidx_to_save, idx],
        index=lb_quantiles, columns=years_to_save)
    df.to_csv(Parameters.LOG_DIR + f'/{varname}_boxplot_v{versionidx}.csv')

for varname in ['P', 'SCC']:
    idx = econ_defs.index(varname)
    df = pd.DataFrame(
        quantile_defined[:, tidx_to_save, idx],
        index=lb_quantiles, columns=years_to_save)
    df.to_csv(Parameters.LOG_DIR + f'/{varname}_boxplot_v{versionidx}.csv')

# Compute the range of each variable
range_state = np.percentile(state_episode_batch_scaled, q=[1, 99], axis=1)
range_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=[1, 99], axis=1)
range_defined = np.percentile(
    defined_episode_batch_scaled, q=[1, 99], axis=1)


# Plot the distribution of state variables
for sidx, state in enumerate(Parameters.states):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_state[0, :, sidx], range_state[1, :, sidx],
        facecolor='tab:gray', alpha=0.1,
        label='Range of sample paths (1% to 99%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_state[qidx, :, sidx],
                label='{}% quantile'.format(lb_quantiles[qidx]))
    #plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(state_labels[state])
    if state=='T':
        plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + state + '_v' + str(versionidx) + '.pdf')
    plt.close()

#kd and kc in the same subplot
fig, ax = plt.subplots(figsize=fsize)
for state in ['kc', 'kd']:
    sidx = list(Parameters.states).index(state)
    ax.fill_between(
        ts, range_state[0, :, sidx], range_state[1, :, sidx],
        facecolor='tab:gray', alpha=0.1)
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_state[qidx, :, sidx])
ax.set_xlim([ts_beg, ts_end])
ax.set_ylabel('Capital (per unit of effective labour)')
plt.tight_layout()
plt.savefig(Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
            + '_kc_kd_v' + str(versionidx) + '.pdf')
plt.close()

#policies
for pidx, policy in enumerate(Parameters.policy_states):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_policy_state[0, :, pidx], range_policy_state[1, :, pidx],
        facecolor='tab:gray', alpha=0.1,
        label='Range of sample paths (1% to 99%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_policy_state[qidx, :, pidx],
                label='{}% quantile'.format(lb_quantiles[qidx]))
    #plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(policy_state_labels[policy])
    #plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + policy + '_v' + str(versionidx) + '.pdf')
    plt.close()

#definitions
for didx, de in enumerate(econ_defs):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_defined[0, :, didx], range_defined[1, :, didx],
        facecolor='tab:gray', alpha=0.1,
        label='Range of sample paths (1% to 99%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_defined[qidx, :, didx],
                label='{}% quantile'.format(lb_quantiles[qidx]))
    #plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(econ_def_labels[de])
    plt.tight_layout()
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '_v' + str(versionidx) + '.pdf')
    plt.close()

# ----------------------------------------------------------------------------#
# Compute the mean of each variable in a dictionary
# --------------------------------------------------------------------------- #
state_means = {
    s: state_episode_batch_scaled[:, :, sidx].mean(axis=1)
    for sidx, s in enumerate(Parameters.states)
}
policy_means = {
    ps: policy_state_episode_batch_scaled[:, :, pidx].mean(axis=1)
    for pidx, ps in enumerate(Parameters.policy_states)
}
defined_means = {
    de: defined_episode_batch_scaled[:, :, didx].mean(axis=1)
    for didx, de in enumerate(econ_defs)
}

# --------------------------------------------------------------------------- #
# Impulse Response Functions: shock T, kf and kc
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Compute IRF for {} periods and {} simulations".format(
    N_irf , N_sim_batch))

irf_shocks = {'T': (list(Parameters.states).index('T') , 3*0.026*1.38),  #0.1 before
             'kf': (list(Parameters.states).index('kf'), 3*0.015*4.3), #0.43 before
             'kc': (list(Parameters.states).index('kc'), 3*0.015*0.8)} #0.08 before

for shockname, (shockidx,shocksize) in irf_shocks.items():          
    # --- Create shocked starting state ---
    starting_state_shocked = starting_state.numpy().copy()
    starting_state_shocked[0, shockidx] += shocksize
    starting_state_shocked = tf.constant(starting_state_shocked)
    
    # --- Simulate shocked economy ---
    simulation_starting_state_batch_shocked = tf.tile(
        tf.expand_dims(starting_state_shocked, axis=0), [N_episode_length, N_sim_batch, 1])
    state_episode_batch_shocked = run_episode(simulation_starting_state_batch_shocked)
    
    # --- Policy states for shocked episode ---
    #policy_state_episode_batch_shocked = np.empty(
    #    shape=[N_episode_length, N_sim_batch, N_policy_state], dtype=np.float32)
    #for tidx in range(N_episode_length):
    #    policy_state_episode_batch_shocked[tidx, :, :] = Parameters.policy(state_episode_batch_shocked[tidx, :, :])
    
    # --- Scale shocked variables ---
    state_episode_batch_scaled_shocked = np.empty_like(state_episode_batch_shocked, dtype=np.float32)
    policy_state_episode_batch_scaled_shocked = np.empty_like(policy_state_episode_batch, dtype=np.float32)
    defined_episode_batch_scaled_shocked = np.empty(shape=[N_episode_length, N_sim_batch, N_defined], dtype=np.float32)
    for tidx in range(N_episode_length):
        AL_batch = 1 if Unscaled == 1 else Definitions.AL(state_batch, policy_state_batch)
        state_batch = state_episode_batch_shocked[tidx, :, :]
        policy_state_batch = Parameters.policy(state_episode_batch_shocked[tidx, :, :])
        #state variables
        for sidx, state in enumerate(Parameters.states):
            state_val = getattr(State, state)(state_batch)
            if state in ['ks']:
                state_val = AL_batch * state_val * Parameters.ks_scale
            elif state in ['kf']:
                state_val = AL_batch * state_val * Parameters.kf_scale
            elif state in ['kc', 'kd']:
                state_val = AL_batch * state_val
            state_episode_batch_scaled_shocked[tidx, :, sidx] = state_val
        # Policy variables
        for pidx, policy in enumerate(Parameters.policy_states):
            policy_val = getattr(PolicyState, policy)(policy_state_batch)
            if policy in ['if_']:
                policy_val = AL_batch * policy_val  
            policy_state_episode_batch_scaled_shocked[tidx, :, pidx] = policy_val    
        # --- Scale shocked defined variables ---
        for didx, de in enumerate(econ_defs):   
            defined_val = getattr(Definitions, de)(state_batch, policy_state_batch)
            if de in ['y', 'c', 'E', 'id', 'ic']:
                defined_val = AL_batch * defined_val
            defined_episode_batch_scaled_shocked[tidx, :, didx] = defined_val
    
    # --------------------------------------------------------------------------- #
    # Compute IRFs: mean difference (shocked - baseline) over 30 years
    # --------------------------------------------------------------------------- #
    
    ts_irf = ts[:N_irf]
    
    irf_vars = {
        'T':  (state_episode_batch_scaled,         state_episode_batch_scaled_shocked,           list(Parameters.states).index('T'),  'Global mean Temperature (°C)'),
        #'kd': (state_episode_batch_scaled,         state_episode_batch_scaled_shocked,           list(Parameters.states).index('kd'), 'Dirty capital (kd)'),
        'y':  (defined_episode_batch_scaled,       defined_episode_batch_scaled_shocked,         econ_defs.index('y'),                 'Production (y)'),
        'E':  (defined_episode_batch_scaled,       defined_episode_batch_scaled_shocked,         econ_defs.index('E'),                 'Final capital (E)'),
        'SCC':(defined_episode_batch_scaled,       defined_episode_batch_scaled_shocked,         econ_defs.index('SCC'),               'SCC (USD 2024/t CO2e)'),
        'P':  (defined_episode_batch_scaled,       defined_episode_batch_scaled_shocked,         econ_defs.index('P'),                 'GHG Emissions (Gt CO2e)'),  
        #'id': (defined_episode_batch_scaled,       defined_episode_batch_scaled_shocked,         econ_defs.index('id'),                 'Dirty investment (id)'), 
        'ic': (defined_episode_batch_scaled,       defined_episode_batch_scaled_shocked,         econ_defs.index('ic'),                 'Clean investment (ic)'), 
        'if_':(policy_state_episode_batch_scaled,  policy_state_episode_batch_scaled_shocked,    list(Parameters.policy_states).index('if_'), 'Final investment (if)') 
    }
    
    for varname, (base_arr, shocked_arr, idx, ylabel) in irf_vars.items():
        baseline_mean = base_arr[:N_irf, :, idx].mean(axis=1)
        shocked_mean  = shocked_arr[:N_irf, :, idx].mean(axis=1)
        irf           = shocked_mean - baseline_mean
    
        fig, ax = plt.subplots(figsize=fsize)
        ax.plot(ts_irf, irf)
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        #ax.set_xlabel('Year')
        #ax.set_ylabel(f'{ylabel}')
        ax.set_xlim([ts_irf[0], ts_irf[-1]])
        #ax.set_ylabel(f'{ylabel}', fontsize=ax.yaxis.label.get_size() * 1.5)
        ax.tick_params(axis='both', labelsize=24)
        plt.tight_layout()
        plt.savefig(Parameters.LOG_DIR + f'/irf_shock{shockname}_{varname}_v{versionidx}.pdf')
        plt.close()
        pd.Series(irf, index=ts_irf, name=varname).rename_axis('year').to_csv(
            Parameters.LOG_DIR + f'/irf_shock{shockname}_{varname}_v{versionidx}.csv', float_format='%.3f')
        
        
#-----------------------------------------------------------------------------#
#GDP volatility
#-----------------------------------------------------------------------------#
print("-" * terminal_size_col)
print(r"Compute the statistics of GDP etc. from {} simulation points".format(
    N_episode_length * N_sim_batch))
# --------------------------------------------------------------------------- #
gdpidx = econ_defs.index('y')  # a scalar 1 with the index where gdp is stored 
idx2030 = 2030 - ts_beg #this gives 5
idx2100 = 2100 - ts_beg

gdp = defined_episode_batch_scaled[:, :, gdpidx] #dimensions N_episode_lenght,N_sim_batch
log_gdp = np.log(gdp)
dln_gdp = np.diff(log_gdp, axis=0)

dln_gdp2030_df = pd.DataFrame(dln_gdp[idx2030-1,:])
dln_gdp2100_df = pd.DataFrame(dln_gdp[idx2100-1,:])

print(r"dln_gdp in 2030")
print(dln_gdp2030_df.describe(percentiles=err_percentiles, include='all'))
dln_gdp2030_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/dln_gdp2030_describe"+ "_v" + str(versionidx) +" .csv", index=True, float_format='%.3f')

print(r"dln_gdp in 2100")
print(dln_gdp2100_df.describe(percentiles=err_percentiles, include='all'))
dln_gdp2100_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/dln_gdp2100_describe"+ "_v" + str(versionidx) + ".csv", index=True, float_format='%.3f')
#-----------------------------------------------------------------------------#
# Risk premium on three capitals
#-----------------------------------------------------------------------------#

chi, delta, g_L, g, r = Parameters.chi, Parameters.delta, Parameters.g_L, Parameters.g, Parameters.r 

T = N_episode_length - 1  #75 periods only because expected return in 2100 is not available.
tidx_range = range(0, T)

R_kf = np.zeros(T)
R_kc = np.zeros(T)
R_kd = np.zeros(T)
R_rf = np.zeros(T)

for tidx in tidx_range:
    # adjustment cost factor at t and t+1
    adj_kf_t   = 1 - 2 * chi * policy_means['if_'][tidx]   / state_means['kf'][tidx]
    adj_kf_tp1 = 1 - 2 * chi * policy_means['if_'][tidx+1] / state_means['kf'][tidx+1]

    adj_kc_t   = 1 - 2 * chi * defined_means['ic'][tidx]    / state_means['kc'][tidx]
    adj_kc_tp1 = 1 - 2 * chi * defined_means['ic'][tidx+1]  / state_means['kc'][tidx+1]

    adj_kd_t   = 1 - 2 * chi * defined_means['id'][tidx]   / state_means['kd'][tidx]
    adj_kd_tp1 = 1 - 2 * chi * defined_means['id'][tidx+1] / state_means['kd'][tidx+1]

    # Returns (same formula for all three capitals (augment by g+g_L to convert to dollars.)
    R_kf[tidx] = (state_means['y_kf'][tidx+1]
                  +  (1 - delta - g - g_L  + chi * (policy_means['if_'][tidx+1] / state_means['kf'][tidx+1])**2)/adj_kf_tp1 
                  ) * adj_kf_t - 1
    R_kc[tidx] = (state_means['y_kc'][tidx+1]
                  +  (1 - delta - g - g_L  + chi * (defined_means['ic'][tidx+1] / state_means['kc'][tidx+1])**2)/adj_kc_tp1 
                  ) * adj_kc_t - 1 
    R_kd[tidx] = (state_means['y_kd'][tidx+1]
                  +  (1 - delta - g - g_L  + chi * (defined_means['id'][tidx+1] / state_means['kd'][tidx+1])**2)/adj_kd_tp1 
                  ) * adj_kd_t - 1
    # Risk-free rate
    R_rf[tidx] = r -g 

t_plot = np.arange(ts_beg , ts_end ) # 2025 to 2099, 75 points

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_plot, R_kf, label=r'$R_{kf}$ (final capital)',  color='brown')
ax.plot(t_plot, R_kc, label=r'$R_{kc}$ (clean capital)',   color='green')
ax.plot(t_plot, R_kd, label=r'$R_{kd}$ (dirty capital)',   color='grey')
ax.plot(t_plot, R_rf, label=r'$r -g$ (risk-free rate)', color='black',
        linestyle='--')
ax.set_xlabel('Year')
ax.set_ylabel('Expected return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Parameters.LOG_DIR + f'/CapitalRiskPremia_v{versionidx}.pdf')
plt.close()

#-----------------------------------------------------------------------------#
#Compare graphs of mean Emissions, Tem, SCC and capitals for different versions
#-----------------------------------------------------------------------------#
print("-" * terminal_size_col)
print(r"Make graphs of mean Emissions, Temperature SCC and Kd&Kc&Ks for {} years".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
# Save means
state_mean = state_episode_batch_scaled.mean(axis=1)
for varname in ['T', 'kd', 'kc']:
    idx = list(Parameters.states).index(varname)
    pd.Series(state_mean[:, idx], index=ts, name=varname).rename_axis('year').to_csv(
        Parameters.LOG_DIR + f"/{varname}_mean{versionidx}.csv")
for varname in ['P', 'SCC']:
    idx = econ_defs.index(varname)
    mean = defined_episode_batch_scaled[:, :, idx].mean(axis=1)
    pd.Series(mean, index=ts, name=varname).rename_axis('year').to_csv(
        Parameters.LOG_DIR + f"/{varname}_mean{versionidx}.csv")

#Make graphs showing this version and the preceding ones 
variables = {
    'T':   ('T_mean',   'Temperature (T)',  'Temp'),
    'kd':  ('kd_mean',  'Dirty capital (kd)', 'kd'),
    'kc':  ('kc_mean',  'Clean capital (kc)', 'kc'),
    'P':   ('P_mean',   'Pollution (P)',       'P'),
    'SCC': ('SCC_mean', 'SCC',               'SCC'),
}

for varname, (fileprefix, ylabel, filelabel) in variables.items():
    for ts_end_graph in [ts_end, 2050]:
        fig, ax = plt.subplots(figsize=fsize)
        for v in range(2, versionidx + 1):  # set range from 2 to exclude the first graph wirh RRA=1.35
            df = pd.read_csv(Parameters.LOG_DIR + f"/{fileprefix}{v}.csv", index_col='year')
            df = df[df.index <= ts_end_graph]
            ax.plot(df.index, df[varname], label=versions[v])
        ax.set_ylabel(ylabel)
        ax.set_xlim([ts_beg, ts_end_graph])
        ax.legend()
        plt.tight_layout()
        plt.savefig(Parameters.LOG_DIR + f'/{filelabel}_to{ts_end_graph}_v{versionidx}.pdf')
        plt.close()

# --------------------------------------------------------------------------- # 
#Calculate Euler approximation errors 
# --------------------------------------------------------------------------- #

print("-" * terminal_size_col)
print(r"Compute the Euler discrepancies for {} years in {} simulation "
      "batch".format(N_episode_length_euler, N_sim_batch))

euler_list = []

for tidx in range(N_episode_length_euler):
    state_batch = state_episode_batch[tidx, :, :]
    policy_state_batch = policy_state_episode_batch[tidx, :, :]
    
    euler_list.append(
        pd.DataFrame(
            Equations.equations(state_batch, policy_state_batch)).abs())

euler_discrepancies_df = pd.concat(euler_list, axis=0, ignore_index=True)

print("-" * terminal_size_col)
print("Print the percentiles 0.1% 25% 50% 75% 99.9% of the Euler discrepancies")

print(euler_discrepancies_df.describe(
    percentiles=err_percentiles, include='all'))

# Save 
euler_discrepancies_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/Euler_discrepancies_equations" + "_v" + str(versionidx) + ".csv", float_format='%.3e')

print("Exit post processing")

