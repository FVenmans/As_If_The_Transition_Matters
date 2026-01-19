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

tf.get_logger().setLevel('CRITICAL')
pd.set_option('display.max_columns', None)

# --------------------------------------------------------------------------- #
# Plot setting
# --------------------------------------------------------------------------- #
# Get the size of the current terminal
terminal_size_col = shutil.get_terminal_size().columns  # Frank this stors the width of the terminal, in order to print a line of hyphens below.


rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
# rc('text', usetex=True)

# Font size
plt.rcParams["font.size"] = 20
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

# Error percentiles used to plot distributions
err_percentiles = [0.001, 0.25, 0.50, 0.75, 0.999]

# seaborn color ##Frank: I  commented this out, because package seaborn could not be found.
#sns_color_list = sns.color_palette()
#sns_blue, sns_orange, sns_green, sns_red, sns_purple, sns_brown, sns_pink, \
#    sns_gray, sns_yellow, sns_cyan = sns.color_palette()

# --------------------------------------------------------------------------- #
# Economic variabples
# --------------------------------------------------------------------------- #
# Exogenous parameters #these ar exogenous parameters, such as Land emissions or exogenous forcing in DICE. 
exparams = [] 
exparam_labels = {} #Frank: curly brackets are used for dictionaries and sets. {} is an empty dictionary, while set() is an empty set. Dictionaries and sets are unordered.

# Defined economic variables #Frank: varialbes that are derived from states and policies
econ_defs = ['y', 'c', 'E', 'P', 'P_kd','P_ks','y_E', 'y_S', 'epsilon', 'y_epsilon' ]

econ_def_labels = {'y':r'$y$', 
                   'c':r'$c$', 
                   'E':r'$E$', 
                   'P':r'$P$', 
                   'P_kd':r'$P_{kd}$',
                   'P_ks':r'$P_{ks}$',
                   'y_E':r'$y_{E}$',
                   'y_S':r'$y_{S}$',   
                   'epsilon':r'$\epsilon$',
                   'y_epsilon':r'$y_{\epsilon}$'}

# State labels
state_labels = {'kf':r'$kf$',
                'kc': r'$kc$',
                'kd': r'$kd$',
                'ks': r'$ks$',
                'T': r'$T$',
                'tau': r'$tau$',
                'y_kf': r'$y_{kf}$',
                'y_kc': r'$y_{kc}$',
                'y_kd': r'$y_{kd}$',
                'y_S': r'$y_{S}$'
               }


# Policy labels
policy_state_labels = {'if_': r'$if$',
                       'ic': r'$ic$',
                       'id': r'$id$',
                       'lambdaks': r'$lambdaks$',
                       'lambdaT': r'$lambdaT$',
                       'lambdatau': r'$lambdatau$',
                       'V': r'$V$'
                       }
    

# --------------------------------------------------------------------------- #
# Simulation periods and batch size
# --------------------------------------------------------------------------- #
begyear = 2020
endyear = 2120
dt=tf.cast(Parameters.dt , dtype=tf.int32)
# Simulate the economy for N_simulated episode length
# N_episode_length = Parameters.N_episode_length + 1
N_episode_length = tf.cast(((endyear - begyear)/dt + 1), dtype=tf.int32)

# Number of simulation batch, it should be arbitrary big enough
# Cai and Lontzek simulate their economy for 10,000 times
#N_sim_batch = 5000
N_sim_batch = 100  #initially 100

print("-" * terminal_size_col)
print("Simulate the economy for {} periods".format(N_episode_length))

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

#starting_state = tf.cast(starting_state, dtype=tf.float32) #FV: i added this, it should normally already be float32


# Simulate the economy for N_episode_length time periods
simulation_starting_state = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, 1, 1]) 


# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Simulate the economy for one episode for {} periods".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
# Simulate for one state episode 
state_1episode = run_episode(simulation_starting_state) #FV: run_episode is defined in Graphs.py
# Simulate for one policy episode
policy_state_1episode = np.empty(
    shape=[N_episode_length, 1, N_policy_state], dtype=np.float32)  #FV: unclear why the 2nd dimension of 1 is needed
for tidx in range(N_episode_length):  
    policy_state_val = Parameters.policy(state_1episode[tidx, :, :])  #FV: unclear why a loop is needed here. Because the function (method) state_1episode can only handle a singel set of states? 
    policy_state_1episode[tidx, :, :] = policy_state_val

state_1episode = tf.reshape(state_1episode, shape=[N_episode_length, N_state])
policy_state_1episode = tf.reshape(
    policy_state_1episode, shape=[N_episode_length, N_policy_state])

# Get simulation time periods
#ts = Definitions.tau2t(state_1episode, policy_state_1episode)
#ts = begyear + ts  # The original base year in DICE-2016 is 2015
#ts_beg, ts_end = int(tf.round(ts[0]).numpy()), int(tf.round(ts[-1]).numpy())
ts_beg=begyear
ts_end=endyear
ts = range(ts_beg,ts_end+1,dt)


# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
"""print(r"Plot the dynamics of the exogenous parameters for {} periods".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
for de in exparams: # I do not have exparams
    fig, ax = plt.subplots(figsize=fsize)
    de_val = getattr(Definitions, de)(state_1episode, policy_state_1episode)

    ax.plot(ts, de_val)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(exparam_labels[de])
    plt.savefig(
        Parameters.LOG_DIR + '/exparams_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '.pdf')
    plt.close()

# import ipdb; ipdb.set_trace()"""
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Plot one simulated episode for {} periods".format(N_episode_length))
# --------------------------------------------------------------------------- #
for sidx, state in enumerate(Parameters.states):
    fig, ax = plt.subplots(figsize=fsize)
    # State variable
    state_val = getattr(State, state)(state_1episode)
    # Adjust state variables
    if state in ['k_tildex']:
        tfp = Definitions.tfp(state_1episode, policy_state_1episode)
        lab = Definitions.lab(state_1episode, policy_state_1episode)
        state_val = tfp * lab * state_val
    elif state in ['MATx', 'MUOx', 'MLOx']:
        # Rescale from GtC to 1000 GtC
        state_val = state_val * 1000
    ax.plot(ts, state_val.numpy(), **line_args) #Frank: .numpy() converts tensor into a numpy array, ** allows to pass a dictionary to the function
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(state_labels[state])

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + state + '.pdf')
    plt.close()

for pidx, ps in enumerate(Parameters.policy_states):
    fig, ax = plt.subplots(figsize=fsize)
    # policy variable
    ps_val = getattr(PolicyState, ps)(policy_state_1episode)

    ax.plot(ts, ps_val.numpy(), **line_args)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(policy_state_labels[ps])

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + ps + '.pdf')
    plt.close()

for didx, de in enumerate(econ_defs): 
    fig, ax = plt.subplots(figsize=fsize)
    # defined economic variable
    de_val = getattr(Definitions, de)(state_1episode, policy_state_1episode)

    ax.plot(ts, de_val.numpy(), **line_args)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(econ_def_labels[de])

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '.pdf')
    plt.close()

# import ipdb; ipdb.set_trace()

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

# Policy variables for N_sim_batch times
policy_state_episode_batch = np.empty(
    shape=[N_episode_length, N_sim_batch, N_policy_state], dtype=np.float32)
for tidx in range(N_episode_length):
    policy_state_batch = Parameters.policy(state_episode_batch[tidx, :, :])
    policy_state_episode_batch[tidx, :, :] = policy_state_batch

# Some variables need to be rescaled for plotting
state_episode_batch_scaled = np.empty_like(
    state_episode_batch, dtype=np.float32)
policy_state_episode_batch_scaled = np.empty_like(
    policy_state_episode_batch, dtype=np.float32)
defined_episode_batch_scaled = np.empty(
    shape=[N_episode_length, N_sim_batch, N_defined], dtype=np.float32)

# State variables
for sidx, state in enumerate(Parameters.states):
    # Adjust state variables
    for tidx in range(N_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        state_val = getattr(State, state)(state_batch)
        """if state in ['k_tildex']:
            tfp = Definitions.tfp(state_batch, policy_state_batch)
            lab = Definitions.lab(state_batch, policy_state_batch)
            state_val = tfp * lab * state_val
        elif state in ['MATx', 'MUOx', 'MLOx']:
            # Rescale to GtC
            state_val = 1000. * state_val"""
        state_episode_batch_scaled[tidx, :, sidx] = state_val

# Policy variables
for pidx, policy in enumerate(Parameters.policy_states):
    # Adjust policy variables
    for tidx in range(N_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        policy_val = getattr(PolicyState, policy)(policy_state_batch)
        """if policy in ['con_tildey', 'inv_tildey', 'pe_tildey']:
            tfp = Definitions.tfp(state_batch, policy_state_batch)
            lab = Definitions.lab(state_batch, policy_state_batch)
            gr_tfp = Definitions.gr_tfp(state_batch, policy_state_batch)
            gr_lab = Definitions.gr_lab(state_batch, policy_state_batch)
            policy_val = tfp * lab * policy_val"""
        policy_state_episode_batch_scaled[tidx, :, pidx] = policy_val

# Defined economic variables
for didx, de in enumerate(econ_defs):
    # Adjust defined variables
    for tidx in range(N_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        defined_val = getattr(Definitions, de)(state_batch, policy_state_batch)
        if de in ['ygross_tilde', 'ynet_tilde']:
            tfp = Definitions.tfp(state_batch, policy_state_batch)
            lab = Definitions.lab(state_batch, policy_state_batch)
            defined_val = defined_val * tfp * lab
        elif de in ['Eind']:
            # Rescale to GtC unit
            defined_val = 1000. * defined_val
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

# Compute the range of each variable
range_state = np.percentile(state_episode_batch_scaled, q=[1, 99], axis=1)
range_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=[1, 99], axis=1)
range_defined = np.percentile(
    defined_episode_batch_scaled, q=[1, 99], axis=1)

# Compute the average of each variable
avg_state = np.average(state_episode_batch_scaled, axis=1)
avg_policy_state = np.average(policy_state_episode_batch_scaled, axis=1)
#avg_defined = np.average(defined_episode_batch_scaled, axis=1)

# Plot the distribution of state variables
for sidx, state in enumerate(Parameters.states):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_state[0, :, sidx], range_state[1, :, sidx],
        facecolor='tab:gray', alpha=0.1,
        label=r'Range of sample paths (1% to 99%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_state[qidx, :, sidx],
                label=r'{}\% quantile'.format(lb_quantiles[qidx]))
    plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(state_labels[state])
    plt.legend(loc='upper left')
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + state + '.pdf')
    plt.close()

for pidx, policy in enumerate(Parameters.policy_states):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_policy_state[0, :, pidx], range_policy_state[1, :, pidx],
        facecolor='tab:gray', alpha=0.1,
        label=r'Range of sample paths (1\% to 99\%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_policy_state[qidx, :, pidx],
                label=r'{}\% quantile'.format(lb_quantiles[qidx]))
    plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(policy_state_labels[policy])
    plt.legend(loc='upper left')
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + policy + '.pdf')
    plt.close()

for didx, de in enumerate(econ_defs):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_defined[0, :, didx], range_defined[1, :, didx],
        facecolor='tab:gray', alpha=0.1,
        label=r'Range of sample paths (1\% to 99\%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_defined[qidx, :, didx],
                label=r'{}\% quantile'.format(lb_quantiles[qidx]))
    plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(econ_def_labels[de])
    plt.legend(loc='upper left')
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '.pdf')
    plt.close()

# --------------------------------------------------------------------------- #
# Uncertainty quantification if certain parameters (like climate sensitivity) are pseudostates. 
"""if Parameters.UQflag:  #FV: UQ=Uncertainty quantification. I do not see where the UQflag is set
    print("-" * terminal_size_col)
    print(r"Starting states of the pseudo states are drawn from the uniform "
          "distributions.")
    # For the UQ analysis, the starting state of the all pseudo states must be
    # different. Here we populate random pseudo states where all numbers are
    # drawn from the corresponding uniform distribution.
    N_episode_length_euler = Parameters.N_episode_length
    starting_state_batch_euler = tf.tile(starting_state, [N_sim_batch, 1])

    # Store the index of uncertain parameters, which is used to initialize the
    # starting state
    pseudo_states = [
        'mufx', 'Sfx',
        'rhox', 'gammax', 'psix', 'pi2x'
        # , 'f2xco2x', 'psix', 'pi2x', 'rx', 'varrhox', 'varsigmax',
        # 't2xco2x'
    ]
    N_pseudos = len(pseudo_states)

    pseudo_batch = np.random.uniform(
        # Lower boundary
        low=[Parameters.muf_lower, Parameters.Sf_lower,
             Parameters.rho_lower, Parameters.gamma_lower,
             Parameters.psi_lower, Parameters.pi2_lower
             # Parameters.f2xco2_lower,
             # Parameters.r_lower,
             # Parameters.varrho_lower, Parameters.varsigma_lower,
             # Parameters.t2xco2_lower
             ],
        # Upper boundary
        high=[Parameters.muf_upper, Parameters.Sf_upper,
              Parameters.rho_upper, Parameters.gamma_upper,
              Parameters.psi_upper, Parameters.pi2_upper
              # Parameters.f2xco2_upper,
              # Parameters.r_upper,
              # Parameters.varrho_upper, Parameters.varsigma_upper,
              # Parameters.t2xco2_upper
              ],
        size=(N_sim_batch, N_pseudos)
    )

    uncertain_param_idx_list = []
    for pstate in pseudo_states:
        uncertain_param_idx_list.append(Parameters.states.index(pstate))

    # Populate pseudo states, which are uniformly distributed, in batch
    for pidx, sidx in enumerate(uncertain_param_idx_list):
        starting_state_batch_euler = tf.tensor_scatter_nd_update(
            starting_state_batch_euler,
            [[idx, sidx] for idx in range(N_sim_batch)],
            [pseudo_batch[idx, pidx] for idx in range(N_sim_batch)])

    simulation_starting_state_batch_euler = tf.tile(tf.expand_dims(
        starting_state_batch_euler, axis=0), [N_episode_length_euler, 1, 1])
"""
# import ipdb; ipdb.set_trace() #this sets a breakpoint, to be investigated with the debugging package ipdb
# --------------------------------------------------------------------------- #
"""print("-" * terminal_size_col)
print(r"Compute the Euler discrepancies for {} years in {} simulation "
      "batch".format(Parameters.N_episode_length, N_sim_batch))
# --------------------------------------------------------------------------- #
# Simulate the economy for N_sim_batch times to compute the collection of state and policy episodes
#FV: this is done for pseudostates and uses code from above. I cannot run this because I have no pseudostates. It would make sense to do it without pseudostates though
state_episode_batch_euler = run_episode(simulation_starting_state_batch_euler)
# Policy variables for N_sim_batch times
policy_state_episode_batch_euler = np.empty(
    shape=[N_episode_length_euler, N_sim_batch, N_policy_state],
    dtype=np.float32)
for tidx in range(N_episode_length_euler):
    policy_state_episode_batch_euler[tidx, :, :] = Parameters.policy(
        state_episode_batch_euler[tidx, :, :])

# Take the absolute numerical value of each element
for tidx in range(N_episode_length_euler):
    state_batch = state_episode_batch_euler[tidx, :, :]
    policy_state_batch = policy_state_episode_batch_euler[tidx, :, :]
    euler_discrepancy_df = pd.DataFrame(
        Equations.equations(state_batch, policy_state_batch)).abs()
    state_episode_df = pd.DataFrame(
        {s: getattr(State, s)(state_batch) for s in Parameters.states})
    policy_episode_df = pd.DataFrame(
        {ps: getattr(PolicyState, ps)(policy_state_batch)
         for ps in Parameters.policy_states})
    defined_episode_df = pd.DataFrame(
        {de: getattr(Definitions, de)(state_batch, policy_state_batch)
         for de in econ_defs})

    # Initialize each dataframe
    if tidx == 0:
        euler_discrepancies_df = euler_discrepancy_df
        state_episodes_df = state_episode_df
        policy_episodes_df = policy_episode_df
        defined_episodes_df = defined_episode_df
    else:
        euler_discrepancies_df = pd.concat([
            euler_discrepancies_df, euler_discrepancy_df], axis=0)
        state_episodes_df = pd.concat([
            state_episodes_df, state_episode_df], axis=0)
        policy_episodes_df = pd.concat([
            policy_episodes_df, policy_episode_df], axis=0)
        defined_episodes_df = pd.concat([
            defined_episodes_df, defined_episode_df], axis=0)

# --------------------------------------------------------------------------- #
# Print the Euler approximation errors
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Print the Euler discrepancies")
# import ipdb; ipdb.set_trace()
print(euler_discrepancies_df.describe(
    percentiles=err_percentiles, include='all'))

# Compute the mean and the max Euler errors for all Euler equations
euler_discrepancies_df_melt = pd.melt(euler_discrepancies_df)
# Convert the Euler equation errors in log10
# euler_discrepancies_df_melt_log10 = np.log10(euler_discrepancies_df_melt['value'])
print("-" * terminal_size_col)
print("Print the mean and the max Euler discrepancies")
print(euler_discrepancies_df_melt.describe(percentiles=err_percentiles))

# Save all relevant quantities along the trajectory
euler_discrepancies_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/simulated_euler_discrepancies_describe_"
    + str(N_episode_length_euler) + 'years_' + str(N_sim_batch) + "batch.csv",
    index=False, float_format='%.3e')

euler_discrepancies_df_melt.describe(percentiles=err_percentiles).to_csv(
    Parameters.LOG_DIR + "/simulated_euler_discrepancies_describe_melt_"
    + str(N_episode_length_euler) + 'years_' + str(N_sim_batch) + "batch.csv",
    index=False, float_format='%.3e')

print("-" * terminal_size_col)
print("Finished calculating Euler discrepancies")
"""
# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
"""print("-" * terminal_size_col)
print(r"Compute the statistics of SCC etc. from {} simulation points".format(
    N_episode_length * N_sim_batch))
# --------------------------------------------------------------------------- #
sccidx = econ_defs.index('scc')
idx2020 = 2020 - ts_beg
idx2050 = 2050 - ts_beg
idx2100 = 2100 - ts_beg
scc2020_df = pd.DataFrame(defined_episode_batch_scaled[idx2020, :, sccidx])
scc2020_log10_df = pd.DataFrame(
    np.log10(defined_episode_batch_scaled[idx2020, :, sccidx]))
scc2050_df = pd.DataFrame(defined_episode_batch_scaled[idx2050, :, sccidx])
scc2050_log10_df = pd.DataFrame(
    np.log10(defined_episode_batch_scaled[idx2050, :, sccidx]))
scc2100_df = pd.DataFrame(defined_episode_batch_scaled[idx2100, :, sccidx])
scc2100_log10_df = pd.DataFrame(
    np.log10(defined_episode_batch_scaled[idx2100, :, sccidx]))

print(r"SCC in 2020")
print(scc2020_df.describe(percentiles=err_percentiles, include='all'))
scc2020_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/scc2020_describe_" + str(ts_beg) + '-' + str(ts_end)
    + ".csv", index=True, float_format='%.3f')

print(r"SCC in 2020 in log 10")
print(scc2020_log10_df.describe(percentiles=err_percentiles, include='all'))
scc2020_log10_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/scc2020_log10_describe_" + str(ts_beg) + '-'
    + str(ts_end) + ".csv", index=True, float_format='%.3f')

print(r"SCC in 2050")
print(scc2050_df.describe(percentiles=err_percentiles, include='all'))
scc2050_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/scc2050_describe_" + str(ts_beg) + '-'
    + str(ts_end) + ".csv", index=True, float_format='%.3f')

print(r"SCC in 2050 in log 10")
print(scc2050_log10_df.describe(percentiles=err_percentiles, include='all'))
scc2050_log10_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/scc2050_log10_describe_" + str(ts_beg) + '-'
    + str(ts_end) + ".csv", index=True, float_format='%.3f')

print(r"SCC in 2100")
print(scc2100_df.describe(percentiles=err_percentiles, include='all'))
scc2100_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/scc2100_describe_" + str(ts_beg) + '-' + str(ts_end)
    + ".csv", index=True, float_format='%.3f')

print(r"SCCin 2100 in log 10")
print(scc2100_log10_df.describe(percentiles=err_percentiles, include='all'))
scc2100_log10_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/scc2100_log10_describe_" + str(ts_beg) + '-'
    + str(ts_end) + ".csv", index=True, float_format='%.3f')
"""
print("Exit post processing")

# import ipdb; ipdb.set_trace()
