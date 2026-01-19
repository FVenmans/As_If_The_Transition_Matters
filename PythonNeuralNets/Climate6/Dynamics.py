#6 states model
import tensorflow as tf
import math
import itertools
import Definitions
import State
import Parameters
#from Parameters import sigma_ax, rho, alpha, beta
import PolicyState

#import parameters 
#production function
delta = Parameters.delta
alpha = Parameters.alpha
epsilon_f = Parameters.epsilon_f
varphi_f = Parameters.varphi_f
varphi_E = Parameters.varphi_E
sigma0 = Parameters.sigma0
varphi_c = Parameters.varphi_c
varphi_d = Parameters.varphi_d
xi = Parameters.xi
chi = Parameters.chi
chi_s = Parameters.chi_s
A0 = Parameters.A0
L0 = Parameters.L0
vareps = Parameters.vareps
vareps2= Parameters.vareps2

#utility and discounting
eta = Parameters.eta
r = Parameters.r
g = Parameters.g
g_L = Parameters.g_L
g_tau = Parameters.g_tau
RRA = Parameters.RRA
dt = Parameters.dt

#Technical change, emissions and abatement costs
omega = Parameters.omega
varrho = Parameters.varrho
g_epsilon = Parameters.g_epsilon
M0 = Parameters.M0
P_BAU = Parameters.P_BAU

#stranding costs
psi1 = Parameters.psi1
psi2 = Parameters.psi2
psi3 = Parameters.psi3
g_psi = Parameters.g_psi

#climate and damages
zeta = Parameters.zeta
gamma = Parameters.gamma

#scaling to obtain states and policies close to [0,1]
kf_scale = Parameters.kf_scale
ks_scale = Parameters.ks_scale
v_scale  = Parameters.v_scale
id_scale = Parameters.id_scale
ic_scale = Parameters.ic_scale

#Uncertainty
Sigmak = Parameters.Sigmak
SigmaS = Parameters.SigmaS

#useful constants 
nu=(1-RRA)/(1-eta)


#code from sudden stop model 
# shocks
# pseudo-random probabilities (set this in config/run)
shocks_kf = [x * math.sqrt(2.0) * math.sqrt(Sigmak*dt) for x in [-1.224744871, 0.0, +1.224744871]]   
probs_kf  = [x / math.sqrt(math.pi) for x in [0.2954089751, 1.181635900, 0.2954089751]]

shocks_kc = [x * math.sqrt(2.0) * math.sqrt(Sigmak*dt) for x in [-1.224744871, 0.0, +1.224744871]]
probs_kc  = probs_kf

shocks_kd = [x * math.sqrt(2.0) * math.sqrt(Sigmak*dt) for x in [-1.224744871, 0.0, +1.224744871]]
probs_kd  = probs_kf
   
shocks_S = [x * math.sqrt(2.0) * math.sqrt(SigmaS*dt) for x in [-1.224744871, 0.0, +1.224744871]]
probs_S  = probs_kd


shock_values = tf.constant(list(itertools.product(shocks_kf, shocks_kc, shocks_kd, shocks_S)))
shock_probs = tf.constant([ p_1 * p_2 * p_3 *p_4 for p_1, p_2, p_3, p_4 in list(itertools.product(probs_kf, probs_kd, probs_kc, probs_S))])

#add the jump
#column=tf.zeros([81,1], dtype=tf.float32)
#row=tf.constant([[0,0,0,0,xi1]], dtype=tf.float32)
#shock_values=tf.concat([shock_values,column], axis=1)
#shock_values=tf.concat([shock_values,row], axis=0)
#shock_probs= tf.concat([shock_probs*(1-lambdaA),[lambdaA]], axis=0)
#shock_values= a nxp matrix with n realizations of combinations of shocks = e in matlab 
#shock_probs= a n vector of probabilities =w in matlab

if Parameters.expectation_type == 'monomial':
    shock_values, shock_probs = State.monomial_rule([math.sqrt(Sigmak*dt), math.sqrt(Sigmak*dt), math.sqrt(Sigmak*dt), math.sqrt(SigmaS * dt)]) 

def total_step_random(s,p): #s=prev_state, p=policystate
    ar = AR_step(s,p)
    shock = shock_step_random(s,p)
    policy = policy_step(s,p)
    #total = ar + shock + policy #original code, to be combined with a function augment_state(state) for multiplicative errors
    total = ar * tf.math.exp(shock) + policy
    #total = tf.where(total > 0, total, tf.constant(0.00001, dtype=tf.float32)) # avoid errors with negative dirty capital This gave infinite adjustment costs.
    total = 0.001 + tf.nn.softplus(200 * (total - 0.001)) / 200 
    return augment_state(total)  

# same as above, but non randomized shock, but rather the same shock for each realization
def total_step_spec_shock(s,p, shock_index):
    ar = AR_step(s,p)
    shock = shock_step_spec_shock(s,p, shock_index)
    policy = policy_step(s,p)
    #total = ar + shock + policy
    total = ar * tf.math.exp(shock) + policy
    total = tf.where(total > 0.00001, total, tf.constant(0.00001, dtype=tf.float32))# avoid errors with negative dirty capital
    return augment_state(total)  

def augment_state(s):
    state = State.update(s, "y_kf", Definitions.y_kf(s,None))
    state = State.update(s, "y_kc", Definitions.y_kc(s,None))
    state = State.update(s, "y_kd", Definitions.y_kd(s,None))
    state = State.update(s, "y_S", Definitions.y_S(s,None))
    return state

def AR_step(s,p):  #I added p as a second argument to be able to use definitions
    # autoregressive components
    ar_step = tf.zeros_like(s)
    ar_step = State.update(ar_step, "kf", State.kf(s) * (1-delta-g_L-g) ** dt) 
    ar_step = State.update(ar_step, "kc", State.kc(s) * (1-delta-g_L-g) ** dt)
    ar_step = State.update(ar_step, "kd", State.kd(s) * (1-delta-g_L-g) ** dt)
    ar_step = State.update(ar_step, "T",  State.T(s))  
    ar_step = State.update(ar_step, "ks", State.ks(s) * (1-delta-g_L-g) ** dt) 
    ar_step = State.update(ar_step, "tau", 1 - (1-State.tau(s)) * tf.math.exp(-g_tau * dt))
    #ar_step = State.update(ar_step, "yT", Parameters.rho_yT * tf.math.log(State.yT(s)) + (1- Parameters.rho_yT) * tf.math.log( Parameters.yT_bar))
    return ar_step

def shock_step_random(s,p): #I added p as a second argument to be able to use definitions
    shock_step = tf.zeros_like(s) 
    random_normals = Parameters.rng.normal([tf.shape(s)[0],4]) #order of shocks: kf,kc,kd,S #original uses s.shape[0],4
    #shock_step = State.update(shock_step, "yT", random_normals[:,1] * Parameters.sigma_yT)
    #      return State.update(shock_step, "delta", random_normals[:,2] * Parameters.sigma_delta)
    shock_step = State.update(shock_step, "kf", random_normals[:,0] * math.sqrt(Sigmak * dt) )
    shock_step = State.update(shock_step, "kc", random_normals[:,1] * math.sqrt(Sigmak * dt) ) 
    shock_step = State.update(shock_step, "kd", random_normals[:,2] * math.sqrt(Sigmak * dt) )
    shock_step = State.update(shock_step, "T" , random_normals[:,3] * math.sqrt(SigmaS * dt) )
    return shock_step

def shock_step_spec_shock(s, p, shock_index):
    # Use a specific shock - for calculating expectations
    shock_step = tf.zeros_like(s) 
    #shock_step = State.update(shock_step,"rf", tf.repeat(shock_values[shock_index,0], s.shape[0]))
    #shock_step = State.update(shock_step,"yT", tf.repeat(shock_values[shock_index,1], s.shape[0]))
    shock_step = State.update(shock_step,"kf", tf.repeat(shock_values[shock_index,0] , tf.shape(s)[0])) #original s.shape[0]
    shock_step = State.update(shock_step,"kc", tf.repeat(shock_values[shock_index,1] , tf.shape(s)[0]))
    shock_step = State.update(shock_step,"kd", tf.repeat(shock_values[shock_index,2] , tf.shape(s)[0]))
    shock_step = State.update(shock_step,"T",  tf.repeat(shock_values[shock_index,3] , tf.shape(s)[0]))
    return    shock_step

def policy_step(s, p):
    #This function gives the policy that is added to each stock 
    policy_step = tf.zeros_like(s) 
    policy_step = State.update(policy_step, "kf", (PolicyState.if_(p)  - chi * Definitions.a_f(s,p) * PolicyState.if_(p)) / kf_scale * dt) 
    policy_step = State.update(policy_step, "kc", (Definitions.ic(s,p) - chi * Definitions.a_c(s,p) * Definitions.ic(s,p)) * dt)
    policy_step = State.update(policy_step, "kd", (Definitions.id(s,p) - chi * Definitions.a_d_pos(s,p) * Definitions.id(s,p)) * dt ) 
    policy_step = State.update(policy_step, "ks", (tf.nn.softplus((-Definitions.id(s,p)+vareps2)*vareps) / vareps -vareps2) / ks_scale * dt)
    policy_step = State.update(policy_step, "T" , Definitions.P(s,p) * zeta * dt) # emissions are scaled because T=zeta S
    return policy_step 
