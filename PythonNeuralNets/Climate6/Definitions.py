# 6 states model
import Parameters
import PolicyState
import State
import sys
import tensorflow as tf

#import parameters (see constant/Climate5.yaml)
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

#initial states
T0 = Parameters.T0  
kd0 = Parameters.kd0

#scaling to obtain states and policies close to [0,1]
kf_scale = Parameters.kf_scale #not used for now.
ks_scale = Parameters.ks_scale
v_scale  = Parameters.v_scale
id_scale = Parameters.id_scale
ic_scale = Parameters.ic_scale
lambdaks_scale = Parameters.lambdaks_scale

#useful constants
Kd0 = kd0 * A0 * L0

###Scaling
def S(s,_):
    return State.T(s) / zeta
def kf(s,_):
    return State.kf(s) * kf_scale
def ks(s,_):
    return State.ks(s) * ks_scale
def lambdaS(_,p):
    return PolicyState.lambdaT(p) * zeta
def lambdaks(_,p):
    return PolicyState.lambdaks(p) * lambdaks_scale
def id(_,p):
    return PolicyState.id(p) * id_scale
def ic(_,p):
    return PolicyState.ic(p) * ic_scale
def v(s,p): #value function (PolicyState V is scaled to be close to 1)
    return PolicyState.V(p) * v_scale

###proper definitions

#investment ratios (adj costs capped to avoid infty at zero capital. This should never be binding on optimal path.)
def a_f(s,p):
    return tf.minimum(PolicyState.if_(p) / kf(s,p), 0.45) # marginal effect of investment is 1-2 chi a_f=10% at boundary. chi a_f is 38% in first year. 
def a_c(s,p):
    return tf.minimum(ic(s,p) / State.kc(s), 0.45)
def a_d_pos(s,p):
    return tf.minimum(tf.nn.relu(id(s,p)) / State.kd(s), 0.45)
def a_d_neg(s,p):
    return - tf.minimum(tf.nn.relu(-id(s,p)) / State.kd(s), 0.45)

def t(s,p):
    return - tf.math.log(1.0-State.tau(s)) / g_tau
#def t_tau(s,p):
#    return 1.0 / g_tau / (1.0-State.tau(s))
def M(s,p):
    M = (M0 + P_BAU* t(s,p) - S(s,p) + T0/zeta) / M0
    M = 1 + tf.nn.relu(M-1) # t and S are not necessarily coherent, therefore values below 1 occur for low t and high S
    return M      

def epsilon(s,p):
    return 1.0 - ((1.0-omega) * M(s,p) **-varrho + omega * tf.math.exp(-g_epsilon * t(s,p))) / sigma0
def epsilon_M(s,p): 
    return 1.0 / sigma0 * (1.0-omega) * varrho * M(s,p)**(-varrho-1.0) 
#def epsilon_t(s,p):
#    return ((1.0-omega) * varrho * M(s,p) ** (-varrho-1.0) * P_BAU / M0 + omega * tf.math.exp(-g_epsilon * t(s,p)) * g_epsilon)/sigma0

def Ebase(s,p):
    return (varphi_c * State.kc(s)**epsilon(s,p) + varphi_d * (State.kd(s) + xi)**epsilon(s,p))
def E(s,p):
    return  Ebase(s,p) ** (1.0/epsilon(s,p))
def E_kc(s,p):
    return Ebase(s,p)**(1.0/epsilon(s,p)-1.0) * varphi_c * State.kc(s)**(epsilon(s,p)-1.0) 
def E_kd(s,p):
    return Ebase(s,p)**(1.0/epsilon(s,p)-1.0) * varphi_d * (State.kd(s) + xi)**(epsilon(s,p)-1.0)
def E_epsilon(s,p):
    return E(s,p) * (- tf.math.log(Ebase(s,p)) / epsilon(s,p)**2 + (varphi_c * State.kc(s) ** epsilon(s,p) * tf.math.log(State.kc(s)) + varphi_d * (State.kd(s) + xi) ** epsilon(s,p) * tf.math.log(State.kd(s) + xi)) / epsilon(s,p) / Ebase(s,p))

def ybase(s,p):
    return (varphi_f * kf(s,p)** epsilon_f + varphi_E * E(s,p)** epsilon_f)
def y(s,p=None):
    return ybase(s,p) ** (alpha/epsilon_f) * tf.math.exp(-0.5 * gamma * State.T(s)**2) 
def y_E(s,p):
    return alpha * y(s,p) * varphi_E * E(s,p)**(epsilon_f-1.0) / ybase(s,p) 
def y_kf(s,p):
    return alpha * y(s,p) * varphi_f * kf(s,p)**(epsilon_f-1.0) / ybase(s,p) 
def y_kc(s,p):
    return y_E(s,p) * E_kc(s,p)
def y_kd(s,p):
    return  y_E(s,p) * E_kd(s,p)
def y_S(s,p):
    return -y(s,p) * gamma * zeta**2 * S(s,p) 
def y_epsilon(s,p):
    return y_E(s,p) * E_epsilon(s,p)

def ks_ratio(s,p):
    return ks(s,p) / (State.kd(s) +  ks(s,p))
def P(s,p):
    return A0 * L0 * tf.math.exp((g + g_L - g_psi)*t(s,p)) * (psi1 * (State.kd(s) + ks(s,p))  / psi2 / Kd0 * (tf.math.exp(-psi2 * Kd0 * ks_ratio(s,p)) - tf.math.exp(-psi2 * Kd0)) + psi3  * State.kd(s))
def P_ks(s,p):
    kd_ratio = State.kd(s) / (State.kd(s) +  ks(s,p))
    return  A0 * L0 * tf.math.exp((g + g_L - g_psi)*t(s,p)) * (-psi1 / psi2 / Kd0 * (tf.math.exp(-psi2 * Kd0 * ks_ratio(s,p)) - tf.math.exp(-psi2 * Kd0)) + psi1 * kd_ratio * tf.math.exp(-psi2 * Kd0 * ks_ratio(s,p))) 
def P_kd(s,p):
    return A0 * L0 * tf.math.exp((g + g_L - g_psi)*t(s,p)) * (psi1 / psi2 * tf.math.exp(-psi2 * Kd0 * ks_ratio(s,p)) * (1/Kd0 + psi2 * ks_ratio(s,p)) - psi1/psi2/Kd0 * tf.math.exp(-psi2 * Kd0) + psi3)
#def P_tau(s,p):
#    return P(s,p) * (g + g_L - g_psi ) * t_tau(s,p)

def c(s,p):
    consumption = y(s,p) - PolicyState.if_(p) - ic(s,p) - tf.nn.softplus(vareps * id(s,p)) / vareps - chi_s * a_d_neg(s,p) * id(s,p) 
    consumption = 0.03 + tf.nn.softplus(10 * (consumption - 0.03)) / 10 #avoid negative consumption, while keeping gradient in negative domain and avoiding explosive marg utility 
    return consumption 
def U_c(s,p): 
    return c(s,p) **(-eta)
#def c_tau(s,p):
#    return  y_epsilon(s,p) * epsilon_t(s,p) * t_tau(s,p)                         

def lambdakf(s,p): 
    return U_c(s,p) / (1.0 - 2.0 * chi * a_f(s,p)) 
def lambdakc(s,p):
    return U_c(s,p) / (1.0 - 2.0 * chi * a_c(s,p)) 
def lambdakd(s,p) :
    numerator = (U_c(s,p) * (tf.nn.sigmoid(vareps * id(s,p)) + 2.0 * chi_s * a_d_neg(s,p)) + 
               lambdaks(s,p) * tf.nn.sigmoid(((-id(s,p)+vareps2) * vareps)))  
    denominator = 1.0 - 2.0 * chi * a_d_pos(s,p) 
    return numerator / denominator


               


