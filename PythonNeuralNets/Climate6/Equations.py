#6 states model
import tensorflow as tf
import Definitions
import PolicyState
import State
import Parameters

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
g_psi0 = Parameters.g_psi0
gg_psi = Parameters.gg_psi

#climate and damages
zeta = Parameters.zeta
gamma = Parameters.gamma

#initial states
kf0 = Parameters.kf0
kc0 = Parameters.kc0
kd0 = Parameters.kd0
ks0 = Parameters.ks0
T0  = Parameters.T0  #in TtCO2
tau0= Parameters.tau0

#bounds on states
kf_max = Parameters.kf_max
kf_min = Parameters.kf_min
kc_max = Parameters.kc_max
kc_min = Parameters.kc_min
kd_max = Parameters.kd_max
kd_min = Parameters.kd_min
T_max  = Parameters.T_max
lambdaks0_min = Parameters.lambdaks0_min
lambdaks_ss_max = Parameters.lambdaks_ss_max
bellman_weight = Parameters.bellman_weight

#useful constants 
nu=(1.0-RRA)/(1.0-eta)

#Definitions

def v_nu(s,p):
    return (-Definitions.v(s,p)) **nu        #minus the value function to avoid complex intermediary results
def v_v_kf(s,p):
    return ((-Definitions.v(s,p))**(nu-1.0) *  
            (Definitions.U_c(s,p) * Definitions.y_kf(s,p) * dt
             + Definitions.lambdakf(s,p) * ((1.0- delta - g_L - g)**dt + chi * Definitions.a_f(s,p)**2.0 * dt)))
def v_v_kc(s,p):
    return ((-Definitions.v(s,p))**(nu-1.0) *  
            (Definitions.U_c(s,p) * Definitions.y_kc(s,p) * dt 
            + Definitions.lambdakc(s,p) * ((1.0 - delta - g_L - g) ** dt + chi * Definitions.a_c(s,p)**2.0 * dt)))
def v_v_kd(s,p):
    return ((-Definitions.v(s,p))**(nu-1.0) *  
            (Definitions.U_c(s,p) * (Definitions.y_kd(s,p) + chi_s * Definitions.a_d_neg(s,p)**2 ) * dt
            + Definitions.lambdakd(s,p) * ((1.0 - delta - g_L - g)**dt + chi * Definitions.a_d_pos(s,p)**2.0 * dt)   
            + Definitions.lambdaS(s,p) * Definitions.P_kd(s,p) * dt))
def v_v_ks(s,p):
    return ((-Definitions.v(s,p))**(nu-1.0) *  
            (Definitions.lambdaks(s,p) * (1.0 - delta - g_L - g) **dt + Definitions.lambdaS(s,p) * Definitions.P_ks(s,p) * dt))
def v_v_S(s,p):
    return ((-Definitions.v(s,p))**(nu-1.0) * 
            (Definitions.U_c(s,p) * (Definitions.y_T(s,p) * zeta ) * dt   # change to Definitions.y_T(s,p) * zeta
            + Definitions.lambdaS(s,p)))

def equations(s,p):
    E_t = State.E_t_gen(s,p) #E_t_gen calculates the expected value at t+1. It takes in a second function (called evalFun in its code), which is passed to E_t using an unnamed function lambda.   
    E_v_nu = E_t(v_nu) # this was written as E_t(lambda snext, pnext: v_nu(snext, pnext)) but lambda functions are slow with tf.function
    def get_lambdaT(snext, pnext):
        return PolicyState.lambdaT(pnext)
    def get_kc(snext, pnext):
        return State.kc(snext)
    def get_kd(snext,pnext):
        return State.kd(snext)
    def get_T(snext,pnext):
        return State.T(snext)   
    E_lambdaT = E_t(get_lambdaT)
    E_kc = E_t(get_kc)
    E_kd = E_t(get_kd)
    E_T = E_t(get_T)
    #E_lambdaT = E_t(lambda snext, pnext: PolicyState.lambdaT(pnext))
    #E_kc = E_t(lambda snext, pnext:State.kc(snext))
    #E_kd = E_t(lambda snext, pnext:State.kd(snext))
    #E_T = E_t(lambda snext, pnext:State.T(snext))
    # FOC    
    loss_dict = {} #defines an empty dictionary  
    #FOC
    loss_dict['eq_kf'] = (tf.math.exp(-(r-g) * dt) * E_v_nu  ** (1.0/nu - 1.0) * E_t(v_v_kf) - Definitions.lambdakf(s,p) )  / 0.7 #I divide by lambdak (0.7) to have a similar scale for all constraints (lambdakd can include zero). 
    loss_dict['eq_kc'] = (tf.math.exp(-(r-g) * dt) * E_v_nu ** (1.0/nu - 1.0) * E_t(v_v_kc) - Definitions.lambdakc(s,p) ) / 0.7  
    loss_dict['eq_kd'] = (tf.math.exp(-(r-g) * dt) * E_v_nu ** (1.0/nu - 1.0) * E_t(v_v_kd) - Definitions.lambdakd(s,p) ) / 0.7  #switch back to Definitions.lambdakd?
    loss_dict['eq_ks'] = (tf.math.exp(-(r-g) * dt) * E_v_nu ** (1.0/nu - 1.0) * E_t(v_v_ks) - Definitions.lambdaks(s,p) ) / 0.7 #lambdaks starts at -1.1 and goes to -1e-10
    loss_dict['eq_S']  = (tf.math.exp(-(r-g) * dt) * E_v_nu ** (1.0/nu - 1.0) * E_t(v_v_S)  - Definitions.lambdaS(s,p)) / Definitions.lambdaS(s,p) * 10 # scaled up because over much longer horizon.  
    loss_dict['eq_Bellman'] = (-Definitions.v(s,p) + Definitions.c(s,p) ** (1.0-eta) / (1.0-eta) * dt - tf.math.exp(-(r-g) * dt) * E_v_nu **(1.0/nu)) / Definitions.v(s,p) * bellman_weight # v is large, around 200 and policies affect it very mildly, so  extra weight may be needed when RRA is not 1.35. 
    
    ##########transversality conditions as maximum growth rate on shadow prices
    loss_dict['upper_lambdaT']  = tf.nn.relu( E_lambdaT / PolicyState.lambdaT(p) - (1.0 + (r-g) * dt)) #shadow price (negative) should decrease at lower rate than the discount. Penality if SCC{t+1}/SCCt>1+r <=> SCCt+1/SCCt-(1+r)>0
    loss_dict['lower_lambdaT']  = tf.nn.relu(-E_lambdaT / PolicyState.lambdaT(p) + (1.0 - 0.007 * dt))  #penalty if SCCt+1/SCCt < 0.995 per year <=> 0 < -SCC_t+1 / SCCt + 0.998 (note that the endog TC component should decrease over time)
    loss_dict['upper_lambdakf'] = tf.nn.relu( E_t(Definitions.lambdakf) / Definitions.lambdakf(s,p) - (1.0 + 0.04 * dt)) #penalty if lambdakf increases faster than 4%. without uncertainty max increase 2%       
    loss_dict['lower_lambdakf'] = tf.nn.relu(-E_t(Definitions.lambdakf) / Definitions.lambdakf(s,p) + (1.0 - 0.02 * dt)) #penalty if lambdakf decreases faster than 2%. Without uncertainty, never a decrease
    loss_dict['upper_cons']     = tf.nn.relu( E_t(Definitions.c) / Definitions.c(s,p) - 1.01) #consumption is declining (increasing damages and abatement costs), unless capital above initial steady state
    loss_dict['lower_cons']     = tf.nn.relu(-E_t(Definitions.c) / Definitions.c(s,p) + (1.0 - 0.03 * dt)) #consumption should not decline faster than 3% per year (increasing damages and abatement costs) (consumption is around 1, close to percentage deviation)
    
    ##########transversality conditions imposing a steady state for state and policy variables
    loss_dict['upper_kf'] = tf.nn.relu(E_t(Definitions.kf) - kf_max)
    loss_dict['lower_kf'] = tf.nn.relu(kf_min - E_t(Definitions.kf)) 
    loss_dict['upper_kc'] = tf.nn.relu(E_kc - kc_max)
    loss_dict['lower_kc'] = tf.nn.relu(kc_min - E_kc)
    loss_dict['upper_kd'] = tf.nn.relu(E_kd - kd_max) 
    loss_dict['lower_kd'] = tf.nn.relu(kd_min - E_kd)
    loss_dict['upper_T' ] = tf.nn.relu(E_T - T_max) * tf.nn.relu(Definitions.id(s,p)) * 100 #(E_t(lambda snext, pnext:State.T(snext))-T_max) doesn't work because it depends on kd_t, which is fixed within a batch (it depends on id_t-1). Error is zero when id is already negative.
    loss_dict['upper_lambdaks_ss'] = tf.nn.relu(Definitions.lambdaks(s,p) - lambdaks_ss_max) * tf.nn.relu(State.tau(s) - 0.8) / 10 #steady state below 0.01 after 110 years
    loss_dict['upper_id_ss'] = tf.nn.relu(Definitions.id(s,p)) * tf.nn.relu(State.tau(s) - 0.8) * 10 #steady state id is zero after 110 years. This also sets lambdakd = 0.5 lambdaks + 0.5 U_c

    ##########conditions on initial policies to help satisfy the transversality conditions
    loss_dict['lower_lambdaks0'] = tf.nn.relu(1 - Definitions.lambdaks(s,p) / lambdaks0_min) * tf.nn.relu(1-Definitions.t(s,p))* 10 #penalty if lambdaks is initially too low. Lambdaks is irrelevant after stranding, therefore transversality condition at the end is hard to enforce. 
    
     ########## help with kink outside the optimum
    loss_dict['lambdaks_lambdakd'] = tf.nn.relu(Definitions.lambdaks(s,p) - Definitions.lambdakd(s,p)-0.05) * tf.nn.relu(Definitions.id(s,p)) * 10 #when lambdaks exceeds lambdakd by 0.05, id should be negative. idmax=0.01, hence the scaling. 
    return loss_dict    


    
    ##########transversality conditions imposing limits on growth rates
    #loss_dict['upper_lambdakc'] = tf.nn.relu( E_t(Definitions.lambdakc) / Definitions.lambdakc(s,p) - (1.0 + 0.03 * dt)) * tf.nn.sigmoid((Definitions.t(s,p) - 15) * 5)                                   
    #loss_dict['lower_lambdakc'] = tf.nn.relu(-E_t(Definitions.lambdakc) / Definitions.lambdakc(s,p) + (1.0 - 0.03 * dt)) * tf.nn.sigmoid((Definitions.t(s,p) - 15) * 5) # after 15 years, lambdakc cannot decrease by more than 3% per year
    #loss_dict['upper_id'] = tf.nn.relu( Definitions.id(s,p)-0.001) * 100  #id should be below 0.001. 
    #loss_dict['lower_id'] = tf.nn.relu(-Definitions.id(s,p)-0.002) * tf.cast(Definitions.t(s,p) > 4, dtype=tf.float32) * 100 #id should be above -0.002 after year 3.(stochastic increase in temperature may require reduction in kd. 
    #loss_dict['upper_lambdakd'] = tf.nn.relu( E_t(Definitions.lambdakd) - Definitions.lambdakd(s,p) - 0.05 * dt) * tf.nn.sigmoid((Definitions.t(s,p) - 20) * 5) #after 20 years, lambdakd cannot increase by more than 0.05 per year. This is very sensitive at long run low levels of kd. Would imposing id=0 be more stable?
    #loss_dict['lower_lambdakd'] = tf.nn.relu( Definitions.lambdakd(s,p) - E_t(Definitions.lambdakd) - 0.05 * dt) * tf.nn.sigmoid((Definitions.t(s,p) - 20) * 5) #after 20 years, lambdakd cannot decrease at by more than -0.05 per year 
    ##########transversality conditions imposing a steady state for state variables
    '''loss_dict['upper_if0'] = tf.nn.relu(PolicyState.if_(p)  / if0_max - 1) * tf.nn.relu(3-Definitions.t(s,p))*10 #initial condition for year 0, 1 and 2.
    loss_dict['lower_if0'] = tf.nn.relu(1 - PolicyState.if_(p)  / if0_min) * tf.nn.relu(3-Definitions.t(s,p))*10 
    loss_dict['upper_ic0'] = tf.nn.relu(Definitions.ic(s,p) / ic0_max - 1) * tf.nn.relu(3-Definitions.t(s,p))*10 
    loss_dict['lower_ic0'] = tf.nn.relu(1 - Definitions.ic(s,p) / ic0_min) * tf.nn.relu(3-Definitions.t(s,p))*10 
    loss_dict['upper_id0'] = tf.nn.relu( Definitions.id(s,p) - id0_max )    * tf.nn.relu(1-Definitions.t(s,p))*500 #id0_max is too small to express in % deviation. 
    loss_dict['lower_id0'] = tf.nn.relu(-Definitions.id(s,p) + id0_min )    * tf.nn.relu(1-Definitions.t(s,p))*500 #absolute difference because id can be both pos and negative. '''
