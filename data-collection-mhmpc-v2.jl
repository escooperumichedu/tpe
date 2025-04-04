# Benzene alkylation with ethlyene
using JuMP, Ipopt, Plots, OrdinaryDiffEq, LinearAlgebra, DataFrames, CSV

# global mhmpc_src_instance = 261

# while mhmpc_src_instance <= 299

N = 10 # Control horizon
dt = 30 # Sampling time of the system
Nd = 10
dtd = 30
P = 1

global tau = 100

global s_path = 0.01
global cpu_max = 10.0 # Maximum cpu time for Ipopt
global dual_inf_tol = Float64(1 * 10^(0))
global opt_tol = Float64(1 * 10^(0))
global constr_viol_tol = Float64(1 * 10^(0))
global compl_inf_tol = Float64(1 * 10^(0))

global c_cpu_max = 10.0 # Maximum cpu time for Ipopt
global c_dual_inf_tol = Float64(1 * 10^(0))
global c_opt_tol = Float64(1 * 10^(0))
global c_constr_viol_tol = Float64(1 * 10^(0))
global c_compl_inf_tol = Float64(1 * 10^(0))

block_size = Int(dtd / dt) - 1
k_indices = Int[]  # Initialize an empty array

# Loop to generate the pattern until N - 1
for start_k in 0:(dtd/dt):N-1
    append!(k_indices, start_k:start_k+block_size-1)
end

global k_indices = filter(x -> x < N, k_indices)

mutable struct Weights
    v::Float64

    t1::Float64
    t2::Float64
    t3::Float64
    t4::Float64
    t5::Float64

    ca1::Float64
    ca2::Float64
    ca3::Float64
    ca4::Float64
    ca5::Float64

    cb1::Float64
    cb2::Float64
    cb3::Float64
    cb4::Float64
    cb5::Float64

    cc1::Float64
    cc2::Float64
    cc3::Float64
    cc4::Float64
    cc5::Float64

    cd1::Float64
    cd2::Float64
    cd3::Float64
    cd4::Float64
    cd5::Float64

    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
    f5::Float64
    f6::Float64
    f7::Float64
    f8::Float64
    f9::Float64
    f10::Float64
    fr1::Float64
    fr2::Float64

    q1::Float64
    q2::Float64
    q3::Float64
    q4::Float64
    q5::Float64

end

mutable struct Decomposition_Trajectory


    V1::Vector{Float64}
    V2::Vector{Float64}
    V3::Vector{Float64}
    V4::Vector{Float64}
    V5::Vector{Float64}

    T1::Vector{Float64}
    T2::Vector{Float64}
    T3::Vector{Float64}
    T4::Vector{Float64}
    T5::Vector{Float64}

    CA1::Vector{Float64}
    CB1::Vector{Float64}
    CC1::Vector{Float64}
    CD1::Vector{Float64}

    CA2::Vector{Float64}
    CB2::Vector{Float64}
    CC2::Vector{Float64}
    CD2::Vector{Float64}

    CA3::Vector{Float64}
    CB3::Vector{Float64}
    CC3::Vector{Float64}
    CD3::Vector{Float64}

    CA4::Vector{Float64}
    CB4::Vector{Float64}
    CC4::Vector{Float64}
    CD4::Vector{Float64}

    CA5::Vector{Float64}
    CB5::Vector{Float64}
    CC5::Vector{Float64}
    CD5::Vector{Float64}

    F1::Vector{Float64}
    F2::Vector{Float64}
    F3::Vector{Float64}
    F4::Vector{Float64}
    F5::Vector{Float64}
    F6::Vector{Float64}
    F7::Vector{Float64}
    F8::Vector{Float64}
    F9::Vector{Float64}
    F10::Vector{Float64}
    Fr1::Vector{Float64}
    Fr2::Vector{Float64}
    Q1::Vector{Float64}
    Q2::Vector{Float64}
    Q3::Vector{Float64}
    Q4::Vector{Float64}
    Q5::Vector{Float64}

end

mutable struct Solve_Strat_Comparison_Trajectory

    V1::Vector{Float64}
    V2::Vector{Float64}
    V3::Vector{Float64}
    V4::Vector{Float64}
    V5::Vector{Float64}

    T1::Vector{Float64}
    T2::Vector{Float64}
    T3::Vector{Float64}
    T4::Vector{Float64}
    T5::Vector{Float64}

    CA1::Vector{Float64}
    CB1::Vector{Float64}
    CC1::Vector{Float64}
    CD1::Vector{Float64}

    CA2::Vector{Float64}
    CB2::Vector{Float64}
    CC2::Vector{Float64}
    CD2::Vector{Float64}

    CA3::Vector{Float64}
    CB3::Vector{Float64}
    CC3::Vector{Float64}
    CD3::Vector{Float64}

    CA4::Vector{Float64}
    CB4::Vector{Float64}
    CC4::Vector{Float64}
    CD4::Vector{Float64}

    CA5::Vector{Float64}
    CB5::Vector{Float64}
    CC5::Vector{Float64}
    CD5::Vector{Float64}

    F1c::Vector{Float64}
    F2c::Vector{Float64}
    F3c::Vector{Float64}
    F4c::Vector{Float64}
    F5c::Vector{Float64}
    F6c::Vector{Float64}
    F7c::Vector{Float64}
    F8c::Vector{Float64}
    F9c::Vector{Float64}
    F10c::Vector{Float64}
    Fr1c::Vector{Float64}
    Fr2c::Vector{Float64}
    Q1c::Vector{Float64}
    Q2c::Vector{Float64}
    Q3c::Vector{Float64}
    Q4c::Vector{Float64}
    Q5c::Vector{Float64}

    F1d::Vector{Float64}
    F2d::Vector{Float64}
    F3d::Vector{Float64}
    F4d::Vector{Float64}
    F5d::Vector{Float64}
    F6d::Vector{Float64}
    F7d::Vector{Float64}
    F8d::Vector{Float64}
    F9d::Vector{Float64}
    F10d::Vector{Float64}
    Fr1d::Vector{Float64}
    Fr2d::Vector{Float64}
    Q1d::Vector{Float64}
    Q2d::Vector{Float64}
    Q3d::Vector{Float64}
    Q4d::Vector{Float64}
    Q5d::Vector{Float64}

    PIc::Vector{Float64}
    ISEc::Vector{Float64}
    ISCc::Vector{Float64}

    PId::Vector{Float64}
    ISEd::Vector{Float64}
    ISCd::Vector{Float64}
end

ex = Solve_Strat_Comparison_Trajectory(zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P))

function init_sp_and_params()

    global T1_init = 484.4053230565816
    global T2_init = 466.5153873384654
    global T3_init = 465.1711444514916
    global T4_init = 465.023824044236
    global T5_init = 468.14646490098806

    global V1_init = 1.0419090999999996
    global V2_init = 0.9580909003000001
    global V3_init = 0.9999999997
    global V4_init = 3.0000000000299996
    global V5_init = 1.0

    global CA1_init = 9411.14074414605
    global CA2_init = 7293.4177491582395
    global CA3_init = 6108.208832008848
    global CA4_init = 1756.3130843620413
    global CA5_init = 5815.600453006247

    global CB1_init = 21.06302743621348
    global CB2_init = 25.023282756136876
    global CB3_init = 25.606263745834575
    global CB4_init = 14.042639798795763
    global CB5_init = 4.612844282717477

    global CC1_init = 1158.434641120098
    global CC2_init = 1870.3466414267186
    global CC3_init = 2606.1047140745954
    global CC4_init = 5403.116213249219
    global CC5_init = 3713.7187822654573

    global CD1_init = 226.97543015383067
    global CD2_init = 367.5264055495822
    global CD3_init = 504.16328315678663
    global CD4_init = 736.1262937969545
    global CD5_init = 204.33547426407827

    global V1_sp = 1.0
    global V2_sp = 1.0
    global V3_sp = 1.0
    global V4_sp = 3.0
    global V5_sp = 1.0

    global CA1_sp = 9101.956173414395
    global CA2_sp = 7548.288392938189
    global CA3_sp = 6163.4490863787205
    global CA4_sp = 1759.2784341257218
    global CA5_sp = 5815.81935832265

    global CB1_sp = 22.402512405273757
    global CB2_sp = 23.777314193012504
    global CB3_sp = 25.23919769017405
    global CB4_sp = 14.039535781282597
    global CB5_sp = 4.593462261757011

    global CC1_sp = 1116.1125107538646
    global CC2_sp = 1905.2818965279127
    global CC3_sp = 2613.7140416021593
    global CC4_sp = 5403.637374689891
    global CC5_sp = 3713.623258304895

    global CD1_sp = 220.27243861712677
    global CD2_sp = 373.0067010269757
    global CD3_sp = 505.3876834579604
    global CD4_sp = 736.2082216620913
    global CD5_sp = 204.29252612541296


    global F1_sp = 7.1E-3 # Parameter
    global F2_sp = 8.697E-4 # Parameter
    global Fr1_sp = 6E-3 # Parameter
    global Fr2_sp = 6E-3 # = Fr1_sp
    global F3_sp = 0.0139697 # = F1_sp + F2_sp + Fr2_sp
    global F4_sp = 8.697E-4 # = F2_sp
    global F6_sp = 8.697E-4 # = F2_sp
    global F5_sp = 0.01483939999 # = F4_sp + F3_sp
    global F7_sp = 0.0157091 # = F5_sp + F6_sp
    global F10_sp = 2.3E-3 # Parameter, disturbance from 2.3E-3 to 4.6E-3
    global F9_sp = 0.0083 # = Fr1_sp + F10_sp
    global F8_sp = 0.012009099999 # = F7_sp + F9_sp - (Fr1_sp + Fr2_sp)

    global Q1_sp = -4.4E6 # Parameter
    global Q2_sp = -4.6E6  # Parameter
    global Q3_sp = -4.7E6  # Parameter
    global Q4_sp = 9.2E6  # Parameter
    global Q5_sp = 5.6E6  # Parameter

    global H_vap_A = 3.073e4
    global H_vap_B = 1.35e4
    global H_vap_C = 4.226e4
    global H_vap_D = 4.55e4

    global H_ref_A = 7.44e4
    global H_ref_B = 5.91e4
    global H_ref_C = 2.01e4
    global H_ref_D = -2.89e4

    global Cp_A = 184.6
    global Cp_B = 59.1
    global Cp_C = 247
    global Cp_D = 301.3

    global CA0 = 1.126e4
    global CB0 = 2.028e4
    global CC0 = 8174
    global CD0 = 6485

    global Tref = 450
    global TA0 = 473 # DISTURBANCE, INCREASES by 30 K
    global TB0 = 473 # DISTURBANCE, INCREASES by 30 K
    global TD0 = 473 # DISTURBANCE, INCREASES by 30 K


    global delH_r1 = -1.53e5
    global delH_r2 = -1.118e5
    global delH_r3 = 4.141e5

    global R = 8.314
end

init_sp_and_params()

function gW(sp)
    wt = -Float64(floor(log10(abs(sp^2))))
    return 1 * 10^(wt)
end


w = Weights(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
# w = Weights(1 * gW(V1_sp), 10 * gW(T1_sp), 10 * gW(T2_sp), 10 * gW(T3_sp), 10 * gW(T4_sp), 10 * gW(T5_sp), gW(CA1_sp), gW(CA2_sp), gW(CA3_sp), gW(CA4_sp), gW(CA5_sp), gW(CB1_sp), gW(CB2_sp), gW(CB3_sp), gW(CB4_sp), gW(CB5_sp), 10 * gW(CC1_sp), 10 * gW(CC2_sp), 10 * gW(CC3_sp), 10 * gW(CC4_sp), 10 * gW(CC5_sp), gW(CD1_sp), gW(CD2_sp), gW(CD3_sp), gW(CD4_sp), gW(CD5_sp), 1E-1 * gW(F1_sp), 1E-1 * gW(F2_sp), 1E-1 * gW(F3_sp), 1E-1 * gW(F4_sp), 1E-1 * gW(F5_sp), 1E-1 * gW(F6_sp), 1E-1 * gW(F7_sp), 1E-1 * gW(F8_sp), 1E-1 * gW(F9_sp), 1E-1 * gW(F10_sp), 1E-1 * gW(Fr1_sp), 1E-1 * gW(Fr2_sp), 1E-1 * gW(Q1_sp), 1E-1 * gW(Q2_sp), 1E-1 * gW(Q3_sp), 1E-1 * gW(Q4_sp), 1E-1 * gW(Q5_sp))

# raw_weights = [
#     100*gW(V1_sp), 10*gW(T1_sp), 10*gW(T2_sp), 10*gW(T3_sp), 10*gW(T4_sp), 10*gW(T5_sp),
#     gW(CA1_sp), gW(CA2_sp), gW(CA3_sp), gW(CA4_sp), gW(CA5_sp),
#     gW(CB1_sp), gW(CB2_sp), gW(CB3_sp), gW(CB4_sp), gW(CB5_sp),
#     10*gW(CC1_sp), 10*gW(CC2_sp), 10*gW(CC3_sp), 10*gW(CC4_sp), 10*gW(CC5_sp),
#     gW(CD1_sp), gW(CD2_sp), gW(CD3_sp), gW(CD4_sp), gW(CD5_sp),
#     1E-1*gW(F1_sp), 1E-1*gW(F2_sp), 1E-1*gW(F3_sp), 1E-1*gW(F4_sp), 1E-1*gW(F5_sp),
#     1E-1*gW(F6_sp), 1E-1*gW(F7_sp), 1E-1*gW(F8_sp), 1E-1*gW(F9_sp), 1E-1*gW(F10_sp),
#     1E-1*gW(Fr1_sp), 1E-1*gW(Fr2_sp), gW(Q1_sp),gW(Q2_sp),
#     gW(Q3_sp), gW(Q4_sp), gW(Q5_sp)
# ]

# raw_weights = [
#     100*1, 10*1, 10*1, 10*1, 10*1, 10*1,
#     1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1,
#     10*1, 10*1, 10*1, 10*1, 10*1,
#     1, 1, 1, 1, 1,
#     1E-1*1, 1E-1*1, 1E-1*1, 1E-1*1, 1E-1*1,
#     1E-1*1, 1E-1*1, 1E-1*1, 1E-1*1, 1E-1*1,
#     1E-1*1, 1E-1*1, 
#     1, 1, 1, 1, 1
# ]

# max_weight = maximum(raw_weights)

# Normalize each weight
# normalized_weights = [w / max_weight for w in raw_weights]

# Construct the normalized weight structure
# w = Weights(normalized_weights...)

# Linear dependence of enthalpies
H_A(T) = H_ref_A + Cp_A * (T - Tref)
H_B(T) = H_ref_B + Cp_B * (T - Tref)
H_C(T) = H_ref_C + Cp_C * (T - Tref)
H_D(T) = H_ref_D + Cp_D * (T - Tref)

kEB2(T) = 0.152 * exp(-3933 / (R * T))
kEB3(T) = 0.490 * exp(-50870 / (R * T))

# Volatilities of the species in the seperator
alpha_A(T) = 0.0449 * T + 10
alpha_B(T) = 0.0260 * T + 10
alpha_C(T) = 0.0065 * T + 0.5
alpha_D(T) = 0.0058 * T + 0.25

# Debugging parameters for dual feasibility:
ub_f = 2.0
lb_f = 0.1

# Reaction rate expressions
r1(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * CA^(0.32) * CB^(1.5)
r2(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * CB^(2.5) * CC^(0.5)) / (1 + kEB2(T) * CD)
r3(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * CA^(1.0218) * CD) / (1 + kEB3(T) * CA)

# Reaction rate expressions
r1_(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * max(1e-6, CA)^(0.32) * max(1e-6, CB)^(1.5)
r2_(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * max(1e-6, CB)^(2.5) * max(1e-6, CC)^(0.5)) / (1 + kEB2(T) * CD)
r3_(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * max(1e-6, CA)^(1.0218) * CD) / (1 + kEB3(T) * CA)

# Molar flow in the overhead stream
MA(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_A(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
MB(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_B(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
MC(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_C(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
MD(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_D(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))

# Concentration of species in the recycle stream
CAr(MA, MB, MC, MD) = MA / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
CBr(MA, MB, MC, MD) = MB / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
CCr(MA, MB, MC, MD) = MC / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
CDr(MA, MB, MC, MD) = MD / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))

fix = Decomposition_Trajectory(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))
ig = Decomposition_Trajectory(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))

# Initialize fix structure for decomp
for set_traj = 1

    # Set control variables to nominal steady state values
    fix.F1 .= F1_sp
    fix.F2 .= F2_sp
    fix.F3 .= F3_sp
    fix.F4 .= F4_sp
    fix.F5 .= F5_sp
    fix.F6 .= F6_sp
    fix.F7 .= F7_sp
    fix.F8 .= F8_sp
    fix.F9 .= F9_sp
    fix.F10 .= F10_sp
    fix.Fr1 .= Fr1_sp
    fix.Fr2 .= Fr2_sp

    fix.Q1 .= Q1_sp
    fix.Q2 .= Q2_sp
    fix.Q3 .= Q3_sp
    fix.Q4 .= Q4_sp
    fix.Q5 .= Q5_sp

    fix.V1[1] = V1_init
    fix.V2[1] = V2_init
    fix.V3[1] = V3_init
    fix.V4[1] = V4_init
    fix.V5[1] = V5_init
    fix.T1[1] = T1_init
    fix.T2[1] = T2_init
    fix.T3[1] = T3_init
    fix.T4[1] = T4_init
    fix.T5[1] = T5_init

    fix.CA1[1] = CA1_init
    fix.CA2[1] = CA2_init
    fix.CA3[1] = CA3_init
    fix.CA4[1] = CA4_init
    fix.CA5[1] = CA5_init
    fix.CB1[1] = CB1_init
    fix.CB2[1] = CB2_init
    fix.CB3[1] = CB3_init
    fix.CB4[1] = CB4_init
    fix.CB5[1] = CB5_init
    fix.CC1[1] = CC1_init
    fix.CC2[1] = CC2_init
    fix.CC3[1] = CC3_init
    fix.CC4[1] = CC4_init
    fix.CC5[1] = CC5_init
    fix.CD1[1] = CD1_init
    fix.CD2[1] = CD2_init
    fix.CD3[1] = CD3_init
    fix.CD4[1] = CD4_init
    fix.CD5[1] = CD5_init

    for i = 1:N
        global k = i

        F1 = fix.F1[k]
        F2 = fix.F2[k]
        F3 = fix.F3[k]
        F4 = fix.F4[k]
        F5 = fix.F5[k]
        F6 = fix.F6[k]
        F7 = fix.F7[k]
        F8 = fix.F8[k]
        F9 = fix.F9[k]
        F10 = fix.F10[k]
        Fr1 = fix.Fr1[k]
        Fr2 = fix.Fr2[k]
        Q1 = fix.Q1[k]
        Q2 = fix.Q2[k]
        Q3 = fix.Q3[k]
        Q4 = fix.Q4[k]
        Q5 = fix.Q5[k]

        V1 = fix.V1[k]
        V2 = fix.V2[k]
        V3 = fix.V3[k]
        V4 = fix.V4[k]
        V5 = fix.V5[k]

        T1 = fix.T1[k]
        T2 = fix.T2[k]
        T3 = fix.T3[k]
        T4 = fix.T4[k]
        T5 = fix.T5[k]

        CA1 = fix.CA1[k]
        CB1 = fix.CB1[k]
        CC1 = fix.CC1[k]
        CD1 = fix.CD1[k]

        CA2 = fix.CA2[k]
        CB2 = fix.CB2[k]
        CC2 = fix.CC2[k]
        CD2 = fix.CD2[k]

        CA3 = fix.CA3[k]
        CB3 = fix.CB3[k]
        CC3 = fix.CC3[k]
        CD3 = fix.CD3[k]

        CA4 = fix.CA4[k]
        CB4 = fix.CB4[k]
        CC4 = fix.CC4[k]
        CD4 = fix.CD4[k]

        CA5 = fix.CA5[k]
        CB5 = fix.CB5[k]
        CC5 = fix.CC5[k]
        CD5 = fix.CD5[k]

        x0 = [fix.V1[k], fix.V2[k], fix.V3[k], fix.V4[k], fix.V5[k], fix.T1[k], fix.T2[k], fix.T3[k], fix.T4[k], fix.T5[k], fix.CA1[k], fix.CB1[k], fix.CC1[k], fix.CD1[k], fix.CA2[k], fix.CB2[k], fix.CC2[k], fix.CD2[k], fix.CA3[k], fix.CB3[k], fix.CC3[k], fix.CD3[k], fix.CA4[k], fix.CB4[k], fix.CC4[k], fix.CD4[k], fix.CA5[k], fix.CB5[k], fix.CC5[k], fix.CD5[k]]

        # Define the ODE function
        function f(y, p, t)
            # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
            V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

            dV1 = F1 + F2 + Fr2 - F3
            dV2 = F3 + F4 - F5
            dV3 = F5 + F6 - F7
            dV4 = F7 + F9 - F8 - Fr1 - Fr2
            dV5 = F10 + Fr1 - F9

            dT1 = (
                ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
                  (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
                  (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
                  (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
                  (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
                 /
                 (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
                (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
                (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

            dT2 = (
                (Q2 + F4 * CB0 * H_B(TB0) +
                 (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
                 (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
                 (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
                 (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
                /
                (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
                +
                (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
                /
                (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

            dT3 = (
                ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
                  (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
                  (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
                  (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
                 /
                 (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
                +
                (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
                /
                (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

            dT4 = ((Q4
                    + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
                    + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
                    + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
                    + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
                   /
                   (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
            dT5 = (
                ((Q5 +
                  F10 * CD0 * H_D(TD0)
                  + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
                  + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
                  + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
                  + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
                 /
                 (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
                +
                ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
                 /
                 (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


            dCA1 = (
                ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CA1)
                 /
                 V1) - r1(T1, CA1, CB1))

            dCB1 = (
                ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CB1) /
                 V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

            dCC1 = (
                ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CC1) /
                 V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

            dCD1 = (
                ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CD1) /
                 V1) + r2(T1, CB1, CC1, CD1))

            dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
            dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
            dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
            dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

            dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
            dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
            dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
            dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

            dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
            dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
            dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
            dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

            dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
            dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
            dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
            dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

            return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
        end

        # Solve ODE 
        tspan = (0.0, dt)
        prob = ODEProblem(f, x0, tspan)
        soln = solve(prob, Rosenbrock23())

        # Obtain the next values for each element
        fix.V1[k+1] = soln.u[end][1]
        fix.V2[k+1] = soln.u[end][2]
        fix.V3[k+1] = soln.u[end][3]
        fix.V4[k+1] = soln.u[end][4]
        fix.V5[k+1] = soln.u[end][5]

        fix.T1[k+1] = soln.u[end][6]
        fix.T2[k+1] = soln.u[end][7]
        fix.T3[k+1] = soln.u[end][8]
        fix.T4[k+1] = soln.u[end][9]
        fix.T5[k+1] = soln.u[end][10]

        fix.CA1[k+1] = soln.u[end][11]
        fix.CB1[k+1] = soln.u[end][12]
        fix.CC1[k+1] = soln.u[end][13]
        fix.CD1[k+1] = soln.u[end][14]

        fix.CA2[k+1] = soln.u[end][15]
        fix.CB2[k+1] = soln.u[end][16]
        fix.CC2[k+1] = soln.u[end][17]
        fix.CD2[k+1] = soln.u[end][18]

        fix.CA3[k+1] = soln.u[end][19]
        fix.CB3[k+1] = soln.u[end][20]
        fix.CC3[k+1] = soln.u[end][21]
        fix.CD3[k+1] = soln.u[end][22]

        fix.CA4[k+1] = soln.u[end][23]
        fix.CB4[k+1] = soln.u[end][24]
        fix.CC4[k+1] = soln.u[end][25]
        fix.CD4[k+1] = soln.u[end][26]

        fix.CA5[k+1] = soln.u[end][27]
        fix.CB5[k+1] = soln.u[end][28]
        fix.CC5[k+1] = soln.u[end][29]
        fix.CD5[k+1] = soln.u[end][30]
    end

end

# Initialize ig structure for centralized
for set_traj = 1

    # Set control variables to nominal steady state values
    ig.F1 .= F1_sp
    ig.F2 .= F2_sp
    ig.F3 .= F3_sp
    ig.F4 .= F4_sp
    ig.F5 .= F5_sp
    ig.F6 .= F6_sp
    ig.F7 .= F7_sp
    ig.F8 .= F8_sp
    ig.F9 .= F9_sp
    ig.F10 .= F10_sp
    ig.Fr1 .= Fr1_sp
    ig.Fr2 .= Fr2_sp

    ig.Q1 .= Q1_sp
    ig.Q2 .= Q2_sp
    ig.Q3 .= Q3_sp
    ig.Q4 .= Q4_sp
    ig.Q5 .= Q5_sp

    ig.V1[1] = V1_init
    ig.V2[1] = V2_init
    ig.V3[1] = V3_init
    ig.V4[1] = V4_init
    ig.V5[1] = V5_init
    ig.T1[1] = T1_init
    ig.T2[1] = T2_init
    ig.T3[1] = T3_init
    ig.T4[1] = T4_init
    ig.T5[1] = T5_init

    ig.CA1[1] = CA1_init
    ig.CA2[1] = CA2_init
    ig.CA3[1] = CA3_init
    ig.CA4[1] = CA4_init
    ig.CA5[1] = CA5_init
    ig.CB1[1] = CB1_init
    ig.CB2[1] = CB2_init
    ig.CB3[1] = CB3_init
    ig.CB4[1] = CB4_init
    ig.CB5[1] = CB5_init
    ig.CC1[1] = CC1_init
    ig.CC2[1] = CC2_init
    ig.CC3[1] = CC3_init
    ig.CC4[1] = CC4_init
    ig.CC5[1] = CC5_init
    ig.CD1[1] = CD1_init
    ig.CD2[1] = CD2_init
    ig.CD3[1] = CD3_init
    ig.CD4[1] = CD4_init
    ig.CD5[1] = CD5_init

    for i = 1:N
        global k = i

        F1 = ig.F1[k]
        F2 = ig.F2[k]
        F3 = ig.F3[k]
        F4 = ig.F4[k]
        F5 = ig.F5[k]
        F6 = ig.F6[k]
        F7 = ig.F7[k]
        F8 = ig.F8[k]
        F9 = ig.F9[k]
        F10 = ig.F10[k]
        Fr1 = ig.Fr1[k]
        Fr2 = ig.Fr2[k]
        Q1 = ig.Q1[k]
        Q2 = ig.Q2[k]
        Q3 = ig.Q3[k]
        Q4 = ig.Q4[k]
        Q5 = ig.Q5[k]

        V1 = ig.V1[k]
        V2 = ig.V2[k]
        V3 = ig.V3[k]
        V4 = ig.V4[k]
        V5 = ig.V5[k]

        T1 = ig.T1[k]
        T2 = ig.T2[k]
        T3 = ig.T3[k]
        T4 = ig.T4[k]
        T5 = ig.T5[k]

        CA1 = ig.CA1[k]
        CB1 = ig.CB1[k]
        CC1 = ig.CC1[k]
        CD1 = ig.CD1[k]

        CA2 = ig.CA2[k]
        CB2 = ig.CB2[k]
        CC2 = ig.CC2[k]
        CD2 = ig.CD2[k]

        CA3 = ig.CA3[k]
        CB3 = ig.CB3[k]
        CC3 = ig.CC3[k]
        CD3 = ig.CD3[k]

        CA4 = ig.CA4[k]
        CB4 = ig.CB4[k]
        CC4 = ig.CC4[k]
        CD4 = ig.CD4[k]

        CA5 = ig.CA5[k]
        CB5 = ig.CB5[k]
        CC5 = ig.CC5[k]
        CD5 = ig.CD5[k]

        x0 = [ig.V1[k], ig.V2[k], ig.V3[k], ig.V4[k], ig.V5[k], ig.T1[k], ig.T2[k], ig.T3[k], ig.T4[k], ig.T5[k], ig.CA1[k], ig.CB1[k], ig.CC1[k], ig.CD1[k], ig.CA2[k], ig.CB2[k], ig.CC2[k], ig.CD2[k], ig.CA3[k], ig.CB3[k], ig.CC3[k], ig.CD3[k], ig.CA4[k], ig.CB4[k], ig.CC4[k], ig.CD4[k], ig.CA5[k], ig.CB5[k], ig.CC5[k], ig.CD5[k]]

        # Define the ODE function
        function f(y, p, t)
            # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
            V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

            dV1 = F1 + F2 + Fr2 - F3
            dV2 = F3 + F4 - F5
            dV3 = F5 + F6 - F7
            dV4 = F7 + F9 - F8 - Fr1 - Fr2
            dV5 = F10 + Fr1 - F9

            dT1 = (
                ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
                  (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
                  (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
                  (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
                  (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
                 /
                 (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
                (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
                (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

            dT2 = (
                (Q2 + F4 * CB0 * H_B(TB0) +
                 (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
                 (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
                 (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
                 (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
                /
                (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
                +
                (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
                /
                (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

            dT3 = (
                ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
                  (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
                  (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
                  (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
                 /
                 (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
                +
                (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
                /
                (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

            dT4 = ((Q4
                    + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
                    + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
                    + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
                    + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
                   /
                   (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
            dT5 = (
                ((Q5 +
                  F10 * CD0 * H_D(TD0)
                  + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
                  + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
                  + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
                  + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
                 /
                 (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
                +
                ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
                 /
                 (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


            dCA1 = (
                ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CA1)
                 /
                 V1) - r1(T1, CA1, CB1))

            dCB1 = (
                ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CB1) /
                 V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

            dCC1 = (
                ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CC1) /
                 V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

            dCD1 = (
                ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3 * CD1) /
                 V1) + r2(T1, CB1, CC1, CD1))

            dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
            dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
            dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
            dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

            dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
            dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
            dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
            dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

            dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
            dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
            dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
            dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

            dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
            dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
            dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
            dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

            return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
        end

        # Solve ODE 
        tspan = (0.0, dt)
        prob = ODEProblem(f, x0, tspan)
        soln = solve(prob, Rosenbrock23())

        # Obtain the next values for each element
        ig.V1[k+1] = soln.u[end][1]
        ig.V2[k+1] = soln.u[end][2]
        ig.V3[k+1] = soln.u[end][3]
        ig.V4[k+1] = soln.u[end][4]
        ig.V5[k+1] = soln.u[end][5]

        ig.T1[k+1] = soln.u[end][6]
        ig.T2[k+1] = soln.u[end][7]
        ig.T3[k+1] = soln.u[end][8]
        ig.T4[k+1] = soln.u[end][9]
        ig.T5[k+1] = soln.u[end][10]

        ig.CA1[k+1] = soln.u[end][11]
        ig.CB1[k+1] = soln.u[end][12]
        ig.CC1[k+1] = soln.u[end][13]
        ig.CD1[k+1] = soln.u[end][14]

        ig.CA2[k+1] = soln.u[end][15]
        ig.CB2[k+1] = soln.u[end][16]
        ig.CC2[k+1] = soln.u[end][17]
        ig.CD2[k+1] = soln.u[end][18]

        ig.CA3[k+1] = soln.u[end][19]
        ig.CB3[k+1] = soln.u[end][20]
        ig.CC3[k+1] = soln.u[end][21]
        ig.CD3[k+1] = soln.u[end][22]

        ig.CA4[k+1] = soln.u[end][23]
        ig.CB4[k+1] = soln.u[end][24]
        ig.CC4[k+1] = soln.u[end][25]
        ig.CD4[k+1] = soln.u[end][26]

        ig.CA5[k+1] = soln.u[end][27]
        ig.CB5[k+1] = soln.u[end][28]
        ig.CC5[k+1] = soln.u[end][29]
        ig.CD5[k+1] = soln.u[end][30]
    end

end

function cmpc()

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # State variables

        # Volume, state, [=] m3
        V1[k=0:N], (lower_bound=min(V1_init, V1_sp) * 0.2, upper_bound=max(V1_init, V1_sp) * 1.8)
        V2[k=0:N], (lower_bound=min(V2_init, V2_sp) * 0.2, upper_bound=max(V2_init, V2_sp) * 1.8)

        # Temperature, state, [=] K
        T1[k=0:N], (lower_bound=min(T1_init, T1_sp) * 0.2, upper_bound=max(T1_init, T1_sp) * 1.8)
        T2[k=0:N], (lower_bound=min(T2_init, T2_sp) * 0.2, upper_bound=max(T2_init, T2_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA1[k=0:N], (lower_bound=min(CA1_init, CA1_sp) * 0.2, upper_bound=max(CA1_init, CA1_sp) * 1.8)
        CA2[k=0:N], (lower_bound=min(CA2_init, CA2_sp) * 0.2, upper_bound=max(CA2_init, CA2_sp) * 1.8)

        CB1[k=0:N], (lower_bound=min(CB1_init, CB1_sp) * 0.2, upper_bound=max(CB1_init, CB1_sp) * 1.8)
        CB2[k=0:N], (lower_bound=min(CB2_init, CB2_sp) * 0.2, upper_bound=max(CB2_init, CB2_sp) * 1.8)

        CC1[k=0:N], (lower_bound=min(CC1_init, CC1_sp) * 0.2, upper_bound=max(CC1_init, CC1_sp) * 1.8)
        CC2[k=0:N], (lower_bound=min(CC2_init, CC2_sp) * 0.2, upper_bound=max(CC2_init, CC2_sp) * 1.8)

        CD1[k=0:N], (lower_bound=min(CD1_init, CD1_sp) * 0.2, upper_bound=max(CD1_init, CD1_sp) * 1.8)
        CD2[k=0:N], (lower_bound=min(CD2_init, CD2_sp) * 0.2, upper_bound=max(CD2_init, CD2_sp) * 1.8)

        # Volume, state, [=] m3    
        V3[k=0:N], (lower_bound=min(V3_init, V3_sp) * 0.2, upper_bound=max(V3_init, V3_sp) * 1.8)

        # Temperature, state, [=] K
        T3[k=0:N], (lower_bound=min(T3_init, T3_sp) * 0.2, upper_bound=max(T3_init, T3_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA3[k=0:N], (lower_bound=min(CA3_init, CA3_sp) * 0.2, upper_bound=max(CA3_init, CA3_sp) * 1.8)

        CB3[k=0:N], (lower_bound=min(CB3_init, CB3_sp) * 0.2, upper_bound=max(CB3_init, CB3_sp) * 1.8)

        CC3[k=0:N], (lower_bound=min(CC3_init, CC3_sp) * 0.2, upper_bound=max(CC3_init, CC3_sp) * 1.8)

        CD3[k=0:N], (lower_bound=min(CD3_init, CD3_sp) * 0.2, upper_bound=max(CD3_init, CD3_sp) * 1.8)

        V4[k=0:N], (lower_bound=min(V4_init, V4_sp) * 0.2, upper_bound=max(V4_init, V4_sp) * 1.8)
        V5[k=0:N], (lower_bound=min(V5_init, V5_sp) * 0.2, upper_bound=max(V5_init, V5_sp) * 1.8)

        # Temperature, state, [=] K
        T4[k=0:N], (lower_bound=min(T4_init, T4_sp) * 0.2, upper_bound=max(T4_init, T4_sp) * 1.8)
        T5[k=0:N], (lower_bound=min(T5_init, T5_sp) * 0.2, upper_bound=max(T5_init, T5_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA4[k=0:N], (lower_bound=min(CA4_init, CA4_sp) * 0.2, upper_bound=max(CA4_init, CA4_sp) * 1.8)
        CA5[k=0:N], (lower_bound=min(CA5_init, CA5_sp) * 0.2, upper_bound=max(CA5_init, CA5_sp) * 1.8)

        CB4[k=0:N], (lower_bound=min(CB4_init, CB4_sp) * 0.2, upper_bound=max(CB4_init, CB4_sp) * 1.8)
        CB5[k=0:N], (lower_bound=min(CB5_init, CB5_sp) * 0.2, upper_bound=max(CB5_init, CB5_sp) * 1.8)

        CC4[k=0:N], (lower_bound=min(CC4_init, CC4_sp) * 0.2, upper_bound=max(CC4_init, CC4_sp) * 1.8)
        CC5[k=0:N], (lower_bound=min(CC5_init, CC5_sp) * 0.2, upper_bound=max(CC5_init, CC5_sp) * 1.8)

        CD4[k=0:N], (lower_bound=min(CD4_init, CD4_sp) * 0.2, upper_bound=max(CD4_init, CD4_sp) * 1.8)
        CD5[k=0:N], (lower_bound=min(CD5_init, CD5_sp) * 0.2, upper_bound=max(CD5_init, CD5_sp) * 1.8)

        F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp)
        F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp)
        F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp)
        F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp)
        F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp)
        F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp)
        F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp)
        F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp)
        F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp)
        F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp)

        Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp)
        Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp)

        Q1[k=0:N], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
        Q2[k=0:N], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)
        Q3[k=0:N], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)
        Q4[k=0:N], (start=Q4_sp, upper_bound=1.8 * Q4_sp, lower_bound=1e-6)
        Q5[k=0:N], (start=Q5_sp, upper_bound=1.8 * Q5_sp, lower_bound=1e-6)

    end

    for k = 0:N
        set_start_value(V1[k], ig.V1[k+1])
        set_start_value(V2[k], ig.V2[k+1])
        set_start_value(V3[k], ig.V3[k+1])
        set_start_value(V4[k], ig.V4[k+1])
        set_start_value(V5[k], ig.V5[k+1])

        set_start_value(T1[k], ig.T1[k+1])
        set_start_value(T2[k], ig.T2[k+1])
        set_start_value(T3[k], ig.T3[k+1])
        set_start_value(T4[k], ig.T4[k+1])
        set_start_value(T5[k], ig.T5[k+1])

        set_start_value(CA1[k], ig.CA1[k+1])
        set_start_value(CA2[k], ig.CA2[k+1])
        set_start_value(CA3[k], ig.CA3[k+1])
        set_start_value(CA4[k], ig.CA4[k+1])
        set_start_value(CA5[k], ig.CA5[k+1])

        set_start_value(CB1[k], ig.CB1[k+1])
        set_start_value(CB2[k], ig.CB2[k+1])
        set_start_value(CB3[k], ig.CB3[k+1])
        set_start_value(CB4[k], ig.CB4[k+1])
        set_start_value(CB5[k], ig.CB5[k+1])

        set_start_value(CC1[k], ig.CC1[k+1])
        set_start_value(CC2[k], ig.CC2[k+1])
        set_start_value(CC3[k], ig.CC3[k+1])
        set_start_value(CC4[k], ig.CC4[k+1])
        set_start_value(CC5[k], ig.CC5[k+1])

        set_start_value(CD1[k], ig.CD1[k+1])
        set_start_value(CD2[k], ig.CD2[k+1])
        set_start_value(CD3[k], ig.CD3[k+1])
        set_start_value(CD4[k], ig.CD4[k+1])
        set_start_value(CD5[k], ig.CD5[k+1])

        set_start_value(F1[k], F1_sp)
        set_start_value(F2[k], F2_sp)
        set_start_value(F3[k], F3_sp)
        set_start_value(F4[k], F4_sp)
        set_start_value(F5[k], F5_sp)
        set_start_value(F6[k], F6_sp)
        set_start_value(F7[k], F7_sp)
        set_start_value(F8[k], F8_sp)
        set_start_value(F9[k], F9_sp)
        set_start_value(F10[k], F10_sp)
        set_start_value(Fr1[k], Fr1_sp)
        set_start_value(Fr2[k], Fr2_sp)

    end

    for k = 0:N
        JuMP.fix(F7[k], F7_sp; force=true)
        JuMP.fix(Fr2[k], Fr2_sp; force=true)
    end

    @constraints mpc begin
        # Initial condition
        V1_inital, V1[0] == V1_init
        V2_inital, V2[0] == V2_init
        V3_inital, V3[0] == V3_init
        V4_inital, V4[0] == V4_init
        V5_inital, V5[0] == V5_init

        T1_inital, T1[0] == T1_init
        T2_inital, T2[0] == T2_init
        T3_inital, T3[0] == T3_init
        T4_inital, T4[0] == T4_init
        T5_inital, T5[0] == T5_init

        CA1_initial, CA1[0] == CA1_init
        CA2_initial, CA2[0] == CA2_init
        CA3_initial, CA3[0] == CA3_init
        CA4_initial, CA4[0] == CA4_init
        CA5_initial, CA5[0] == CA5_init

        CB1_initial, CB1[0] == CB1_init
        CB2_initial, CB2[0] == CB2_init
        CB3_initial, CB3[0] == CB3_init
        CB4_initial, CB4[0] == CB4_init
        CB5_initial, CB5[0] == CB5_init

        CC1_initial, CC1[0] == CC1_init
        CC2_initial, CC2[0] == CC2_init
        CC3_initial, CC3[0] == CC3_init
        CC4_initial, CC4[0] == CC4_init
        CC5_initial, CC5[0] == CC5_init

        CD1_initial, CD1[0] == CD1_init
        CD2_initial, CD2[0] == CD2_init
        CD3_initial, CD3[0] == CD3_init
        CD4_initial, CD4[0] == CD4_init
        CD5_initial, CD5[0] == CD5_init

        F1_hold[k in k_indices], F1[k] == F1[k+1]
        F2_hold[k in k_indices], F2[k] == F2[k+1]
        F3_hold[k in k_indices], F3[k] == F3[k+1]
        F4_hold[k in k_indices], F4[k] == F4[k+1]
        F5_hold[k in k_indices], F5[k] == F5[k+1]
        F6_hold[k in k_indices], F6[k] == F6[k+1]
        F7_hold[k in k_indices], F7[k] == F7[k+1]
        F8_hold[k in k_indices], F8[k] == F8[k+1]
        F9_hold[k in k_indices], F9[k] == F9[k+1]
        F10_hold[k in k_indices], F10[k] == F10[k+1]
        Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
        Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

        Q1_hold[k in k_indices], Q1[k] == Q1[k+1]
        Q2_hold[k in k_indices], Q2[k] == Q2[k+1]
        Q3_hold[k in k_indices], Q3[k] == Q3[k+1]
        Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
        Q5_hold[k in k_indices], Q5[k] == Q5[k+1]


    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system
        dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + Fr2[k] - F3[k]) * dt == V1[k+1]

        dT1_dt[k=0:N-1], T1[k] + (
            ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
              (Fr2[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F3[k] * CA1[k] * H_A(T1[k])) +
              (Fr2[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F3[k] * CB1[k] * H_B(T1[k])) +
              (Fr2[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F3[k] * CC1[k] * H_C(T1[k])) +
              (Fr2[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F3[k] * CD1[k] * H_D(T1[k])))
             /
             (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
            (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
            (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

        dCA1_dt[k=0:N-1], CA1[k] + (
            ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              F3[k] * CA1[k])
             /
             V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

        dCB1_dt[k=0:N-1], CB1[k] + (
            ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              F3[k] * CB1[k]) /
             V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

        dCC1_dt[k=0:N-1], CC1[k] + (
            ((Fr2[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              F3[k] * CC1[k]) /
             V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

        dCD1_dt[k=0:N-1], CD1[k] + (
            ((Fr2[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              F3[k] * CD1[k]) /
             V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

        dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - F5[k]) * dt == V2[k+1]

        dT2_dt[k=0:N-1], T2[k] + (
            (Q2[k] + F4[k] * CB0 * H_B(TB0) +
             (F3[k] * CA1[k] * H_A(T1[k]) - F5[k] * CA2[k] * H_A(T2[k])) +
             (F3[k] * CB1[k] * H_B(T1[k]) - F5[k] * CB2[k] * H_B(T2[k])) +
             (F3[k] * CC1[k] * H_C(T1[k]) - F5[k] * CC2[k] * H_C(T2[k])) +
             (F3[k] * CD1[k] * H_D(T1[k]) - F5[k] * CD2[k] * H_D(T2[k])))
            /
            (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
            +
            (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
            /
            (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


        dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * CA1[k] - F5[k] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
        dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - F5[k] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
        dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * CC1[k] - F5[k] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
        dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * CD1[k] - F5[k] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]

        dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - F7[k]) * dt == V3[k+1]


        dT3_dt[k=0:N-1], T3[k] + (
            ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2[k] * H_A(T2[k]) - F7[k] * CA3[k] * H_A(T3[k])) +
              (F5[k] * CB2[k] * H_B(T2[k]) - F7[k] * CB3[k] * H_B(T3[k])) +
              (F5[k] * CC2[k] * H_C(T2[k]) - F7[k] * CC3[k] * H_C(T3[k])) +
              (F5[k] * CD2[k] * H_D(T2[k]) - F7[k] * CD3[k] * H_D(T3[k])))
             /
             (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
            +
            (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
            /
            (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

        dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * CA2[k] - F7[k] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
        dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * CB2[k] + F6[k] * CB0 - F7[k] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
        dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * CC2[k] - F7[k] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
        dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * CD2[k] - F7[k] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

        dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4[k+1]

        dT4_dt[k=0:N-1], T4[k] +
                         ((Q4[k]
                           + (F7[k] * CA3[k] * H_A(T3[k]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
                           + (F7[k] * CB3[k] * H_B(T3[k]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
                           + (F7[k] * CC3[k] * H_C(T3[k]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
                           + (F7[k] * CD3[k] * H_D(T3[k]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
                          /
                          (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

        dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * CA3[k] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
        dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * CB3[k] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
        dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * CC3[k] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
        dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * CD3[k] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

        dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5[k+1]

        dT5_dt[k=0:N-1], T5[k] + (
            ((Q5[k] +
              F10[k] * CD0 * H_D(TD0)
              + (Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
              + (Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
              + (Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
              + (Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
             /
             (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
            +
            ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
             /
             (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

        dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
        dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
        dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
        dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]

        # volHoldUp1[k in 0:dtd:(N-dtd)], F1[k] + F2[k] + Fr2[k] - F3[k] == -(V1[k] - V1_sp) / tau
        # volHoldUp2[k in 0:dtd:(N-dtd)], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / tau
        # volHoldUp3[k in 0:dtd:(N-dtd)], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / tau
        # volHoldUp4[k in 0:dtd:(N-dtd)], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        # volHoldUp5[k in 0:dtd:(N-dtd)], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau

    end


    @NLobjective(mpc, Min, sum(
        w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 + w.v * (V3[k] - V3_sp)^2 + w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
        w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
        w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
        w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 + w.cb3 * (CB3[k] - CB3_sp)^2 + w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
        w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 + w.cc3 * (CC3[k] - CC3_sp)^2 + w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
        w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2 + w.cd3 * (CD3[k] - CD3_sp)^2 + w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
        for k = 0:N)
                           +
                           sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 + w.f5 * (F5[k] - F5_sp)^2 +
                               w.f6 * (F6[k] - F6_sp)^2 + w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 + w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 + w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
                               w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2 for k = 0:N)
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", c_opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", c_dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", c_constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", c_compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", c_cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global cmpc_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

    F1 = Vector(JuMP.value.(F1))
    F2 = Vector(JuMP.value.(F2))
    F3 = Vector(JuMP.value.(F3))
    F4 = Vector(JuMP.value.(F4))
    Q1 = Vector(JuMP.value.(Q1))
    Q2 = Vector(JuMP.value.(Q2))

    F5 = Vector(JuMP.value.(F5))
    F6 = Vector(JuMP.value.(F6))
    Q3 = Vector(JuMP.value.(Q3))

    F7 = Vector(JuMP.value.(F7))
    F8 = Vector(JuMP.value.(F8))
    F9 = Vector(JuMP.value.(F9))
    F10 = Vector(JuMP.value.(F10))
    Fr1 = Vector(JuMP.value.(Fr1))
    Fr2 = Vector(JuMP.value.(Fr2))
    Q4 = Vector(JuMP.value.(Q4))
    Q5 = Vector(JuMP.value.(Q5))

    return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5

end

function dmpc_1_decomp1()

    # Control variables: F1, F2, F3, F4, Q1, Q2
    # State variables: All state variables associated with CSTR-1 and CSTR-2

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin

        # State variables

        # Volume, state, [=] m3
        V1[k=0:N], (lower_bound=min(V1_init, V1_sp) * 0.2, upper_bound=max(V1_init, V1_sp) * 1.8)
        V2[k=0:N], (lower_bound=min(V2_init, V2_sp) * 0.2, upper_bound=max(V2_init, V2_sp) * 1.8)

        # Temperature, state, [=] K
        T1[k=0:N], (lower_bound=min(T1_init, T1_sp) * 0.2, upper_bound=max(T1_init, T1_sp) * 1.8)
        T2[k=0:N], (lower_bound=min(T2_init, T2_sp) * 0.2, upper_bound=max(T2_init, T2_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA1[k=0:N], (lower_bound=min(CA1_init, CA1_sp) * 0.2, upper_bound=max(CA1_init, CA1_sp) * 1.8)
        CA2[k=0:N], (lower_bound=min(CA2_init, CA2_sp) * 0.2, upper_bound=max(CA2_init, CA2_sp) * 1.8)

        CB1[k=0:N], (lower_bound=min(CB1_init, CB1_sp) * 0.2, upper_bound=max(CB1_init, CB1_sp) * 1.8)
        CB2[k=0:N], (lower_bound=min(CB2_init, CB2_sp) * 0.2, upper_bound=max(CB2_init, CB2_sp) * 1.8)

        CC1[k=0:N], (lower_bound=min(CC1_init, CC1_sp) * 0.2, upper_bound=max(CC1_init, CC1_sp) * 1.8)
        CC2[k=0:N], (lower_bound=min(CC2_init, CC2_sp) * 0.2, upper_bound=max(CC2_init, CC2_sp) * 1.8)

        CD1[k=0:N], (lower_bound=min(CD1_init, CD1_sp) * 0.2, upper_bound=max(CD1_init, CD1_sp) * 1.8)
        CD2[k=0:N], (lower_bound=min(CD2_init, CD2_sp) * 0.2, upper_bound=max(CD2_init, CD2_sp) * 1.8)

        F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
        F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)
        F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
        F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)

        Q1[k=0:N], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
        Q2[k=0:N], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)

    end

    for k = 0:N
        set_start_value(V1[k], fix.V1[k+1])
        set_start_value(V2[k], fix.V2[k+1])

        set_start_value(T1[k], fix.T1[k+1])
        set_start_value(T2[k], fix.T2[k+1])

        set_start_value(CA1[k], fix.CA1[k+1])
        set_start_value(CA2[k], fix.CA2[k+1])

        set_start_value(CB1[k], fix.CB1[k+1])
        set_start_value(CB2[k], fix.CB2[k+1])

        set_start_value(CC1[k], fix.CC1[k+1])
        set_start_value(CC2[k], fix.CC2[k+1])

        set_start_value(CD1[k], fix.CD1[k+1])
        set_start_value(CD2[k], fix.CD2[k+1])

        set_start_value(F1[k], fix.F1[k+1])
        set_start_value(F2[k], fix.F2[k+1])
        set_start_value(F3[k], fix.F3[k+1])
        set_start_value(F4[k], fix.F4[k+1])

        set_start_value(Q1[k], fix.Q1[k+1])
        set_start_value(Q2[k], fix.Q2[k+1])
    end

    @constraints mpc begin

        # Initial condition
        V1_inital, V1[0] == V1_init
        V2_inital, V2[0] == V2_init

        T1_inital, T1[0] == T1_init
        T2_inital, T2[0] == T2_init

        CA1_initial, CA1[0] == CA1_init
        CA2_initial, CA2[0] == CA2_init

        CB1_initial, CB1[0] == CB1_init
        CB2_initial, CB2[0] == CB2_init

        CC1_initial, CC1[0] == CC1_init
        CC2_initial, CC2[0] == CC2_init

        CD1_initial, CD1[0] == CD1_init
        CD2_initial, CD2[0] == CD2_init

        F1_hold[k in k_indices], F1[k] == F1[k+1]
        F2_hold[k in k_indices], F2[k] == F2[k+1]
        F3_hold[k in k_indices], F3[k] == F3[k+1]
        F4_hold[k in k_indices], F4[k] == F4[k+1]

        Q1_hold[k in k_indices], Q1[k] == Q1[k+1]
        Q2_hold[k in k_indices], Q2[k] == Q2[k+1]


    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system
        dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + fix.Fr2[k+1] - F3[k]) * dt == V1[k+1]

        dT1_dt[k=0:N-1], T1[k] + (
            ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
              (fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_A(fix.T4[k+1]) - F3[k] * CA1[k] * H_A(T1[k])) +
              (fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_B(fix.T4[k+1]) - F3[k] * CB1[k] * H_B(T1[k])) +
              (fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_C(fix.T4[k+1]) - F3[k] * CC1[k] * H_C(T1[k])) +
              (fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_D(fix.T4[k+1]) - F3[k] * CD1[k] * H_D(T1[k])))
             /
             (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
            (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
            (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

        dCA1_dt[k=0:N-1], CA1[k] + (
            ((F1[k] * CA0 + fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CA1[k])
             /
             V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

        dCB1_dt[k=0:N-1], CB1[k] + (
            ((F2[k] * CB0 + fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CB1[k]) /
             V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

        dCC1_dt[k=0:N-1], CC1[k] + (
            ((fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CC1[k]) /
             V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

        dCD1_dt[k=0:N-1], CD1[k] + (
            ((fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CD1[k]) /
             V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

        dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - fix.F5[k+1]) * dt == V2[k+1]

        dT2_dt[k=0:N-1], T2[k] + (
            (Q2[k] + F4[k] * CB0 * H_B(TB0) +
             (F3[k] * CA1[k] * H_A(T1[k]) - fix.F5[k+1] * CA2[k] * H_A(T2[k])) +
             (F3[k] * CB1[k] * H_B(T1[k]) - fix.F5[k+1] * CB2[k] * H_B(T2[k])) +
             (F3[k] * CC1[k] * H_C(T1[k]) - fix.F5[k+1] * CC2[k] * H_C(T2[k])) +
             (F3[k] * CD1[k] * H_D(T1[k]) - fix.F5[k+1] * CD2[k] * H_D(T2[k])))
            /
            (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
            +
            (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
            /
            (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


        dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * CA1[k] - fix.F5[k+1] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
        dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - fix.F5[k+1] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
        dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * CC1[k] - fix.F5[k+1] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
        dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * CD1[k] - fix.F5[k+1] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]


        volHoldUp1[k in 0:dtd:(N-dtd)], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / tau
        volHoldUp2[k in 0:dtd:(N-dtd)], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / tau

    end



    @NLobjective(mpc, Min, sum(
        w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 +
        w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 +
        w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 +
        w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 +
        w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 +
        w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2
        for k = 0:N) + sum(
        w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
        w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)
    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc1_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

    F1 = Vector(JuMP.value.(F1))
    F2 = Vector(JuMP.value.(F2))
    F3 = Vector(JuMP.value.(F3))
    F4 = Vector(JuMP.value.(F4))
    Q1 = Vector(JuMP.value.(Q1))
    Q2 = Vector(JuMP.value.(Q2))

    V1 = Vector(JuMP.value.(V1))
    V2 = Vector(JuMP.value.(V2))

    T1 = Vector(JuMP.value.(T1))
    T2 = Vector(JuMP.value.(T2))

    CA1 = Vector(JuMP.value.(CA1))
    CA2 = Vector(JuMP.value.(CA2))

    CB1 = Vector(JuMP.value.(CB1))
    CB2 = Vector(JuMP.value.(CB2))

    CC1 = Vector(JuMP.value.(CC1))
    CC2 = Vector(JuMP.value.(CC2))

    CD1 = Vector(JuMP.value.(CD1))
    CD2 = Vector(JuMP.value.(CD2))

    return F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2
end

function dmpc_2_decomp1()

    # Control variables: F5, F6, Q3
    # State variables: All state variables associated with CSTR-3

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # State variables

        # Volume, state, [=] m3    
        V3[k=0:N], (lower_bound=min(V3_init, V3_sp) * 0.2, upper_bound=max(V3_init, V3_sp) * 1.8)

        # Temperature, state, [=] K
        T3[k=0:N], (lower_bound=min(T3_init, T3_sp) * 0.2, upper_bound=max(T3_init, T3_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA3[k=0:N], (lower_bound=min(CA3_init, CA3_sp) * 0.2, upper_bound=max(CA3_init, CA3_sp) * 1.8)

        CB3[k=0:N], (lower_bound=min(CB3_init, CB3_sp) * 0.2, upper_bound=max(CB3_init, CB3_sp) * 1.8)

        CC3[k=0:N], (lower_bound=min(CC3_init, CC3_sp) * 0.2, upper_bound=max(CC3_init, CC3_sp) * 1.8)

        CD3[k=0:N], (lower_bound=min(CD3_init, CD3_sp) * 0.2, upper_bound=max(CD3_init, CD3_sp) * 1.8)

        F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
        F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)

        Q3[k=0:N], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)


    end



    for k = 0:N
        set_start_value(V3[k], fix.V3[k+1])

        set_start_value(T3[k], fix.T3[k+1])

        set_start_value(CA3[k], fix.CA3[k+1])

        set_start_value(CB3[k], fix.CB3[k+1])

        set_start_value(CC3[k], fix.CC3[k+1])

        set_start_value(CD3[k], fix.CD3[k+1])

        set_start_value(F5[k], fix.F5[k+1])
        set_start_value(F6[k], fix.F6[k+1])
    end

    @constraints mpc begin
        # Initial condition
        V3_inital, V3[0] == V3_init

        T3_inital, T3[0] == T3_init

        CA3_initial, CA3[0] == CA3_init

        CB3_initial, CB3[0] == CB3_init

        CC3_initial, CC3[0] == CC3_init

        CD3_initial, CD3[0] == CD3_init


        F5_hold[k in k_indices], F5[k] == F5[k+1]
        F6_hold[k in k_indices], F6[k] == F6[k+1]

        Q3_hold[k in k_indices], Q3[k] == Q3[k+1]

    end

    @NLconstraints mpc begin

        dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - fix.F7[k+1]) * dt == V3[k+1]

        dT3_dt[k=0:N-1], T3[k] + (
            ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * fix.CA2[k+1] * H_A(fix.T2[k+1]) - fix.F7[k+1] * CA3[k] * H_A(T3[k])) +
              (F5[k] * fix.CB2[k+1] * H_B(fix.T2[k+1]) - fix.F7[k+1] * CB3[k] * H_B(T3[k])) +
              (F5[k] * fix.CC2[k+1] * H_C(fix.T2[k+1]) - fix.F7[k+1] * CC3[k] * H_C(T3[k])) +
              (F5[k] * fix.CD2[k+1] * H_D(fix.T2[k+1]) - fix.F7[k+1] * CD3[k] * H_D(T3[k])))
             /
             (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
            +
            (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
            /
            (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

        dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * fix.CA2[k+1] - fix.F7[k+1] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
        dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * fix.CB2[k+1] + F6[k] * CB0 - fix.F7[k+1] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
        dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * fix.CC2[k+1] - fix.F7[k+1] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
        dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * fix.CD2[k+1] - fix.F7[k+1] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

        volHoldUp3[k in 0:dtd:(N-dtd)], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / tau

    end


    @NLobjective(
        mpc,
        Min,
        sum(
            w.v * (V3[k] - V3_sp)^2 +
            w.t3 * (T3[k] - T3_sp)^2 +
            w.ca3 * (CA3[k] - CA3_sp)^2 +
            w.cb3 * (CB3[k] - CB3_sp)^2 +
            w.cc3 * (CC3[k] - CC3_sp)^2 +
            w.cd3 * (CD3[k] - CD3_sp)^2
            for k = 0:N) +
        sum(
            w.f5 * (F5[k] - F5_sp)^2 +
            w.f6 * (F6[k] - F6_sp)^2 +
            w.q3 * (Q3[k] - Q3_sp)^2
            for k = 0:N
        )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())


    F5 = Vector(JuMP.value.(F5))
    F6 = Vector(JuMP.value.(F6))
    Q3 = Vector(JuMP.value.(Q3))

    V3 = Vector(JuMP.value.(V3))

    T3 = Vector(JuMP.value.(T3))

    CA3 = Vector(JuMP.value.(CA3))

    CB3 = Vector(JuMP.value.(CB3))

    CC3 = Vector(JuMP.value.(CC3))

    CD3 = Vector(JuMP.value.(CD3))

    return F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3

end

function dmpc_3_decomp1()

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # Volume, state, [=] m3
        V4[k=0:N], (lower_bound=min(V4_init, V4_sp) * 0.2, upper_bound=max(V4_init, V4_sp) * 1.8)
        V5[k=0:N], (lower_bound=min(V5_init, V5_sp) * 0.2, upper_bound=max(V5_init, V5_sp) * 1.8)

        # Temperature, state, [=] K
        T4[k=0:N], (lower_bound=min(T4_init, T4_sp) * 0.2, upper_bound=max(T4_init, T4_sp) * 1.8)
        T5[k=0:N], (lower_bound=min(T5_init, T5_sp) * 0.2, upper_bound=max(T5_init, T5_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA4[k=0:N], (lower_bound=min(CA4_init, CA4_sp) * 0.2, upper_bound=max(CA4_init, CA4_sp) * 1.8)
        CA5[k=0:N], (lower_bound=min(CA5_init, CA5_sp) * 0.2, upper_bound=max(CA5_init, CA5_sp) * 1.8)

        CB4[k=0:N], (lower_bound=min(CB4_init, CB4_sp) * 0.2, upper_bound=max(CB4_init, CB4_sp) * 1.8)
        CB5[k=0:N], (lower_bound=min(CB5_init, CB5_sp) * 0.2, upper_bound=max(CB5_init, CB5_sp) * 1.8)

        CC4[k=0:N], (lower_bound=min(CC4_init, CC4_sp) * 0.2, upper_bound=max(CC4_init, CC4_sp) * 1.8)
        CC5[k=0:N], (lower_bound=min(CC5_init, CC5_sp) * 0.2, upper_bound=max(CC5_init, CC5_sp) * 1.8)

        CD4[k=0:N], (lower_bound=min(CD4_init, CD4_sp) * 0.2, upper_bound=max(CD4_init, CD4_sp) * 1.8)
        CD5[k=0:N], (lower_bound=min(CD5_init, CD5_sp) * 0.2, upper_bound=max(CD5_init, CD5_sp) * 1.8)

        # Control variables
        # Flow, control [=] m3/s
        F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
        F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
        F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
        F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
        Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
        Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)

        Q4[k=0:N], (start=Q4_sp, upper_bound=1.2 * Q4_sp, lower_bound=0.8 * Q4_sp)
        Q5[k=0:N], (start=Q5_sp, upper_bound=1.2 * Q5_sp, lower_bound=0.8 * Q5_sp)
    end

    for k = 0:N
        JuMP.fix(F7[k], F7_sp; force=true)
        # JuMP.fix(F8[k], F8_sp; force=true)
        # JuMP.fix(F9[k], F9_sp; force=true)
        # JuMP.fix(F10[k], F10_sp; force=true)
        # JuMP.fix(Fr1[k], Fr1_sp; force=true)
        JuMP.fix(Fr2[k], Fr2_sp; force=true)
    end


    for k = 0:N
        set_start_value(V4[k], fix.V4[k+1])
        set_start_value(V5[k], fix.V5[k+1])

        set_start_value(T4[k], fix.T4[k+1])
        set_start_value(T5[k], fix.T5[k+1])

        set_start_value(CA4[k], fix.CA4[k+1])
        set_start_value(CA5[k], fix.CA5[k+1])

        set_start_value(CB4[k], fix.CB4[k+1])
        set_start_value(CB5[k], fix.CB5[k+1])

        set_start_value(CC4[k], fix.CC4[k+1])
        set_start_value(CC5[k], fix.CC5[k+1])

        set_start_value(CD4[k], fix.CD4[k+1])
        set_start_value(CD5[k], fix.CD5[k+1])

        set_start_value(F7[k], fix.F7[k+1])
        set_start_value(F8[k], fix.F8[k+1])
        set_start_value(F9[k], fix.F9[k+1])
        set_start_value(F10[k], fix.F10[k+1])
        set_start_value(Fr1[k], fix.Fr1[k+1])
        set_start_value(Fr2[k], fix.Fr2[k+1])

    end

    @constraints mpc begin

        # Initial condition
        V4_inital, V4[0] == V4_init
        V5_inital, V5[0] == V5_init

        T4_inital, T4[0] == T4_init
        T5_inital, T5[0] == T5_init

        CA4_initial, CA4[0] == CA4_init
        CA5_initial, CA5[0] == CA5_init

        CB4_initial, CB4[0] == CB4_init
        CB5_initial, CB5[0] == CB5_init

        CC4_initial, CC4[0] == CC4_init
        CC5_initial, CC5[0] == CC5_init

        CD4_initial, CD4[0] == CD4_init
        CD5_initial, CD5[0] == CD5_init


        F7_hold[k in k_indices], F7[k] == F7[k+1]
        F8_hold[k in k_indices], F8[k] == F8[k+1]
        F9_hold[k in k_indices], F9[k] == F9[k+1]
        F10_hold[k in k_indices], F10[k] == F10[k+1]
        Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
        Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

        Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
        Q5_hold[k in k_indices], Q5[k] == Q5[k+1]


    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system

        dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4[k+1]
        dT4_dt[k=0:N-1], T4[k] +
                         ((Q4[k]
                           + (F7[k] * fix.CA3[k+1] * H_A(fix.T3[k+1]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
                           + (F7[k] * fix.CB3[k+1] * H_B(fix.T3[k+1]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
                           + (F7[k] * fix.CC3[k+1] * H_C(fix.T3[k+1]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
                           + (F7[k] * fix.CD3[k+1] * H_D(fix.T3[k+1]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
                          /
                          (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

        dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * fix.CA3[k+1] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
        dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * fix.CB3[k+1] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
        dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * fix.CC3[k+1] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
        dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * fix.CD3[k+1] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

        dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5[k+1]

        dT5_dt[k=0:N-1], T5[k] + (
            ((Q5[k] +
              F10[k] * CD0 * H_D(TD0)
              + (Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
              + (Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
              + (Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
              + (Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
             /
             (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
            +
            ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
             /
             (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

        dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
        dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
        dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
        dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]


        volHoldUp4[k in 0:dtd:(N-dtd)], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        volHoldUp5[k in 0:dtd:(N-dtd)], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau


    end


    @NLobjective(mpc, Min, sum(
        w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
        w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
        w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
        w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
        w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
        w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
        for k = 0:N) +
                           sum(
        w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
        w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
        w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
        w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
        for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc3_solve_time = MOI.get(mpc, MOI.SolveTimeSec())
    F7 = Vector(JuMP.value.(F7))
    F8 = Vector(JuMP.value.(F8))
    F9 = Vector(JuMP.value.(F9))
    F10 = Vector(JuMP.value.(F10))
    Fr1 = Vector(JuMP.value.(Fr1))
    Fr2 = Vector(JuMP.value.(Fr2))

    Q4 = Vector(JuMP.value.(Q4))
    Q5 = Vector(JuMP.value.(Q5))

    V4 = Vector(JuMP.value.(V4))
    V5 = Vector(JuMP.value.(V5))

    T4 = Vector(JuMP.value.(T4))
    T5 = Vector(JuMP.value.(T5))

    CA4 = Vector(JuMP.value.(CA4))
    CA5 = Vector(JuMP.value.(CA5))

    CB4 = Vector(JuMP.value.(CB4))
    CB5 = Vector(JuMP.value.(CB5))

    CC4 = Vector(JuMP.value.(CC4))
    CC5 = Vector(JuMP.value.(CC5))

    CD4 = Vector(JuMP.value.(CD4))
    CD5 = Vector(JuMP.value.(CD5))

    return F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5
end

function dmpc_1_decomp2()

    # Control variables: F1, F2, F3, F4, Q1, Q2
    # State variables: All state variables associated with CSTR-1 and CSTR-2

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin

        # State variables

        # Volume, state, [=] m3
        V1[k=0:N], (lower_bound=min(V1_init, V1_sp) * 0.2, upper_bound=max(V1_init, V1_sp) * 1.8)

        # Temperature, state, [=] K
        T1[k=0:N], (lower_bound=min(T1_init, T1_sp) * 0.2, upper_bound=max(T1_init, T1_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA1[k=0:N], (lower_bound=min(CA1_init, CA1_sp) * 0.2, upper_bound=max(CA1_init, CA1_sp) * 1.8)

        CB1[k=0:N], (lower_bound=min(CB1_init, CB1_sp) * 0.2, upper_bound=max(CB1_init, CB1_sp) * 1.8)

        CC1[k=0:N], (lower_bound=min(CC1_init, CC1_sp) * 0.2, upper_bound=max(CC1_init, CC1_sp) * 1.8)

        CD1[k=0:N], (lower_bound=min(CD1_init, CD1_sp) * 0.2, upper_bound=max(CD1_init, CD1_sp) * 1.8)


        F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
        F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)

        Q1[k=0:N], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)

    end

    for k = 0:N
        set_start_value(V1[k], fix.V1[k+1])

        set_start_value(T1[k], fix.T1[k+1])

        set_start_value(CA1[k], fix.CA1[k+1])

        set_start_value(CB1[k], fix.CB1[k+1])

        set_start_value(CC1[k], fix.CC1[k+1])

        set_start_value(CD1[k], fix.CD1[k+1])

        set_start_value(F1[k], fix.F1[k+1])
        set_start_value(F2[k], fix.F2[k+1])

        set_start_value(Q1[k], fix.Q1[k+1])
    end

    @constraints mpc begin

        # Initial condition
        V1_inital, V1[0] == V1_init

        T1_inital, T1[0] == T1_init

        CA1_initial, CA1[0] == CA1_init

        CB1_initial, CB1[0] == CB1_init

        CC1_initial, CC1[0] == CC1_init

        CD1_initial, CD1[0] == CD1_init

        F1_hold[k in k_indices], F1[k] == F1[k+1]
        F2_hold[k in k_indices], F2[k] == F2[k+1]

        Q1_hold[k in k_indices], Q1[k] == Q1[k+1]


    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system
        dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + fix.Fr2[k+1] - fix.F3[k+1]) * dt == V1[k+1]

        dT1_dt[k=0:N-1], T1[k] + (
            ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
              (fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_A(fix.T4[k+1]) - fix.F3[k+1] * CA1[k] * H_A(T1[k])) +
              (fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_B(fix.T4[k+1]) - fix.F3[k+1] * CB1[k] * H_B(T1[k])) +
              (fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_C(fix.T4[k+1]) - fix.F3[k+1] * CC1[k] * H_C(T1[k])) +
              (fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_D(fix.T4[k+1]) - fix.F3[k+1] * CD1[k] * H_D(T1[k])))
             /
             (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
            (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
            (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

        dCA1_dt[k=0:N-1], CA1[k] + (
            ((F1[k] * CA0 + fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CA1[k])
             /
             V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

        dCB1_dt[k=0:N-1], CB1[k] + (
            ((F2[k] * CB0 + fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CB1[k]) /
             V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

        dCC1_dt[k=0:N-1], CC1[k] + (
            ((fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CC1[k]) /
             V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

        dCD1_dt[k=0:N-1], CD1[k] + (
            ((fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CD1[k]) /
             V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

        volHoldUp1[k in 0:dtd:(N-dtd)], F1[k] + F2[k] + fix.Fr2[k+1] - fix.F3[k+1] == -(V1[k] - V1_sp) / tau

    end

    @NLobjective(mpc, Min, sum(
        w.v * (V1[k] - V1_sp)^2 +
        w.t1 * (T1[k] - T1_sp)^2 +
        w.ca1 * (CA1[k] - CA1_sp)^2 +
        w.cb1 * (CB1[k] - CB1_sp)^2 +
        w.cc1 * (CC1[k] - CC1_sp)^2 +
        w.cd1 * (CD1[k] - CD1_sp)^2
        for k = 0:N) + sum(
        w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 +
        w.q1 * (Q1[k] - Q1_sp)^2 for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)
    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc1_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

    F1 = Vector(JuMP.value.(F1))
    F2 = Vector(JuMP.value.(F2))
    Q1 = Vector(JuMP.value.(Q1))

    V1 = Vector(JuMP.value.(V1))

    T1 = Vector(JuMP.value.(T1))

    CA1 = Vector(JuMP.value.(CA1))

    CB1 = Vector(JuMP.value.(CB1))

    CC1 = Vector(JuMP.value.(CC1))

    CD1 = Vector(JuMP.value.(CD1))

    return F1, F2, Q1, V1, T1, CA1, CB1, CC1, CD1
end

function dmpc_2_decomp2()

    # Control variables: F5, F6, Q3
    # State variables: All state variables associated with CSTR-3

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # State variables

        # Volume, state, [=] m3
        V2[k=0:N], (lower_bound=min(V2_init, V2_sp) * 0.2, upper_bound=max(V2_init, V2_sp) * 1.8)

        # Temperature, state, [=] K
        T2[k=0:N], (lower_bound=min(T2_init, T2_sp) * 0.2, upper_bound=max(T2_init, T2_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA2[k=0:N], (lower_bound=min(CA2_init, CA2_sp) * 0.2, upper_bound=max(CA2_init, CA2_sp) * 1.8)

        CB2[k=0:N], (lower_bound=min(CB2_init, CB2_sp) * 0.2, upper_bound=max(CB2_init, CB2_sp) * 1.8)

        CC2[k=0:N], (lower_bound=min(CC2_init, CC2_sp) * 0.2, upper_bound=max(CC2_init, CC2_sp) * 1.8)

        CD2[k=0:N], (lower_bound=min(CD2_init, CD2_sp) * 0.2, upper_bound=max(CD2_init, CD2_sp) * 1.8)

        F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
        F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)

        Q2[k=0:N], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)

    end


    for k = 0:N
        set_start_value(V2[k], fix.V2[k+1])

        set_start_value(T2[k], fix.T2[k+1])

        set_start_value(CA2[k], fix.CA2[k+1])

        set_start_value(CB2[k], fix.CB2[k+1])

        set_start_value(CC2[k], fix.CC2[k+1])

        set_start_value(CD2[k], fix.CD2[k+1])

        set_start_value(F3[k], fix.F3[k+1])
        set_start_value(F4[k], fix.F4[k+1])
    end

    @constraints mpc begin
        # Initial condition
        V2_inital, V2[0] == V2_init

        T2_inital, T2[0] == T2_init

        CA2_initial, CA2[0] == CA2_init

        CB2_initial, CB2[0] == CB2_init

        CC2_initial, CC2[0] == CC2_init

        CD2_initial, CD2[0] == CD2_init

        # F2_hold[k in k_indices], F2[k] == F2[k+1]
        F3_hold[k in k_indices], F3[k] == F3[k+1]
        F4_hold[k in k_indices], F4[k] == F4[k+1]

        Q2_hold[k in k_indices], Q2[k] == Q2[k+1]

    end

    @NLconstraints mpc begin

        dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - fix.F5[k+1]) * dt == V2[k+1]

        dT2_dt[k=0:N-1], T2[k] + (
            (Q2[k] + F4[k] * CB0 * H_B(TB0) +
             (F3[k] * fix.CA1[k+1] * H_A(fix.T1[k+1]) - fix.F5[k+1] * CA2[k] * H_A(T2[k])) +
             (F3[k] * fix.CB1[k+1] * H_B(fix.T1[k+1]) - fix.F5[k+1] * CB2[k] * H_B(T2[k])) +
             (F3[k] * fix.CC1[k+1] * H_C(fix.T1[k+1]) - fix.F5[k+1] * CC2[k] * H_C(T2[k])) +
             (F3[k] * fix.CD1[k+1] * H_D(fix.T1[k+1]) - fix.F5[k+1] * CD2[k] * H_D(T2[k])))
            /
            (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
            +
            (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
            /
            (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


        dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * fix.CA1[k+1] - fix.F5[k+1] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
        dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * fix.CB1[k+1] + F4[k] * CB0 - fix.F5[k+1] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
        dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * fix.CC1[k+1] - fix.F5[k+1] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
        dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * fix.CD1[k+1] - fix.F5[k+1] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]

        volHoldUp2[k in 0:dtd:(N-dtd)], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / tau

    end


    @NLobjective(
        mpc,
        Min,
        sum(
            w.v * (V2[k] - V2_sp)^2 +
            w.t2 * (T2[k] - T2_sp)^2 +
            w.ca2 * (CA2[k] - CA2_sp)^2 +
            w.cb2 * (CB2[k] - CB2_sp)^2 +
            w.cc2 * (CC2[k] - CC2_sp)^2 +
            w.cd2 * (CD2[k] - CD2_sp)^2
            for k = 0:N) +
        sum(
            w.f3 * (F3[k] - F3_sp)^2 +
            w.f4 * (F4[k] - F4_sp)^2 +
            w.q2 * (Q2[k] - Q2_sp)^2
            for k = 0:N
        )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())


    F3 = Vector(JuMP.value.(F3))
    F4 = Vector(JuMP.value.(F4))
    Q2 = Vector(JuMP.value.(Q2))

    V2 = Vector(JuMP.value.(V2))

    T2 = Vector(JuMP.value.(T2))

    CA2 = Vector(JuMP.value.(CA2))

    CB2 = Vector(JuMP.value.(CB2))

    CC2 = Vector(JuMP.value.(CC2))

    CD2 = Vector(JuMP.value.(CD2))

    return F3, F4, Q2, V2, T2, CA2, CB2, CC2, CD2

end

function dmpc_3_decomp2()

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # Volume, state, [=] m3    
        V3[k=0:N], (lower_bound=min(V3_init, V3_sp) * 0.2, upper_bound=max(V3_init, V3_sp) * 1.8)

        # Temperature, state, [=] K
        T3[k=0:N], (lower_bound=min(T3_init, T3_sp) * 0.2, upper_bound=max(T3_init, T3_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA3[k=0:N], (lower_bound=min(CA3_init, CA3_sp) * 0.2, upper_bound=max(CA3_init, CA3_sp) * 1.8)

        CB3[k=0:N], (lower_bound=min(CB3_init, CB3_sp) * 0.2, upper_bound=max(CB3_init, CB3_sp) * 1.8)

        CC3[k=0:N], (lower_bound=min(CC3_init, CC3_sp) * 0.2, upper_bound=max(CC3_init, CC3_sp) * 1.8)

        CD3[k=0:N], (lower_bound=min(CD3_init, CD3_sp) * 0.2, upper_bound=max(CD3_init, CD3_sp) * 1.8)

        V4[k=0:N], (lower_bound=min(V4_init, V4_sp) * 0.2, upper_bound=max(V4_init, V4_sp) * 1.8)
        V5[k=0:N], (lower_bound=min(V5_init, V5_sp) * 0.2, upper_bound=max(V5_init, V5_sp) * 1.8)

        # Temperature, state, [=] K
        T4[k=0:N], (lower_bound=min(T4_init, T4_sp) * 0.2, upper_bound=max(T4_init, T4_sp) * 1.8)
        T5[k=0:N], (lower_bound=min(T5_init, T5_sp) * 0.2, upper_bound=max(T5_init, T5_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA4[k=0:N], (lower_bound=min(CA4_init, CA4_sp) * 0.2, upper_bound=max(CA4_init, CA4_sp) * 1.8)
        CA5[k=0:N], (lower_bound=min(CA5_init, CA5_sp) * 0.2, upper_bound=max(CA5_init, CA5_sp) * 1.8)

        CB4[k=0:N], (lower_bound=min(CB4_init, CB4_sp) * 0.2, upper_bound=max(CB4_init, CB4_sp) * 1.8)
        CB5[k=0:N], (lower_bound=min(CB5_init, CB5_sp) * 0.2, upper_bound=max(CB5_init, CB5_sp) * 1.8)

        CC4[k=0:N], (lower_bound=min(CC4_init, CC4_sp) * 0.2, upper_bound=max(CC4_init, CC4_sp) * 1.8)
        CC5[k=0:N], (lower_bound=min(CC5_init, CC5_sp) * 0.2, upper_bound=max(CC5_init, CC5_sp) * 1.8)

        CD4[k=0:N], (lower_bound=min(CD4_init, CD4_sp) * 0.2, upper_bound=max(CD4_init, CD4_sp) * 1.8)
        CD5[k=0:N], (lower_bound=min(CD5_init, CD5_sp) * 0.2, upper_bound=max(CD5_init, CD5_sp) * 1.8)

        # Control variables
        # Flow, control [=] m3/s
        F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
        F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)
        F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
        F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
        F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
        F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
        Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
        Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)

        Q3[k=0:N], (start=Q3_sp, lower_bound=1.2 * Q3_sp, upper_bound=0.8 * Q3_sp)
        Q4[k=0:N], (start=Q4_sp, upper_bound=1.2 * Q4_sp, lower_bound=0.8 * Q4_sp)
        Q5[k=0:N], (start=Q5_sp, upper_bound=1.2 * Q5_sp, lower_bound=0.8 * Q5_sp)

    end

    for k = 0:N
        JuMP.fix(F7[k], F7_sp; force=true)
        JuMP.fix(Fr2[k], Fr2_sp; force=true)
    end



    for k = 0:N
        set_start_value(V3[k], fix.V3[k+1])
        set_start_value(V4[k], fix.V4[k+1])
        set_start_value(V5[k], fix.V5[k+1])

        set_start_value(T3[k], fix.T3[k+1])
        set_start_value(T4[k], fix.T4[k+1])
        set_start_value(T5[k], fix.T5[k+1])

        set_start_value(CA3[k], fix.CA3[k+1])
        set_start_value(CA4[k], fix.CA4[k+1])
        set_start_value(CA5[k], fix.CA5[k+1])

        set_start_value(CB3[k], fix.CB3[k+1])
        set_start_value(CB4[k], fix.CB4[k+1])
        set_start_value(CB5[k], fix.CB5[k+1])

        set_start_value(CC3[k], fix.CC3[k+1])
        set_start_value(CC4[k], fix.CC4[k+1])
        set_start_value(CC5[k], fix.CC5[k+1])

        set_start_value(CD3[k], fix.CD3[k+1])
        set_start_value(CD4[k], fix.CD4[k+1])
        set_start_value(CD5[k], fix.CD5[k+1])

        set_start_value(F5[k], fix.F5[k+1])
        set_start_value(F6[k], fix.F6[k+1])
        set_start_value(F7[k], fix.F7[k+1])
        set_start_value(F8[k], fix.F8[k+1])
        set_start_value(F9[k], fix.F9[k+1])
        set_start_value(F10[k], fix.F10[k+1])
        set_start_value(Fr1[k], fix.Fr1[k+1])
        set_start_value(Fr2[k], fix.Fr2[k+1])

    end

    @constraints mpc begin

        # Initial condition
        V3_inital, V3[0] == V3_init
        V4_inital, V4[0] == V4_init
        V5_inital, V5[0] == V5_init

        T3_inital, T3[0] == T3_init
        T4_inital, T4[0] == T4_init
        T5_inital, T5[0] == T5_init

        CA3_initial, CA3[0] == CA3_init
        CA4_initial, CA4[0] == CA4_init
        CA5_initial, CA5[0] == CA5_init

        CB3_initial, CB3[0] == CB3_init
        CB4_initial, CB4[0] == CB4_init
        CB5_initial, CB5[0] == CB5_init

        CC3_initial, CC3[0] == CC3_init
        CC4_initial, CC4[0] == CC4_init
        CC5_initial, CC5[0] == CC5_init

        CD3_initial, CD3[0] == CD3_init
        CD4_initial, CD4[0] == CD4_init
        CD5_initial, CD5[0] == CD5_init


        F5_hold[k in k_indices], F5[k] == F5[k+1]
        F6_hold[k in k_indices], F6[k] == F6[k+1]
        F7_hold[k in k_indices], F7[k] == F7[k+1]
        F8_hold[k in k_indices], F8[k] == F8[k+1]
        F9_hold[k in k_indices], F9[k] == F9[k+1]
        F10_hold[k in k_indices], F10[k] == F10[k+1]
        Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
        Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

        Q3_hold[k in k_indices], Q3[k] == Q3[k+1]
        Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
        Q5_hold[k in k_indices], Q5[k] == Q5[k+1]


    end

    @NLconstraints mpc begin

        dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - F7[k]) * dt == V3[k+1]

        dT3_dt[k=0:N-1], T3[k] + (
            ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * fix.CA2[k+1] * H_A(fix.T2[k+1]) - F7[k] * CA3[k] * H_A(T3[k])) +
              (F5[k] * fix.CB2[k+2] * H_B(fix.T2[k+1]) - F7[k] * CB3[k] * H_B(T3[k])) +
              (F5[k] * fix.CC2[k+1] * H_C(fix.T2[k+1]) - F7[k] * CC3[k] * H_C(T3[k])) +
              (F5[k] * fix.CD2[k+1] * H_D(fix.T2[k+1]) - F7[k] * CD3[k] * H_D(T3[k])))
             /
             (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
            +
            (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
            /
            (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

        dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * fix.CA2[k+1] - F7[k] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
        dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * fix.CB2[k+2] + F6[k] * CB0 - F7[k] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
        dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * fix.CC2[k+1] - F7[k] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
        dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * fix.CD2[k+1] - F7[k] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

        dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4[k+1]

        dT4_dt[k=0:N-1], T4[k] +
                         ((Q4[k]
                           + (F7[k] * CA3[k] * H_A(T3[k]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
                           + (F7[k] * CB3[k] * H_B(T3[k]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
                           + (F7[k] * CC3[k] * H_C(T3[k]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
                           + (F7[k] * CD3[k] * H_D(T3[k]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
                          /
                          (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

        dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * CA3[k] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
        dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * CB3[k] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
        dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * CC3[k] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
        dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * CD3[k] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

        dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5[k+1]

        dT5_dt[k=0:N-1], T5[k] + (
            ((Q5[k] +
              F10[k] * CD0 * H_D(TD0)
              + (Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
              + (Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
              + (Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
              + (Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
             /
             (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
            +
            ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
             /
             (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

        dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
        dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
        dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
        dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]

        volHoldUp3[k in 0:dtd:(N-dtd)], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / tau
        volHoldUp4[k in 0:dtd:(N-dtd)], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        volHoldUp5[k in 0:dtd:(N-dtd)], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau



    end


    @NLobjective(mpc, Min, sum(
        w.v * (V3[k] - V3_sp)^2 + w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
        w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
        w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
        w.cb3 * (CB3[k] - CB3_sp)^2 + w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
        w.cc3 * (CC3[k] - CC3_sp)^2 + w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
        w.cd3 * (CD3[k] - CD3_sp)^2 + w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
        for k = 0:N) +
                           sum(
        w.f5 * (F5[k] - F5_sp)^2 + w.f6 * (F6[k] - F6_sp)^2 +
        w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
        w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
        w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
        w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
        for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc3_solve_time = MOI.get(mpc, MOI.SolveTimeSec())
    F5 = Vector(JuMP.value.(F5))
    F6 = Vector(JuMP.value.(F6))
    F7 = Vector(JuMP.value.(F7))
    F8 = Vector(JuMP.value.(F8))
    F9 = Vector(JuMP.value.(F9))
    F10 = Vector(JuMP.value.(F10))
    Fr1 = Vector(JuMP.value.(Fr1))
    Fr2 = Vector(JuMP.value.(Fr2))

    Q3 = Vector(JuMP.value.(Q3))
    Q4 = Vector(JuMP.value.(Q4))
    Q5 = Vector(JuMP.value.(Q5))

    V3 = Vector(JuMP.value.(V3))
    V4 = Vector(JuMP.value.(V4))
    V5 = Vector(JuMP.value.(V5))

    T3 = Vector(JuMP.value.(T3))
    T4 = Vector(JuMP.value.(T4))
    T5 = Vector(JuMP.value.(T5))

    CA3 = Vector(JuMP.value.(CA3))
    CA4 = Vector(JuMP.value.(CA4))
    CA5 = Vector(JuMP.value.(CA5))

    CB3 = Vector(JuMP.value.(CB3))
    CB4 = Vector(JuMP.value.(CB4))
    CB5 = Vector(JuMP.value.(CB5))

    CC3 = Vector(JuMP.value.(CC3))
    CC4 = Vector(JuMP.value.(CC4))
    CC5 = Vector(JuMP.value.(CC5))

    CD3 = Vector(JuMP.value.(CD3))
    CD4 = Vector(JuMP.value.(CD4))
    CD5 = Vector(JuMP.value.(CD5))

    return F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q3, Q4, Q5, V3, V4, V5, T3, T4, T5, CA3, CA4, CA5, CB3, CB4, CB5, CC3, CC4, CC5, CD3, CD4, CD5
end

function dmpc_1_decomp3()

    # Control variables: F5, F6, Q3
    # State variables: All state variables associated with CSTR-3

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # State variables

        # Volume, state, [=] m3
        V2[k=0:N], (lower_bound=min(V2_init, V2_sp) * 0.2, upper_bound=max(V2_init, V2_sp) * 1.8)

        # Temperature, state, [=] K
        T2[k=0:N], (lower_bound=min(T2_init, T2_sp) * 0.2, upper_bound=max(T2_init, T2_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA2[k=0:N], (lower_bound=min(CA2_init, CA2_sp) * 0.2, upper_bound=max(CA2_init, CA2_sp) * 1.8)

        CB2[k=0:N], (lower_bound=min(CB2_init, CB2_sp) * 0.2, upper_bound=max(CB2_init, CB2_sp) * 1.8)

        CC2[k=0:N], (lower_bound=min(CC2_init, CC2_sp) * 0.2, upper_bound=max(CC2_init, CC2_sp) * 1.8)

        CD2[k=0:N], (lower_bound=min(CD2_init, CD2_sp) * 0.2, upper_bound=max(CD2_init, CD2_sp) * 1.8)

        F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
        F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)

        Q2[k=0:N], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)


    end


    for k = 0:N
        set_start_value(V2[k], fix.V2[k+1])

        set_start_value(T2[k], fix.T2[k+1])

        set_start_value(CA2[k], fix.CA2[k+1])

        set_start_value(CB2[k], fix.CB2[k+1])

        set_start_value(CC2[k], fix.CC2[k+1])

        set_start_value(CD2[k], fix.CD2[k+1])

        set_start_value(F3[k], fix.F3[k+1])
        set_start_value(F4[k], fix.F4[k+1])
    end

    @constraints mpc begin
        # Initial condition
        V2_inital, V2[0] == V2_init

        T2_inital, T2[0] == T2_init

        CA2_initial, CA2[0] == CA2_init

        CB2_initial, CB2[0] == CB2_init

        CC2_initial, CC2[0] == CC2_init

        CD2_initial, CD2[0] == CD2_init

        # F2_hold[k in k_indices], F2[k] == F2[k+1]
        F3_hold[k in k_indices], F3[k] == F3[k+1]
        F4_hold[k in k_indices], F4[k] == F4[k+1]

        Q2_hold[k in k_indices], Q2[k] == Q2[k+1]

    end

    @NLconstraints mpc begin

        dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - fix.F5[k+1]) * dt == V2[k+1]

        dT2_dt[k=0:N-1], T2[k] + (
            (Q2[k] + F4[k] * CB0 * H_B(TB0) +
             (F3[k] * fix.CA1[k+1] * H_A(fix.T1[k+1]) - fix.F5[k+1] * CA2[k] * H_A(T2[k])) +
             (F3[k] * fix.CB1[k+1] * H_B(fix.T1[k+1]) - fix.F5[k+1] * CB2[k] * H_B(T2[k])) +
             (F3[k] * fix.CC1[k+1] * H_C(fix.T1[k+1]) - fix.F5[k+1] * CC2[k] * H_C(T2[k])) +
             (F3[k] * fix.CD1[k+1] * H_D(fix.T1[k+1]) - fix.F5[k+1] * CD2[k] * H_D(T2[k])))
            /
            (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
            +
            (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
            /
            (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


        dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * fix.CA1[k+1] - fix.F5[k+1] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
        dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * fix.CB1[k+1] + F4[k] * CB0 - fix.F5[k+1] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
        dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * fix.CC1[k+1] - fix.F5[k+1] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
        dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * fix.CD1[k+1] - fix.F5[k+1] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]

        volHoldUp2[k in 0:dtd:(N-dtd)], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / tau

    end


    @NLobjective(
        mpc,
        Min,
        sum(
            w.v * (V2[k] - V2_sp)^2 +
            w.t2 * (T2[k] - T2_sp)^2 +
            w.ca2 * (CA2[k] - CA2_sp)^2 +
            w.cb2 * (CB2[k] - CB2_sp)^2 +
            w.cc2 * (CC2[k] - CC2_sp)^2 +
            w.cd2 * (CD2[k] - CD2_sp)^2
            for k = 0:N) +
        sum(
            w.f3 * (F3[k] - F3_sp)^2 +
            w.f4 * (F4[k] - F4_sp)^2 +
            w.q2 * (Q2[k] - Q2_sp)^2
            for k = 0:N
        )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())


    F3 = Vector(JuMP.value.(F3))
    F4 = Vector(JuMP.value.(F4))
    Q2 = Vector(JuMP.value.(Q2))

    V2 = Vector(JuMP.value.(V2))

    T2 = Vector(JuMP.value.(T2))

    CA2 = Vector(JuMP.value.(CA2))

    CB2 = Vector(JuMP.value.(CB2))

    CC2 = Vector(JuMP.value.(CC2))

    CD2 = Vector(JuMP.value.(CD2))

    return F3, F4, Q2, V2, T2, CA2, CB2, CC2, CD2

end

function dmpc_2_decomp3()

    # Control variables: F5, F6, Q3
    # State variables: All state variables associated with CSTR-3

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # State variables

        # Volume, state, [=] m3    
        V3[k=0:N], (lower_bound=min(V3_init, V3_sp) * 0.2, upper_bound=max(V3_init, V3_sp) * 1.8)

        # Temperature, state, [=] K
        T3[k=0:N], (lower_bound=min(T3_init, T3_sp) * 0.2, upper_bound=max(T3_init, T3_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA3[k=0:N], (lower_bound=min(CA3_init, CA3_sp) * 0.2, upper_bound=max(CA3_init, CA3_sp) * 1.8)

        CB3[k=0:N], (lower_bound=min(CB3_init, CB3_sp) * 0.2, upper_bound=max(CB3_init, CB3_sp) * 1.8)

        CC3[k=0:N], (lower_bound=min(CC3_init, CC3_sp) * 0.2, upper_bound=max(CC3_init, CC3_sp) * 1.8)

        CD3[k=0:N], (lower_bound=min(CD3_init, CD3_sp) * 0.2, upper_bound=max(CD3_init, CD3_sp) * 1.8)

        F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
        F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)

        Q3[k=0:N], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)


    end

    for k = 0:N
        # JuMP.fix(F5[k], F5_sp; force=true)
        # JuMP.fix(F6[k], F6_sp; force=true)
    end

    for k = 0:N
        # JuMP.fix(Q1[k], Q1_sp; force=true)
        # JuMP.fix(Q2[k], Q2_sp; force=true)
        # JuMP.fix(Q3[k], Q3_sp; force=true)
        # JuMP.fix(Q4[k], Q4_sp; force=true)
        # JuMP.fix(Q5[k], Q5_sp; force=true)
    end

    for k = 0:N
        set_start_value(V3[k], fix.V3[k+1])

        set_start_value(T3[k], fix.T3[k+1])

        set_start_value(CA3[k], fix.CA3[k+1])

        set_start_value(CB3[k], fix.CB3[k+1])

        set_start_value(CC3[k], fix.CC3[k+1])

        set_start_value(CD3[k], fix.CD3[k+1])

        set_start_value(F5[k], fix.F5[k+1])
        set_start_value(F6[k], fix.F6[k+1])
    end

    @constraints mpc begin
        # Initial condition
        V3_inital, V3[0] == V3_init

        T3_inital, T3[0] == T3_init

        CA3_initial, CA3[0] == CA3_init

        CB3_initial, CB3[0] == CB3_init

        CC3_initial, CC3[0] == CC3_init

        CD3_initial, CD3[0] == CD3_init


        F5_hold[k in k_indices], F5[k] == F5[k+1]
        F6_hold[k in k_indices], F6[k] == F6[k+1]

        Q3_hold[k in k_indices], Q3[k] == Q3[k+1]

    end

    @NLconstraints mpc begin

        dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - fix.F7[k+1]) * dt == V3[k+1]

        dT3_dt[k=0:N-1], T3[k] + (
            ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * fix.CA2[k+1] * H_A(fix.T2[k+1]) - fix.F7[k+1] * CA3[k] * H_A(T3[k])) +
              (F5[k] * fix.CB2[k+1] * H_B(fix.T2[k+1]) - fix.F7[k+1] * CB3[k] * H_B(T3[k])) +
              (F5[k] * fix.CC2[k+1] * H_C(fix.T2[k+1]) - fix.F7[k+1] * CC3[k] * H_C(T3[k])) +
              (F5[k] * fix.CD2[k+1] * H_D(fix.T2[k+1]) - fix.F7[k+1] * CD3[k] * H_D(T3[k])))
             /
             (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
            +
            (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
            /
            (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

        dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * fix.CA2[k+1] - fix.F7[k+1] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
        dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * fix.CB2[k+1] + F6[k] * CB0 - fix.F7[k+1] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
        dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * fix.CC2[k+1] - fix.F7[k+1] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
        dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * fix.CD2[k+1] - fix.F7[k+1] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

        volHoldUp3[k in 0:dtd:(N-dtd)], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / tau
    end


    @NLobjective(
        mpc,
        Min,
        sum(
            w.v * (V3[k] - V3_sp)^2 +
            w.t3 * (T3[k] - T3_sp)^2 +
            w.ca3 * (CA3[k] - CA3_sp)^2 +
            w.cb3 * (CB3[k] - CB3_sp)^2 +
            w.cc3 * (CC3[k] - CC3_sp)^2 +
            w.cd3 * (CD3[k] - CD3_sp)^2
            for k = 0:N) +
        sum(
            w.f5 * (F5[k] - F5_sp)^2 +
            w.f6 * (F6[k] - F6_sp)^2 +
            w.q3 * (Q3[k] - Q3_sp)^2
            for k = 0:N
        )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())


    F5 = Vector(JuMP.value.(F5))
    F6 = Vector(JuMP.value.(F6))
    Q3 = Vector(JuMP.value.(Q3))

    V3 = Vector(JuMP.value.(V3))

    T3 = Vector(JuMP.value.(T3))

    CA3 = Vector(JuMP.value.(CA3))

    CB3 = Vector(JuMP.value.(CB3))

    CC3 = Vector(JuMP.value.(CC3))

    CD3 = Vector(JuMP.value.(CD3))

    return F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3

end

function dmpc_3_decomp3()

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # Volume, state, [=] m3    
        V1[k=0:N], (lower_bound=min(V1_init, V1_sp) * 0.2, upper_bound=max(V1_init, V1_sp) * 1.8)

        # Temperature, state, [=] K
        T1[k=0:N], (lower_bound=min(T1_init, T1_sp) * 0.2, upper_bound=max(T1_init, T1_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA1[k=0:N], (lower_bound=min(CA1_init, CA1_sp) * 0.2, upper_bound=max(CA1_init, CA1_sp) * 1.8)

        CB1[k=0:N], (lower_bound=min(CB1_init, CB1_sp) * 0.2, upper_bound=max(CB1_init, CB1_sp) * 1.8)

        CC1[k=0:N], (lower_bound=min(CC1_init, CC1_sp) * 0.2, upper_bound=max(CC1_init, CC1_sp) * 1.8)

        CD1[k=0:N], (lower_bound=min(CD1_init, CD1_sp) * 0.2, upper_bound=max(CD1_init, CD1_sp) * 1.8)

        V4[k=0:N], (lower_bound=min(V4_init, V4_sp) * 0.2, upper_bound=max(V4_init, V4_sp) * 1.8)
        V5[k=0:N], (lower_bound=min(V5_init, V5_sp) * 0.2, upper_bound=max(V5_init, V5_sp) * 1.8)

        # Temperature, state, [=] K
        T4[k=0:N], (lower_bound=min(T4_init, T4_sp) * 0.2, upper_bound=max(T4_init, T4_sp) * 1.8)
        T5[k=0:N], (lower_bound=min(T5_init, T5_sp) * 0.2, upper_bound=max(T5_init, T5_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA4[k=0:N], (lower_bound=min(CA4_init, CA4_sp) * 0.2, upper_bound=max(CA4_init, CA4_sp) * 1.8)
        CA5[k=0:N], (lower_bound=min(CA5_init, CA5_sp) * 0.2, upper_bound=max(CA5_init, CA5_sp) * 1.8)

        CB4[k=0:N], (lower_bound=min(CB4_init, CB4_sp) * 0.2, upper_bound=max(CB4_init, CB4_sp) * 1.8)
        CB5[k=0:N], (lower_bound=min(CB5_init, CB5_sp) * 0.2, upper_bound=max(CB5_init, CB5_sp) * 1.8)

        CC4[k=0:N], (lower_bound=min(CC4_init, CC4_sp) * 0.2, upper_bound=max(CC4_init, CC4_sp) * 1.8)
        CC5[k=0:N], (lower_bound=min(CC5_init, CC5_sp) * 0.2, upper_bound=max(CC5_init, CC5_sp) * 1.8)

        CD4[k=0:N], (lower_bound=min(CD4_init, CD4_sp) * 0.2, upper_bound=max(CD4_init, CD4_sp) * 1.8)
        CD5[k=0:N], (lower_bound=min(CD5_init, CD5_sp) * 0.2, upper_bound=max(CD5_init, CD5_sp) * 1.8)

        # Control variables
        # Flow, control [=] m3/s
        F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
        F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)
        F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
        F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
        F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
        F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
        Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
        Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)

        Q1[k=0:N], (start=Q1_sp, lower_bound=1.2 * Q1_sp, upper_bound=0.8 * Q1_sp)
        Q4[k=0:N], (start=Q4_sp, upper_bound=1.2 * Q4_sp, lower_bound=0.8 * Q4_sp)
        Q5[k=0:N], (start=Q5_sp, upper_bound=1.2 * Q5_sp, lower_bound=0.8 * Q5_sp)

    end

    for k = 0:N
        JuMP.fix(F7[k], F7_sp; force=true)
        JuMP.fix(Fr2[k], Fr2_sp; force=true)
    end



    for k = 0:N
        set_start_value(V1[k], fix.V1[k+1])
        set_start_value(V4[k], fix.V4[k+1])
        set_start_value(V5[k], fix.V5[k+1])

        set_start_value(T1[k], fix.T1[k+1])
        set_start_value(T4[k], fix.T4[k+1])
        set_start_value(T5[k], fix.T5[k+1])

        set_start_value(CA1[k], fix.CA1[k+1])
        set_start_value(CA4[k], fix.CA4[k+1])
        set_start_value(CA5[k], fix.CA5[k+1])

        set_start_value(CB1[k], fix.CB1[k+1])
        set_start_value(CB4[k], fix.CB4[k+1])
        set_start_value(CB5[k], fix.CB5[k+1])

        set_start_value(CC1[k], fix.CC1[k+1])
        set_start_value(CC4[k], fix.CC4[k+1])
        set_start_value(CC5[k], fix.CC5[k+1])

        set_start_value(CD1[k], fix.CD1[k+1])
        set_start_value(CD4[k], fix.CD4[k+1])
        set_start_value(CD5[k], fix.CD5[k+1])

        set_start_value(F1[k], fix.F1[k+1])
        set_start_value(F2[k], fix.F2[k+1])
        set_start_value(F7[k], fix.F7[k+1])
        set_start_value(F8[k], fix.F8[k+1])
        set_start_value(F9[k], fix.F9[k+1])
        set_start_value(F10[k], fix.F10[k+1])
        set_start_value(Fr1[k], fix.Fr1[k+1])
        set_start_value(Fr2[k], fix.Fr2[k+1])

    end

    @constraints mpc begin

        # Initial condition
        V3_inital, V1[0] == V1_init
        V4_inital, V4[0] == V4_init
        V5_inital, V5[0] == V5_init

        T3_inital, T1[0] == T1_init
        T4_inital, T4[0] == T4_init
        T5_inital, T5[0] == T5_init

        CA3_initial, CA1[0] == CA1_init
        CA4_initial, CA4[0] == CA4_init
        CA5_initial, CA5[0] == CA5_init

        CB3_initial, CB1[0] == CB1_init
        CB4_initial, CB4[0] == CB4_init
        CB5_initial, CB5[0] == CB5_init

        CC3_initial, CC1[0] == CC1_init
        CC4_initial, CC4[0] == CC4_init
        CC5_initial, CC5[0] == CC5_init

        CD3_initial, CD1[0] == CD1_init
        CD4_initial, CD4[0] == CD4_init
        CD5_initial, CD5[0] == CD5_init


        F1_hold[k in k_indices], F1[k] == F1[k+1]
        F2_hold[k in k_indices], F2[k] == F2[k+1]
        F7_hold[k in k_indices], F7[k] == F7[k+1]
        F8_hold[k in k_indices], F8[k] == F8[k+1]
        F9_hold[k in k_indices], F9[k] == F9[k+1]
        F10_hold[k in k_indices], F10[k] == F10[k+1]
        Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
        Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

        Q1_hold[k in k_indices], Q1[k] == Q1[k+1]
        Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
        Q5_hold[k in k_indices], Q5[k] == Q5[k+1]


    end

    @NLconstraints mpc begin

        # NLconstraints are the differential equations that describe the dynamics of the system
        dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + Fr2[k] - fix.F3[k+1]) * dt == V1[k+1]

        dT1_dt[k=0:N-1], T1[k] + (
            ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
              (Fr2[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - fix.F3[k+1] * CA1[k] * H_A(T1[k])) +
              (Fr2[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - fix.F3[k+1] * CB1[k] * H_B(T1[k])) +
              (Fr2[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - fix.F3[k+1] * CC1[k] * H_C(T1[k])) +
              (Fr2[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - fix.F3[k+1] * CD1[k] * H_D(T1[k])))
             /
             (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
            (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
            (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

        dCA1_dt[k=0:N-1], CA1[k] + (
            ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              fix.F3[k+1] * CA1[k])
             /
             V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

        dCB1_dt[k=0:N-1], CB1[k] + (
            ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              fix.F3[k+1] * CB1[k]) /
             V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

        dCC1_dt[k=0:N-1], CC1[k] + (
            ((Fr2[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              fix.F3[k+1] * CC1[k]) /
             V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

        dCD1_dt[k=0:N-1], CD1[k] + (
            ((Fr2[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]))
              -
              fix.F3[k+1] * CD1[k]) /
             V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

        dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4[k+1]

        dT4_dt[k=0:N-1], T4[k] +
                         ((Q4[k]
                           + (F7[k] * fix.CA3[k+1] * H_A(fix.T3[k+1]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
                           + (F7[k] * fix.CB3[k+1] * H_B(fix.T3[k+1]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
                           + (F7[k] * fix.CC3[k+1] * H_C(fix.T3[k+1]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
                           + (F7[k] * fix.CD3[k+1] * H_D(fix.T3[k+1]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
                          /
                          (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

        dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * fix.CA3[k+1] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
        dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * fix.CB3[k+1] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
        dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * fix.CC3[k+1] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
        dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * fix.CD3[k+1] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

        dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5[k+1]

        dT5_dt[k=0:N-1], T5[k] + (
            ((Q5[k] +
              F10[k] * CD0 * H_D(TD0)
              + (Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
              + (Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
              + (Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
              + (Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
             /
             (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
            +
            ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
             /
             (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

        dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
        dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
        dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
        dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]

        volHoldUp1[k in 0:dtd:(N-dtd)], F1[k] + F2[k] + Fr2[k] - fix.F3[k+1] == -(V1[k] - V1_sp) / tau
        volHoldUp4[k in 0:dtd:(N-dtd)], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        volHoldUp5[k in 0:dtd:(N-dtd)], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau

    end


    @NLobjective(mpc, Min, sum(
        w.v * (V1[k] - V1_sp)^2 + w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
        w.t1 * (T1[k] - T1_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
        w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
        w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
        w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
        w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
        for k = 0:N) +
                           sum(
        w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 +
        w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
        w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
        w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
        w.q1 * (Q1[k] - Q1_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
        for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc3_solve_time = MOI.get(mpc, MOI.SolveTimeSec())
    F1 = Vector(JuMP.value.(F1))
    F2 = Vector(JuMP.value.(F2))
    F7 = Vector(JuMP.value.(F7))
    F8 = Vector(JuMP.value.(F8))
    F9 = Vector(JuMP.value.(F9))
    F10 = Vector(JuMP.value.(F10))
    Fr1 = Vector(JuMP.value.(Fr1))
    Fr2 = Vector(JuMP.value.(Fr2))

    Q1 = Vector(JuMP.value.(Q1))
    Q4 = Vector(JuMP.value.(Q4))
    Q5 = Vector(JuMP.value.(Q5))

    V1 = Vector(JuMP.value.(V1))
    V4 = Vector(JuMP.value.(V4))
    V5 = Vector(JuMP.value.(V5))

    T1 = Vector(JuMP.value.(T1))
    T4 = Vector(JuMP.value.(T4))
    T5 = Vector(JuMP.value.(T5))

    CA1 = Vector(JuMP.value.(CA1))
    CA4 = Vector(JuMP.value.(CA4))
    CA5 = Vector(JuMP.value.(CA5))

    CB1 = Vector(JuMP.value.(CB1))
    CB4 = Vector(JuMP.value.(CB4))
    CB5 = Vector(JuMP.value.(CB5))

    CC1 = Vector(JuMP.value.(CC1))
    CC4 = Vector(JuMP.value.(CC4))
    CC5 = Vector(JuMP.value.(CC5))

    CD1 = Vector(JuMP.value.(CD1))
    CD4 = Vector(JuMP.value.(CD4))
    CD5 = Vector(JuMP.value.(CD5))

    return F1, F2, F7, F8, F9, F10, Fr1, Fr2, Q1, Q4, Q5, V1, V4, V5, T1, T4, T5, CA1, CA4, CA5, CB1, CB4, CB5, CC1, CC4, CC5, CD1, CD4, CD5
end

function dmpc_1_decomp4()

    # Control variables: F1, F2, F3, F4, Q1, Q2
    # State variables: All state variables associated with CSTR-1 and CSTR-2

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin

        # State variables

        # Volume, state, [=] m3
        V1[k=0:N], (lower_bound=min(V1_init, V1_sp) * 0.2, upper_bound=max(V1_init, V1_sp) * 1.8)

        # Temperature, state, [=] K
        T1[k=0:N], (lower_bound=min(T1_init, T1_sp) * 0.2, upper_bound=max(T1_init, T1_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA1[k=0:N], (lower_bound=min(CA1_init, CA1_sp) * 0.2, upper_bound=max(CA1_init, CA1_sp) * 1.8)

        CB1[k=0:N], (lower_bound=min(CB1_init, CB1_sp) * 0.2, upper_bound=max(CB1_init, CB1_sp) * 1.8)

        CC1[k=0:N], (lower_bound=min(CC1_init, CC1_sp) * 0.2, upper_bound=max(CC1_init, CC1_sp) * 1.8)

        CD1[k=0:N], (lower_bound=min(CD1_init, CD1_sp) * 0.2, upper_bound=max(CD1_init, CD1_sp) * 1.8)


        F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
        F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)

        Q1[k=0:N], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)

    end

    for k = 0:N
        set_start_value(V1[k], fix.V1[k+1])

        set_start_value(T1[k], fix.T1[k+1])

        set_start_value(CA1[k], fix.CA1[k+1])

        set_start_value(CB1[k], fix.CB1[k+1])

        set_start_value(CC1[k], fix.CC1[k+1])

        set_start_value(CD1[k], fix.CD1[k+1])

        set_start_value(F1[k], fix.F1[k+1])
        set_start_value(F2[k], fix.F2[k+1])

        set_start_value(Q1[k], fix.Q1[k+1])
    end

    @constraints mpc begin

        # Initial condition
        V1_inital, V1[0] == V1_init

        T1_inital, T1[0] == T1_init

        CA1_initial, CA1[0] == CA1_init

        CB1_initial, CB1[0] == CB1_init

        CC1_initial, CC1[0] == CC1_init

        CD1_initial, CD1[0] == CD1_init

        F1_hold[k in k_indices], F1[k] == F1[k+1]
        F2_hold[k in k_indices], F2[k] == F2[k+1]

        Q1_hold[k in k_indices], Q1[k] == Q1[k+1]


    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system
        dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + fix.Fr2[k+1] - fix.F3[k+1]) * dt == V1[k+1]

        dT1_dt[k=0:N-1], T1[k] + (
            ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
              (fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_A(fix.T4[k+1]) - fix.F3[k+1] * CA1[k] * H_A(T1[k])) +
              (fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_B(fix.T4[k+1]) - fix.F3[k+1] * CB1[k] * H_B(T1[k])) +
              (fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_C(fix.T4[k+1]) - fix.F3[k+1] * CC1[k] * H_C(T1[k])) +
              (fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_D(fix.T4[k+1]) - fix.F3[k+1] * CD1[k] * H_D(T1[k])))
             /
             (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
            (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
            (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

        dCA1_dt[k=0:N-1], CA1[k] + (
            ((F1[k] * CA0 + fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CA1[k])
             /
             V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

        dCB1_dt[k=0:N-1], CB1[k] + (
            ((F2[k] * CB0 + fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CB1[k]) /
             V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

        dCC1_dt[k=0:N-1], CC1[k] + (
            ((fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CC1[k]) /
             V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

        dCD1_dt[k=0:N-1], CD1[k] + (
            ((fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              fix.F3[k+1] * CD1[k]) /
             V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

        volHoldUp1[k in 0:dtd:(N-dtd)], F1[k] + F2[k] + fix.Fr2[k+1] - fix.F3[k+1] == -(V1[k] - V1_sp) / tau

    end



    @NLobjective(mpc, Min, sum(
        w.v * (V1[k] - V1_sp)^2 +
        w.t1 * (T1[k] - T1_sp)^2 +
        w.ca1 * (CA1[k] - CA1_sp)^2 +
        w.cb1 * (CB1[k] - CB1_sp)^2 +
        w.cc1 * (CC1[k] - CC1_sp)^2 +
        w.cd1 * (CD1[k] - CD1_sp)^2
        for k = 0:N) + sum(
        w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 +
        w.q1 * (Q1[k] - Q1_sp)^2 for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)
    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc1_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

    F1 = Vector(JuMP.value.(F1))
    F2 = Vector(JuMP.value.(F2))
    Q1 = Vector(JuMP.value.(Q1))

    V1 = Vector(JuMP.value.(V1))

    T1 = Vector(JuMP.value.(T1))

    CA1 = Vector(JuMP.value.(CA1))

    CB1 = Vector(JuMP.value.(CB1))

    CC1 = Vector(JuMP.value.(CC1))

    CD1 = Vector(JuMP.value.(CD1))

    return F1, F2, Q1, V1, T1, CA1, CB1, CC1, CD1
end

function dmpc_2_decomp4()

    # Control variables: F5, F6, Q3
    # State variables: All state variables associated with CSTR-3

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # State variables

        # Volume, state, [=] m3    
        V2[k=0:N], (lower_bound=min(V2_init, V2_sp) * 0.2, upper_bound=max(V2_init, V2_sp) * 1.8)
        V3[k=0:N], (lower_bound=min(V3_init, V3_sp) * 0.2, upper_bound=max(V3_init, V3_sp) * 1.8)

        # Temperature, state, [=] K
        T2[k=0:N], (lower_bound=min(T2_init, T2_sp) * 0.2, upper_bound=max(T2_init, T2_sp) * 1.8)
        T3[k=0:N], (lower_bound=min(T3_init, T3_sp) * 0.2, upper_bound=max(T3_init, T3_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA2[k=0:N], (lower_bound=min(CA2_init, CA2_sp) * 0.2, upper_bound=max(CA2_init, CA2_sp) * 1.8)
        CA3[k=0:N], (lower_bound=min(CA3_init, CA3_sp) * 0.2, upper_bound=max(CA3_init, CA3_sp) * 1.8)

        CB2[k=0:N], (lower_bound=min(CB2_init, CB2_sp) * 0.2, upper_bound=max(CB2_init, CB2_sp) * 1.8)
        CB3[k=0:N], (lower_bound=min(CB3_init, CB3_sp) * 0.2, upper_bound=max(CB3_init, CB3_sp) * 1.8)

        CC2[k=0:N], (lower_bound=min(CC2_init, CC2_sp) * 0.2, upper_bound=max(CC2_init, CC2_sp) * 1.8)
        CC3[k=0:N], (lower_bound=min(CC3_init, CC3_sp) * 0.2, upper_bound=max(CC3_init, CC3_sp) * 1.8)

        CD2[k=0:N], (lower_bound=min(CD2_init, CD2_sp) * 0.2, upper_bound=max(CD2_init, CD2_sp) * 1.8)
        CD3[k=0:N], (lower_bound=min(CD3_init, CD3_sp) * 0.2, upper_bound=max(CD3_init, CD3_sp) * 1.8)

        F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
        F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)
        F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
        F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)

        Q2[k=0:N], (start=Q2_sp, lower_bound=1.2 * Q2_sp, upper_bound=0.8 * Q2_sp)
        Q3[k=0:N], (start=Q3_sp, lower_bound=1.2 * Q3_sp, upper_bound=0.8 * Q3_sp)


    end


    for k = 0:N
        set_start_value(V2[k], fix.V2[k+1])
        set_start_value(V3[k], fix.V3[k+1])

        set_start_value(T2[k], fix.T2[k+1])
        set_start_value(T3[k], fix.T3[k+1])

        set_start_value(CA2[k], fix.CA2[k+1])
        set_start_value(CA3[k], fix.CA3[k+1])

        set_start_value(CB2[k], fix.CB2[k+1])
        set_start_value(CB3[k], fix.CB3[k+1])

        set_start_value(CC2[k], fix.CC2[k+1])
        set_start_value(CC3[k], fix.CC3[k+1])

        set_start_value(CD2[k], fix.CD2[k+1])
        set_start_value(CD3[k], fix.CD3[k+1])

        set_start_value(F3[k], fix.F3[k+1])
        set_start_value(F4[k], fix.F4[k+1])
        set_start_value(F5[k], fix.F5[k+1])
        set_start_value(F6[k], fix.F6[k+1])
    end

    @constraints mpc begin
        # Initial condition
        V2_inital, V2[0] == V2_init
        V3_inital, V3[0] == V3_init

        T2_inital, T2[0] == T2_init
        T3_inital, T3[0] == T3_init

        CA2_initial, CA2[0] == CA2_init
        CA3_initial, CA3[0] == CA3_init

        CB2_initial, CB2[0] == CB2_init
        CB3_initial, CB3[0] == CB3_init

        CC2_initial, CC2[0] == CC2_init
        CC3_initial, CC3[0] == CC3_init

        CD2_initial, CD2[0] == CD2_init
        CD3_initial, CD3[0] == CD3_init

        F3_hold[k in k_indices], F3[k] == F3[k+1]
        F4_hold[k in k_indices], F4[k] == F4[k+1]
        F5_hold[k in k_indices], F5[k] == F5[k+1]
        F6_hold[k in k_indices], F6[k] == F6[k+1]

        Q2_hold[k in k_indices], Q2[k] == Q2[k+1]
        Q3_hold[k in k_indices], Q3[k] == Q3[k+1]

    end

    @NLconstraints mpc begin

        dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - F5[k]) * dt == V2[k+1]

        dT2_dt[k=0:N-1], T2[k] + (
            (Q2[k] + F4[k] * CB0 * H_B(TB0) +
             (F3[k] * fix.CA1[k+1] * H_A(fix.T1[k+1]) - F5[k] * CA2[k] * H_A(T2[k])) +
             (F3[k] * fix.CB1[k+1] * H_B(fix.T1[k+1]) - F5[k] * CB2[k] * H_B(T2[k])) +
             (F3[k] * fix.CC1[k+1] * H_C(fix.T1[k+1]) - F5[k] * CC2[k] * H_C(T2[k])) +
             (F3[k] * fix.CD1[k+1] * H_D(fix.T1[k+1]) - F5[k] * CD2[k] * H_D(T2[k])))
            /
            (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
            +
            (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
            /
            (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


        dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * fix.CA1[k+1] - F5[k] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
        dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * fix.CB1[k+1] + F4[k] * CB0 - F5[k] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
        dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * fix.CC1[k+1] - F5[k] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
        dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * fix.CD1[k+1] - F5[k] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]

        dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - fix.F7[k+1]) * dt == V3[k+1]


        dT3_dt[k=0:N-1], T3[k] + (
            ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2[k] * H_A(T2[k]) - fix.F7[k+1] * CA3[k] * H_A(T3[k])) +
              (F5[k] * CB2[k] * H_B(T2[k]) - fix.F7[k+1] * CB3[k] * H_B(T3[k])) +
              (F5[k] * CC2[k] * H_C(T2[k]) - fix.F7[k+1] * CC3[k] * H_C(T3[k])) +
              (F5[k] * CD2[k] * H_D(T2[k]) - fix.F7[k+1] * CD3[k] * H_D(T3[k])))
             /
             (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
            +
            (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
            /
            (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

        dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * CA2[k] - fix.F7[k+1] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
        dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * CB2[k] + F6[k] * CB0 - fix.F7[k+1] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
        dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * CC2[k] - fix.F7[k+1] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
        dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * CD2[k] - fix.F7[k+1] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

        volHoldUp2[k in 0:dtd:(N-dtd)], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / tau
        volHoldUp3[k in 0:dtd:(N-dtd)], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / tau

    end


    @NLobjective(
        mpc,
        Min,
        sum(
            w.v * (V2[k] - V2_sp)^2 + w.v * (V3[k] - V3_sp)^2 +
            w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 +
            w.ca3 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 +
            w.cb3 * (CB2[k] - CB2_sp)^2 + w.cb3 * (CB3[k] - CB3_sp)^2 +
            w.cc2 * (CC2[k] - CC2_sp)^2 + w.cc3 * (CC3[k] - CC3_sp)^2 +
            w.cd2 * (CD2[k] - CD2_sp)^2 + w.cd3 * (CD3[k] - CD3_sp)^2
            for k = 0:N) +
        sum(
            w.f3 * (F3[k] - F3_sp)^2 +
            w.f4 * (F4[k] - F4_sp)^2 +
            w.f5 * (F5[k] - F5_sp)^2 +
            w.f6 * (F6[k] - F6_sp)^2 +
            w.q2 * (Q2[k] - Q2_sp)^2 +
            w.q3 * (Q3[k] - Q3_sp)^2
            for k = 0:N
        )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

    F3 = Vector(JuMP.value.(F3))
    F4 = Vector(JuMP.value.(F4))
    F5 = Vector(JuMP.value.(F5))
    F6 = Vector(JuMP.value.(F6))
    Q2 = Vector(JuMP.value.(Q2))
    Q3 = Vector(JuMP.value.(Q3))

    V2 = Vector(JuMP.value.(V2))
    V3 = Vector(JuMP.value.(V3))

    T2 = Vector(JuMP.value.(T2))
    T3 = Vector(JuMP.value.(T3))

    CA2 = Vector(JuMP.value.(CA2))
    CA3 = Vector(JuMP.value.(CA3))

    CB2 = Vector(JuMP.value.(CB2))
    CB3 = Vector(JuMP.value.(CB3))

    CC2 = Vector(JuMP.value.(CC2))
    CC3 = Vector(JuMP.value.(CC3))

    CD2 = Vector(JuMP.value.(CD2))
    CD3 = Vector(JuMP.value.(CD3))

    return F3, F4, F5, F6, Q2, Q3, V2, T2, V3, T3, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3

end

function dmpc_3_decomp4()

    mpc = JuMP.Model(Ipopt.Optimizer)

    # Register functions, supressing warnings
    register(mpc, :H_A, 1, H_A; autodiff=true)
    register(mpc, :H_B, 1, H_B; autodiff=true)
    register(mpc, :H_C, 1, H_C; autodiff=true)
    register(mpc, :H_D, 1, H_D; autodiff=true)
    register(mpc, :r1, 3, r1; autodiff=true)
    register(mpc, :r2, 4, r2; autodiff=true)
    register(mpc, :r3, 3, r3; autodiff=true)
    register(mpc, :CAr, 4, CAr; autodiff=true)
    register(mpc, :CBr, 4, CBr; autodiff=true)
    register(mpc, :CCr, 4, CCr; autodiff=true)
    register(mpc, :CDr, 4, CDr; autodiff=true)
    register(mpc, :MA, 13, MA; autodiff=true)
    register(mpc, :MB, 13, MB; autodiff=true)
    register(mpc, :MC, 13, MC; autodiff=true)
    register(mpc, :MD, 13, MD; autodiff=true)

    JuMP.@variables mpc begin
        # Volume, state, [=] m3
        V4[k=0:N], (lower_bound=min(V4_init, V4_sp) * 0.2, upper_bound=max(V4_init, V4_sp) * 1.8)
        V5[k=0:N], (lower_bound=min(V5_init, V5_sp) * 0.2, upper_bound=max(V5_init, V5_sp) * 1.8)

        # Temperature, state, [=] K
        T4[k=0:N], (lower_bound=min(T4_init, T4_sp) * 0.2, upper_bound=max(T4_init, T4_sp) * 1.8)
        T5[k=0:N], (lower_bound=min(T5_init, T5_sp) * 0.2, upper_bound=max(T5_init, T5_sp) * 1.8)

        # Concentration, state, [=] mol/m3
        CA4[k=0:N], (lower_bound=min(CA4_init, CA4_sp) * 0.2, upper_bound=max(CA4_init, CA4_sp) * 1.8)
        CA5[k=0:N], (lower_bound=min(CA5_init, CA5_sp) * 0.2, upper_bound=max(CA5_init, CA5_sp) * 1.8)

        CB4[k=0:N], (lower_bound=min(CB4_init, CB4_sp) * 0.2, upper_bound=max(CB4_init, CB4_sp) * 1.8)
        CB5[k=0:N], (lower_bound=min(CB5_init, CB5_sp) * 0.2, upper_bound=max(CB5_init, CB5_sp) * 1.8)

        CC4[k=0:N], (lower_bound=min(CC4_init, CC4_sp) * 0.2, upper_bound=max(CC4_init, CC4_sp) * 1.8)
        CC5[k=0:N], (lower_bound=min(CC5_init, CC5_sp) * 0.2, upper_bound=max(CC5_init, CC5_sp) * 1.8)

        CD4[k=0:N], (lower_bound=min(CD4_init, CD4_sp) * 0.2, upper_bound=max(CD4_init, CD4_sp) * 1.8)
        CD5[k=0:N], (lower_bound=min(CD5_init, CD5_sp) * 0.2, upper_bound=max(CD5_init, CD5_sp) * 1.8)

        # Control variables
        # Flow, control [=] m3/s
        F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
        F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
        F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
        F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
        Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
        Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)

        Q4[k=0:N], (start=Q4_sp, upper_bound=1.2 * Q4_sp, lower_bound=0.8 * Q4_sp)
        Q5[k=0:N], (start=Q5_sp, upper_bound=1.2 * Q5_sp, lower_bound=0.8 * Q5_sp)
    end

    for k = 0:N
        JuMP.fix(F7[k], F7_sp; force=true)
        # JuMP.fix(F8[k], F8_sp; force=true)
        # JuMP.fix(F9[k], F9_sp; force=true)
        # JuMP.fix(F10[k], F10_sp; force=true)
        # JuMP.fix(Fr1[k], Fr1_sp; force=true)
        JuMP.fix(Fr2[k], Fr2_sp; force=true)
    end

    for k = 0:N
        # JuMP.fix(Q1[k], Q1_sp; force=true)
        # JuMP.fix(Q2[k], Q2_sp; force=true)
        # JuMP.fix(Q3[k], Q3_sp; force=true)
        # JuMP.fix(Q4[k], Q4_sp; force=true)
        # JuMP.fix(Q5[k], Q5_sp; force=true)
    end

    for k = 0:N
        set_start_value(V4[k], fix.V4[k+1])
        set_start_value(V5[k], fix.V5[k+1])

        set_start_value(T4[k], fix.T4[k+1])
        set_start_value(T5[k], fix.T5[k+1])

        set_start_value(CA4[k], fix.CA4[k+1])
        set_start_value(CA5[k], fix.CA5[k+1])

        set_start_value(CB4[k], fix.CB4[k+1])
        set_start_value(CB5[k], fix.CB5[k+1])

        set_start_value(CC4[k], fix.CC4[k+1])
        set_start_value(CC5[k], fix.CC5[k+1])

        set_start_value(CD4[k], fix.CD4[k+1])
        set_start_value(CD5[k], fix.CD5[k+1])

        set_start_value(F7[k], fix.F7[k+1])
        set_start_value(F8[k], fix.F8[k+1])
        set_start_value(F9[k], fix.F9[k+1])
        set_start_value(F10[k], fix.F10[k+1])
        set_start_value(Fr1[k], fix.Fr1[k+1])
        set_start_value(Fr2[k], fix.Fr2[k+1])

    end

    @constraints mpc begin

        # Initial condition
        V4_inital, V4[0] == V4_init
        V5_inital, V5[0] == V5_init

        T4_inital, T4[0] == T4_init
        T5_inital, T5[0] == T5_init

        CA4_initial, CA4[0] == CA4_init
        CA5_initial, CA5[0] == CA5_init

        CB4_initial, CB4[0] == CB4_init
        CB5_initial, CB5[0] == CB5_init

        CC4_initial, CC4[0] == CC4_init
        CC5_initial, CC5[0] == CC5_init

        CD4_initial, CD4[0] == CD4_init
        CD5_initial, CD5[0] == CD5_init

        # F1_hold[k in k_indices], F1[k] == F1[k+1]
        # F2_hold[k in k_indices], F2[k] == F2[k+1]
        # F3_hold[k in k_indices], F3[k] == F3[k+1]
        # F4_hold[k in k_indices], F4[k] == F4[k+1]
        # F5_hold[k in k_indices], F5[k] == F5[k+1]
        # F6_hold[k in k_indices], F6[k] == F6[k+1]
        F7_hold[k in k_indices], F7[k] == F7[k+1]
        F8_hold[k in k_indices], F8[k] == F8[k+1]
        F9_hold[k in k_indices], F9[k] == F9[k+1]
        F10_hold[k in k_indices], F10[k] == F10[k+1]
        Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
        Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

        # Q1_hold[k in k_indices], Q1[k] == Q1[k+1]
        # Q2_hold[k in k_indices], Q2[k] == Q2[k+1]
        # Q3_hold[k in k_indices], Q3[k] == Q3[k+1]
        Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
        Q5_hold[k in k_indices], Q5[k] == Q5[k+1]

        # volDec4[k=0:N-1], (V4[k+1] - V4_sp) <= 0.8 * (V4[k] - V4_sp)
        # volDec5[k=0:N-1], (V5[k+1] - V5_sp) <= 0.8 * (V5[k] - V5_sp)

    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system

        dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4[k+1]
        dT4_dt[k=0:N-1], T4[k] +
                         ((Q4[k]
                           + (F7[k] * fix.CA3[k+1] * H_A(fix.T3[k+1]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
                           + (F7[k] * fix.CB3[k+1] * H_B(fix.T3[k+1]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
                           + (F7[k] * fix.CC3[k+1] * H_C(fix.T3[k+1]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
                           + (F7[k] * fix.CD3[k+1] * H_D(fix.T3[k+1]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
                          /
                          (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

        dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * fix.CA3[k+1] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
        dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * fix.CB3[k+1] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
        dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * fix.CC3[k+1] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
        dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * fix.CD3[k+1] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

        dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5[k+1]

        dT5_dt[k=0:N-1], T5[k] + (
            ((Q5[k] +
              F10[k] * CD0 * H_D(TD0)
              + (Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
              + (Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
              + (Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
              + (Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
             /
             (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
            +
            ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
             /
             (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

        dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
        dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
        dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
        dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]


        volHoldUp4[k in 0:dtd:(N-dtd)], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        volHoldUp5[k in 0:dtd:(N-dtd)], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau


    end


    @NLobjective(mpc, Min, sum(
        w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
        w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
        w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
        w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
        w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
        w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
        for k = 0:N) +
                           sum(
        w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
        w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
        w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
        w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
        for k = 0:N
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    set_optimizer_attribute(mpc, "print_frequency_iter", 300)

    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    # set_silent(mpc)

    optimize!(mpc)

    global dmpc3_solve_time = MOI.get(mpc, MOI.SolveTimeSec())
    F7 = Vector(JuMP.value.(F7))
    F8 = Vector(JuMP.value.(F8))
    F9 = Vector(JuMP.value.(F9))
    F10 = Vector(JuMP.value.(F10))
    Fr1 = Vector(JuMP.value.(Fr1))
    Fr2 = Vector(JuMP.value.(Fr2))

    Q4 = Vector(JuMP.value.(Q4))
    Q5 = Vector(JuMP.value.(Q5))

    V4 = Vector(JuMP.value.(V4))
    V5 = Vector(JuMP.value.(V5))

    T4 = Vector(JuMP.value.(T4))
    T5 = Vector(JuMP.value.(T5))

    CA4 = Vector(JuMP.value.(CA4))
    CA5 = Vector(JuMP.value.(CA5))

    CB4 = Vector(JuMP.value.(CB4))
    CB5 = Vector(JuMP.value.(CB5))

    CC4 = Vector(JuMP.value.(CC4))
    CC5 = Vector(JuMP.value.(CC5))

    CD4 = Vector(JuMP.value.(CD4))
    CD5 = Vector(JuMP.value.(CD5))

    return F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5
end

function dmpc(arc)

    max_steps = 10
    global dmpc_solve_time = 0
    for steps = 1:max_steps

        # Structural Coupling Architecture:
        if arc == 1
            F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2 = dmpc_1_decomp1()
            F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3 = dmpc_2_decomp1()
            F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5 = dmpc_3_decomp1()
        elseif arc == 2
            # # Intuition Architecture 1:
            F1, F2, Q1, V1, T1, CA1, CB1, CC1, CD1 = dmpc_1_decomp2()
            F3, F4, Q2, V2, T2, CA2, CB2, CC2, CD2 = dmpc_2_decomp2()
            F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q3, Q4, Q5, V3, V4, V5, T3, T4, T5, CA3, CA4, CA5, CB3, CB4, CB5, CC3, CC4, CC5, CD3, CD4, CD5 = dmpc_3_decomp2()
        elseif arc == 3
            # Intuition Architecture 2:
            F3, F4, Q2, V2, T2, CA2, CB2, CC2, CD2 = dmpc_1_decomp3()
            F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3 = dmpc_2_decomp3()
            F1, F2, F7, F8, F9, F10, Fr1, Fr2, Q1, Q4, Q5, V1, V4, V5, T1, T4, T5, CA1, CA4, CA5, CB1, CB4, CB5, CC1, CC4, CC5, CD1, CD4, CD5 = dmpc_3_decomp3()
        elseif arc == 4
            # Intuition Architecture 3:
            F1, F2, Q1, V1, T1, CA1, CB1, CC1, CD1 = dmpc_1_decomp4()
            F3, F4, F5, F6, Q2, Q3, V2, T2, V3, T3, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3 = dmpc_2_decomp4()
            F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5 = dmpc_3_decomp4()
        end


        if steps >= 2

            global dmpc_solve_time = dmpc_solve_time + max(dmpc1_solve_time, dmpc2_solve_time, dmpc3_solve_time)

            if dmpc_solve_time > c_cpu_max
                break
            end

        end

        fix.F1 = F1
        fix.F2 = F2
        fix.F3 = F3
        fix.F4 = F4
        fix.F5 = F5
        fix.F6 = F6
        fix.F7 = F7
        fix.F8 = F8
        fix.F9 = F9
        fix.F10 = F10
        fix.Fr1 = Fr1
        fix.Fr2 = Fr2

        fix.Q1 = Q1
        fix.Q2 = Q2
        fix.Q3 = Q3
        fix.Q4 = Q4
        fix.Q5 = Q5

        if steps == 1

            global dmpc_solve_time = dmpc_solve_time + max(dmpc1_solve_time, dmpc2_solve_time, dmpc3_solve_time)

            if dmpc_solve_time > c_cpu_max
                break
            end

        end

        fix.T1 = T1
        fix.T2 = T2
        fix.T3 = T3
        fix.T4 = T4
        fix.T5 = T5

        fix.V1 = V1
        fix.V2 = V2
        fix.V3 = V3
        fix.V4 = V4
        fix.V5 = V5

        fix.CA1 = CA1
        fix.CA2 = CA2
        fix.CA3 = CA3
        fix.CA4 = CA4
        fix.CA5 = CA5
        fix.CB1 = CB1
        fix.CB2 = CB2
        fix.CB3 = CB3
        fix.CB4 = CB4
        fix.CB5 = CB5
        fix.CC1 = CC1
        fix.CC2 = CC2
        fix.CC3 = CC3
        fix.CC4 = CC4
        fix.CC5 = CC5
        fix.CD1 = CD1
        fix.CD2 = CD2
        fix.CD3 = CD3
        fix.CD4 = CD4
        fix.CD5 = CD5
    end

    F1 = fix.F1
    F2 = fix.F2
    F3 = fix.F3
    F4 = fix.F4
    F5 = fix.F5
    F6 = fix.F6
    F7 = fix.F7
    F8 = fix.F8
    F9 = fix.F9
    F10 = fix.F10
    Fr1 = fix.Fr1
    Fr2 = fix.Fr2
    Q1 = fix.Q1
    Q2 = fix.Q2
    Q3 = fix.Q3
    Q4 = fix.Q4
    Q5 = fix.Q5



    return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5


end

function getTraj(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5)

    V1_vec = zeros(N + 1)
    V2_vec = zeros(N + 1)
    V3_vec = zeros(N + 1)
    V4_vec = zeros(N + 1)
    V5_vec = zeros(N + 1)
    T1_vec = zeros(N + 1)
    T2_vec = zeros(N + 1)
    T3_vec = zeros(N + 1)
    T4_vec = zeros(N + 1)
    T5_vec = zeros(N + 1)

    CA1_vec = zeros(N + 1)
    CA2_vec = zeros(N + 1)
    CA3_vec = zeros(N + 1)
    CA4_vec = zeros(N + 1)
    CA5_vec = zeros(N + 1)
    CB1_vec = zeros(N + 1)
    CB2_vec = zeros(N + 1)
    CB3_vec = zeros(N + 1)
    CB4_vec = zeros(N + 1)
    CB5_vec = zeros(N + 1)
    CC1_vec = zeros(N + 1)
    CC2_vec = zeros(N + 1)
    CC3_vec = zeros(N + 1)
    CC4_vec = zeros(N + 1)
    CC5_vec = zeros(N + 1)
    CD1_vec = zeros(N + 1)
    CD2_vec = zeros(N + 1)
    CD3_vec = zeros(N + 1)
    CD4_vec = zeros(N + 1)
    CD5_vec = zeros(N + 1)



    V1_vec[1] = V1_init
    V2_vec[1] = V2_init
    V3_vec[1] = V3_init
    V4_vec[1] = V4_init
    V5_vec[1] = V5_init
    T1_vec[1] = T1_init
    T2_vec[1] = T2_init
    T3_vec[1] = T3_init
    T4_vec[1] = T4_init
    T5_vec[1] = T5_init

    CA1_vec[1] = CA1_init
    CA2_vec[1] = CA2_init
    CA3_vec[1] = CA3_init
    CA4_vec[1] = CA4_init
    CA5_vec[1] = CA5_init
    CB1_vec[1] = CB1_init
    CB2_vec[1] = CB2_init
    CB3_vec[1] = CB3_init
    CB4_vec[1] = CB4_init
    CB5_vec[1] = CB5_init
    CC1_vec[1] = CC1_init
    CC2_vec[1] = CC2_init
    CC3_vec[1] = CC3_init
    CC4_vec[1] = CC4_init
    CC5_vec[1] = CC5_init
    CD1_vec[1] = CD1_init
    CD2_vec[1] = CD2_init
    CD3_vec[1] = CD3_init
    CD4_vec[1] = CD4_init
    CD5_vec[1] = CD5_init

    for j = 1:N
        global k = j

        V1 = V1_vec[k]
        V2 = V2_vec[k]
        V3 = V3_vec[k]
        V4 = V4_vec[k]
        V5 = V5_vec[k]

        T1 = T1_vec[k]
        T2 = T2_vec[k]
        T3 = T3_vec[k]
        T4 = T4_vec[k]
        T5 = T5_vec[k]

        CA1 = CA1_vec[k]
        CB1 = CB1_vec[k]
        CC1 = CC1_vec[k]
        CD1 = CD1_vec[k]

        CA2 = CA2_vec[k]
        CB2 = CB2_vec[k]
        CC2 = CC2_vec[k]
        CD2 = CD2_vec[k]

        CA3 = CA3_vec[k]
        CB3 = CB3_vec[k]
        CC3 = CC3_vec[k]
        CD3 = CD3_vec[k]

        CA4 = CA4_vec[k]
        CB4 = CB4_vec[k]
        CC4 = CC4_vec[k]
        CD4 = CD4_vec[k]

        CA5 = CA5_vec[k]
        CB5 = CB5_vec[k]
        CC5 = CC5_vec[k]
        CD5 = CD5_vec[k]

        x0 = [V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5]

        # Define the ODE function
        function f(y, p, t)
            # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
            V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

            dV1 = F1[k] + F2[k] + Fr2[k] - F3[k]
            dV2 = F3[k] + F4[k] - F5[k]
            dV3 = F5[k] + F6[k] - F7[k]
            dV4 = F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]
            dV5 = F10[k] + Fr1[k] - F9[k]

            dT1 = (
                ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
                  (Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3[k] * CA1 * H_A(T1)) +
                  (Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3[k] * CB1 * H_B(T1)) +
                  (Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3[k] * CC1 * H_C(T1)) +
                  (Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3[k] * CD1 * H_D(T1)))
                 /
                 (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
                (-delH_r1 * r1_(T1, CA1, CB1) - delH_r2 * r2_(T1, CB1, CC1, CD1)) /
                (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

            dT2 = (
                (Q2[k] + F4[k] * CB0 * H_B(TB0) +
                 (F3[k] * CA1 * H_A(T1) - F5[k] * CA2 * H_A(T2)) +
                 (F3[k] * CB1 * H_B(T1) - F5[k] * CB2 * H_B(T2)) +
                 (F3[k] * CC1 * H_C(T1) - F5[k] * CC2 * H_C(T2)) +
                 (F3[k] * CD1 * H_D(T1) - F5[k] * CD2 * H_D(T2)))
                /
                (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
                +
                (-delH_r1 * r1_(T2, CA2, CB2) - delH_r2 * r2_(T2, CB2, CC2, CD2))
                /
                (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


            dT3 = (
                ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2 * H_A(T2) - F7[k] * CA3 * H_A(T3)) +
                  (F5[k] * CB2 * H_B(T2) - F7[k] * CB3 * H_B(T3)) +
                  (F5[k] * CC2 * H_C(T2) - F7[k] * CC3 * H_C(T3)) +
                  (F5[k] * CD2 * H_D(T2) - F7[k] * CD3 * H_D(T3)))
                 /
                 (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
                +
                (-delH_r1 * r1_(T3, CA3, CB3) - delH_r2 * r2_(T3, CB3, CC3, CD3))
                /
                (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

            dT4 = ((Q4[k]
                    + (F7[k] * CA3 * H_A(T3) + F9[k] * CA5 * H_A(T5) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8[k] * CA4 * H_A(T4) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
                    + (F7[k] * CB3 * H_B(T3) + F9[k] * CB5 * H_B(T5) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8[k] * CB4 * H_B(T4) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
                    + (F7[k] * CC3 * H_C(T3) + F9[k] * CC5 * H_C(T5) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8[k] * CC4 * H_C(T4) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
                    + (F7[k] * CD3 * H_D(T3) + F9[k] * CD5 * H_D(T5) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8[k] * CD4 * H_D(T4) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
                   /
                   (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
            dT5 = (
                ((Q5[k] +
                  F10[k] * CD0 * H_D(TD0)
                  + (Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9[k] * CA5 * H_A(T5))
                  + (Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9[k] * CB5 * H_B(T5))
                  + (Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9[k] * CC5 * H_C(T5))
                  + (Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9[k] * CD5 * H_D(T5)))
                 /
                 (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
                +
                ((-delH_r2 * r2_(T5, CB5, CC5, CD5) - delH_r3 * r3_(T5, CA5, CD5))
                 /
                 (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


            dCA1 = (
                ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CA1)
                 /
                 V1) - r1_(T1, CA1, CB1))

            dCB1 = (
                ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CB1) /
                 V1) - r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

            dCC1 = (
                ((Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CC1) /
                 V1) + r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

            dCD1 = (
                ((Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CD1) /
                 V1) + r2_(T1, CB1, CC1, CD1))

            dCA2 = (((F3[k] * CA1 - F5[k] * CA2) / V2) - r1_(T2, CA2, CB2))
            dCB2 = ((F3[k] * CB1 + F4[k] * CB0 - F5[k] * CB2) / V2 - r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
            dCC2 = ((F3[k] * CC1 - F5[k] * CC2) / V2 + r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
            dCD2 = ((F3[k] * CD1 - F5[k] * CD2) / V2 + r2_(T2, CB2, CC2, CD2))

            dCA3 = (((F5[k] * CA2 - F7[k] * CA3) / V3) - r1_(T3, CA3, CB3))
            dCB3 = (((F5[k] * CB2 + F6[k] * CB0 - F7[k] * CB3) / V3) - r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
            dCC3 = ((F5[k] * CC2 - F7[k] * CC3) / V3 + r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
            dCD3 = ((F5[k] * CD2 - F7[k] * CD3) / V3 + r2_(T3, CB3, CC3, CD3))

            dCA4 = ((F7[k] * CA3 + F9[k] * CA5 - (Fr1[k] + Fr2[k]) * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CA4) / V4)
            dCB4 = ((F7[k] * CB3 + F9[k] * CB5 - (Fr1[k] + Fr2[k]) * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CB4) / V4)
            dCC4 = ((F7[k] * CC3 + F9[k] * CC5 - (Fr1[k] + Fr2[k]) * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CC4) / V4)
            dCD4 = ((F7[k] * CD3 + F9[k] * CD5 - (Fr1[k] + Fr2[k]) * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CD4) / V4)

            dCA5 = ((Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CA5) / V5 - r3_(T5, CA5, CD5))
            dCB5 = ((Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CB5) / V5 - r2_(T5, CB5, CC5, CD5))
            dCC5 = ((Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CC5) / V5 - r2_(T5, CB5, CC5, CD5) + 2 * r3_(T5, CA5, CD5))
            dCD5 = ((Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10[k] * CD0 - F9[k] * CD5) / V5 + r2_(T5, CB5, CC5, CD5) - r3_(T5, CA5, CD5))

            return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
        end

        # Solve ODE 
        tspan = (0.0, dt)
        prob = ODEProblem(f, x0, tspan)
        soln = solve(prob, Rosenbrock23())

        # Obtain the next values for each element
        V1_vec[k+1] = max(1e-6, soln.u[end][1])
        V2_vec[k+1] = max(1e-6, soln.u[end][2])
        V3_vec[k+1] = max(1e-6, soln.u[end][3])
        V4_vec[k+1] = max(1e-6, soln.u[end][4])
        V5_vec[k+1] = max(1e-6, soln.u[end][5])

        T1_vec[k+1] = max(1e-6, soln.u[end][6])
        T2_vec[k+1] = max(1e-6, soln.u[end][7])
        T3_vec[k+1] = max(1e-6, soln.u[end][8])
        T4_vec[k+1] = max(1e-6, soln.u[end][9])
        T5_vec[k+1] = max(1e-6, soln.u[end][10])

        CA1_vec[k+1] = max(1e-6, soln.u[end][11])
        CB1_vec[k+1] = max(1e-6, soln.u[end][12])
        CC1_vec[k+1] = max(1e-6, soln.u[end][13])
        CD1_vec[k+1] = max(1e-6, soln.u[end][14])

        CA2_vec[k+1] = max(1e-6, soln.u[end][15])
        CB2_vec[k+1] = max(1e-6, soln.u[end][16])
        CC2_vec[k+1] = max(1e-6, soln.u[end][17])
        CD2_vec[k+1] = max(1e-6, soln.u[end][18])

        CA3_vec[k+1] = max(1e-6, soln.u[end][19])
        CB3_vec[k+1] = max(1e-6, soln.u[end][20])
        CC3_vec[k+1] = max(1e-6, soln.u[end][21])
        CD3_vec[k+1] = max(1e-6, soln.u[end][22])

        CA4_vec[k+1] = max(1e-6, soln.u[end][23])
        CB4_vec[k+1] = max(1e-6, soln.u[end][24])
        CC4_vec[k+1] = max(1e-6, soln.u[end][25])
        CD4_vec[k+1] = max(1e-6, soln.u[end][26])

        CA5_vec[k+1] = max(1e-6, soln.u[end][27])
        CB5_vec[k+1] = max(1e-6, soln.u[end][28])
        CC5_vec[k+1] = max(1e-6, soln.u[end][29])
        CD5_vec[k+1] = max(1e-6, soln.u[end][30])

    end

    V1 = V1_vec
    V2 = V2_vec
    V3 = V3_vec
    V4 = V4_vec
    V5 = V5_vec

    T1 = T1_vec
    T2 = T2_vec
    T3 = T3_vec
    T4 = T4_vec
    T5 = T5_vec

    CA1 = CA1_vec
    CB1 = CB1_vec
    CC1 = CC1_vec
    CD1 = CD1_vec

    CA2 = CA2_vec
    CB2 = CB2_vec
    CC2 = CC2_vec
    CD2 = CD2_vec

    CA3 = CA3_vec
    CB3 = CB3_vec
    CC3 = CC3_vec
    CD3 = CD3_vec

    CA4 = CA4_vec
    CB4 = CB4_vec
    CC4 = CC4_vec
    CD4 = CD4_vec

    CA5 = CA5_vec
    CB5 = CB5_vec
    CC5 = CC5_vec
    CD5 = CD5_vec

    return V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5
end

function getPI(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5)

    V1_vec = zeros(N + 1)
    V2_vec = zeros(N + 1)
    V3_vec = zeros(N + 1)
    V4_vec = zeros(N + 1)
    V5_vec = zeros(N + 1)
    T1_vec = zeros(N + 1)
    T2_vec = zeros(N + 1)
    T3_vec = zeros(N + 1)
    T4_vec = zeros(N + 1)
    T5_vec = zeros(N + 1)

    CA1_vec = zeros(N + 1)
    CA2_vec = zeros(N + 1)
    CA3_vec = zeros(N + 1)
    CA4_vec = zeros(N + 1)
    CA5_vec = zeros(N + 1)
    CB1_vec = zeros(N + 1)
    CB2_vec = zeros(N + 1)
    CB3_vec = zeros(N + 1)
    CB4_vec = zeros(N + 1)
    CB5_vec = zeros(N + 1)
    CC1_vec = zeros(N + 1)
    CC2_vec = zeros(N + 1)
    CC3_vec = zeros(N + 1)
    CC4_vec = zeros(N + 1)
    CC5_vec = zeros(N + 1)
    CD1_vec = zeros(N + 1)
    CD2_vec = zeros(N + 1)
    CD3_vec = zeros(N + 1)
    CD4_vec = zeros(N + 1)
    CD5_vec = zeros(N + 1)

    V1_vec[1] = V1_init
    V2_vec[1] = V2_init
    V3_vec[1] = V3_init
    V4_vec[1] = V4_init
    V5_vec[1] = V5_init
    T1_vec[1] = T1_init
    T2_vec[1] = T2_init
    T3_vec[1] = T3_init
    T4_vec[1] = T4_init
    T5_vec[1] = T5_init

    CA1_vec[1] = CA1_init
    CA2_vec[1] = CA2_init
    CA3_vec[1] = CA3_init
    CA4_vec[1] = CA4_init
    CA5_vec[1] = CA5_init
    CB1_vec[1] = CB1_init
    CB2_vec[1] = CB2_init
    CB3_vec[1] = CB3_init
    CB4_vec[1] = CB4_init
    CB5_vec[1] = CB5_init
    CC1_vec[1] = CC1_init
    CC2_vec[1] = CC2_init
    CC3_vec[1] = CC3_init
    CC4_vec[1] = CC4_init
    CC5_vec[1] = CC5_init
    CD1_vec[1] = CD1_init
    CD2_vec[1] = CD2_init
    CD3_vec[1] = CD3_init
    CD4_vec[1] = CD4_init
    CD5_vec[1] = CD5_init

    for j = 1:N
        global k = j

        V1 = V1_vec[k]
        V2 = V2_vec[k]
        V3 = V3_vec[k]
        V4 = V4_vec[k]
        V5 = V5_vec[k]

        T1 = T1_vec[k]
        T2 = T2_vec[k]
        T3 = T3_vec[k]
        T4 = T4_vec[k]
        T5 = T5_vec[k]

        CA1 = CA1_vec[k]
        CB1 = CB1_vec[k]
        CC1 = CC1_vec[k]
        CD1 = CD1_vec[k]

        CA2 = CA2_vec[k]
        CB2 = CB2_vec[k]
        CC2 = CC2_vec[k]
        CD2 = CD2_vec[k]

        CA3 = CA3_vec[k]
        CB3 = CB3_vec[k]
        CC3 = CC3_vec[k]
        CD3 = CD3_vec[k]

        CA4 = CA4_vec[k]
        CB4 = CB4_vec[k]
        CC4 = CC4_vec[k]
        CD4 = CD4_vec[k]

        CA5 = CA5_vec[k]
        CB5 = CB5_vec[k]
        CC5 = CC5_vec[k]
        CD5 = CD5_vec[k]

        x0 = [V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5]

        # Define the ODE function
        function f(y, p, t)
            # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
            V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

            dV1 = F1[k] + F2[k] + Fr2[k] - F3[k]
            dV2 = F3[k] + F4[k] - F5[k]
            dV3 = F5[k] + F6[k] - F7[k]
            dV4 = F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]
            dV5 = F10[k] + Fr1[k] - F9[k]

            dT1 = (
                ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
                  (Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3[k] * CA1 * H_A(T1)) +
                  (Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3[k] * CB1 * H_B(T1)) +
                  (Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3[k] * CC1 * H_C(T1)) +
                  (Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3[k] * CD1 * H_D(T1)))
                 /
                 (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
                (-delH_r1 * r1_(T1, CA1, CB1) - delH_r2 * r2_(T1, CB1, CC1, CD1)) /
                (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

            dT2 = (
                (Q2[k] + F4[k] * CB0 * H_B(TB0) +
                 (F3[k] * CA1 * H_A(T1) - F5[k] * CA2 * H_A(T2)) +
                 (F3[k] * CB1 * H_B(T1) - F5[k] * CB2 * H_B(T2)) +
                 (F3[k] * CC1 * H_C(T1) - F5[k] * CC2 * H_C(T2)) +
                 (F3[k] * CD1 * H_D(T1) - F5[k] * CD2 * H_D(T2)))
                /
                (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
                +
                (-delH_r1 * r1_(T2, CA2, CB2) - delH_r2 * r2_(T2, CB2, CC2, CD2))
                /
                (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


            dT3 = (
                ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2 * H_A(T2) - F7[k] * CA3 * H_A(T3)) +
                  (F5[k] * CB2 * H_B(T2) - F7[k] * CB3 * H_B(T3)) +
                  (F5[k] * CC2 * H_C(T2) - F7[k] * CC3 * H_C(T3)) +
                  (F5[k] * CD2 * H_D(T2) - F7[k] * CD3 * H_D(T3)))
                 /
                 (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
                +
                (-delH_r1 * r1_(T3, CA3, CB3) - delH_r2 * r2_(T3, CB3, CC3, CD3))
                /
                (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

            dT4 = ((Q4[k]
                    + (F7[k] * CA3 * H_A(T3) + F9[k] * CA5 * H_A(T5) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8[k] * CA4 * H_A(T4) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
                    + (F7[k] * CB3 * H_B(T3) + F9[k] * CB5 * H_B(T5) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8[k] * CB4 * H_B(T4) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
                    + (F7[k] * CC3 * H_C(T3) + F9[k] * CC5 * H_C(T5) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8[k] * CC4 * H_C(T4) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
                    + (F7[k] * CD3 * H_D(T3) + F9[k] * CD5 * H_D(T5) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8[k] * CD4 * H_D(T4) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
                   /
                   (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
            dT5 = (
                ((Q5[k] +
                  F10[k] * CD0 * H_D(TD0)
                  + (Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9[k] * CA5 * H_A(T5))
                  + (Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9[k] * CB5 * H_B(T5))
                  + (Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9[k] * CC5 * H_C(T5))
                  + (Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9[k] * CD5 * H_D(T5)))
                 /
                 (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
                +
                ((-delH_r2 * r2_(T5, CB5, CC5, CD5) - delH_r3 * r3_(T5, CA5, CD5))
                 /
                 (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


            dCA1 = (
                ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CA1)
                 /
                 V1) - r1_(T1, CA1, CB1))

            dCB1 = (
                ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CB1) /
                 V1) - r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

            dCC1 = (
                ((Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CC1) /
                 V1) + r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

            dCD1 = (
                ((Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                  -
                  F3[k] * CD1) /
                 V1) + r2_(T1, CB1, CC1, CD1))

            dCA2 = (((F3[k] * CA1 - F5[k] * CA2) / V2) - r1_(T2, CA2, CB2))
            dCB2 = ((F3[k] * CB1 + F4[k] * CB0 - F5[k] * CB2) / V2 - r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
            dCC2 = ((F3[k] * CC1 - F5[k] * CC2) / V2 + r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
            dCD2 = ((F3[k] * CD1 - F5[k] * CD2) / V2 + r2_(T2, CB2, CC2, CD2))

            dCA3 = (((F5[k] * CA2 - F7[k] * CA3) / V3) - r1_(T3, CA3, CB3))
            dCB3 = (((F5[k] * CB2 + F6[k] * CB0 - F7[k] * CB3) / V3) - r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
            dCC3 = ((F5[k] * CC2 - F7[k] * CC3) / V3 + r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
            dCD3 = ((F5[k] * CD2 - F7[k] * CD3) / V3 + r2_(T3, CB3, CC3, CD3))

            dCA4 = ((F7[k] * CA3 + F9[k] * CA5 - (Fr1[k] + Fr2[k]) * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CA4) / V4)
            dCB4 = ((F7[k] * CB3 + F9[k] * CB5 - (Fr1[k] + Fr2[k]) * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CB4) / V4)
            dCC4 = ((F7[k] * CC3 + F9[k] * CC5 - (Fr1[k] + Fr2[k]) * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CC4) / V4)
            dCD4 = ((F7[k] * CD3 + F9[k] * CD5 - (Fr1[k] + Fr2[k]) * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CD4) / V4)

            dCA5 = ((Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CA5) / V5 - r3_(T5, CA5, CD5))
            dCB5 = ((Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CB5) / V5 - r2_(T5, CB5, CC5, CD5))
            dCC5 = ((Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CC5) / V5 - r2_(T5, CB5, CC5, CD5) + 2 * r3_(T5, CA5, CD5))
            dCD5 = ((Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10[k] * CD0 - F9[k] * CD5) / V5 + r2_(T5, CB5, CC5, CD5) - r3_(T5, CA5, CD5))

            return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
        end

        # Solve ODE 
        tspan = (0.0, dt)
        prob = ODEProblem(f, x0, tspan)
        soln = solve(prob, Rosenbrock23(), alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, save_everystep=false)

        # Obtain the next values for each element
        V1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][1]))
        V2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][2]))
        V3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][3]))
        V4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][4]))
        V5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][5]))

        T1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][6]))
        T2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][7]))
        T3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][8]))
        T4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][9]))
        T5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][10]))

        CA1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][11]))
        CB1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][12]))
        CC1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][13]))
        CD1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][14]))

        CA2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][15]))
        CB2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][16]))
        CC2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][17]))
        CD2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][18]))

        CA3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][19]))
        CB3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][20]))
        CC3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][21]))
        CD3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][22]))

        CA4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][23]))
        CB4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][24]))
        CC4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][25]))
        CD4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][26]))

        CA5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][27]))
        CB5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][28]))
        CC5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][29]))
        CD5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][30]))


    end

    V1 = V1_vec
    V2 = V2_vec
    V3 = V3_vec
    V4 = V4_vec
    V5 = V5_vec

    T1 = T1_vec
    T2 = T2_vec
    T3 = T3_vec
    T4 = T4_vec
    T5 = T5_vec

    CA1 = CA1_vec
    CB1 = CB1_vec
    CC1 = CC1_vec
    CD1 = CD1_vec

    CA2 = CA2_vec
    CB2 = CB2_vec
    CC2 = CC2_vec
    CD2 = CD2_vec

    CA3 = CA3_vec
    CB3 = CB3_vec
    CC3 = CC3_vec
    CD3 = CD3_vec

    CA4 = CA4_vec
    CB4 = CB4_vec
    CC4 = CC4_vec
    CD4 = CD4_vec

    CA5 = CA5_vec
    CB5 = CB5_vec
    CC5 = CC5_vec
    CD5 = CD5_vec

    ISE = sum(w.v * (V1_vec[k] - V1_sp)^2 + w.v * (V2_vec[k] - V2_sp)^2 + w.v * (V3_vec[k] - V3_sp)^2 + w.v * (V4_vec[k] - V4_sp)^2 + w.v * (V5_vec[k] - V5_sp)^2 +
              w.t1 * (T1_vec[k] - T1_sp)^2 + w.t2 * (T2_vec[k] - T2_sp)^2 + w.t3 * (T3_vec[k] - T3_sp)^2 + w.t4 * (T4_vec[k] - T4_sp)^2 + w.t5 * (T5_vec[k] - T5_sp)^2 +
              w.ca1 * (CA1_vec[k] - CA1_sp)^2 + w.ca2 * (CA2_vec[k] - CA2_sp)^2 + w.ca3 * (CA3_vec[k] - CA3_sp)^2 + w.ca4 * (CA4_vec[k] - CA4_sp)^2 + w.ca5 * (CA5_vec[k] - CA5_sp)^2 +
              w.cb1 * (CB1_vec[k] - CB1_sp)^2 + w.cb2 * (CB2_vec[k] - CB2_sp)^2 + w.cb3 * (CB3_vec[k] - CB3_sp)^2 + w.cb4 * (CB4_vec[k] - CB4_sp)^2 + w.cb5 * (CB5_vec[k] - CB5_sp)^2 +
              w.cc1 * (CC1_vec[k] - CC1_sp)^2 + w.cc2 * (CC2_vec[k] - CC2_sp)^2 + w.cc3 * (CC3_vec[k] - CC3_sp)^2 + w.cc4 * (CC4_vec[k] - CC4_sp)^2 + w.cc5 * (CC5_vec[k] - CC5_sp)^2 +
              w.cd1 * (CD1_vec[k] - CD1_sp)^2 + w.cd2 * (CD2_vec[k] - CD2_sp)^2 + w.cd3 * (CD3_vec[k] - CD3_sp)^2 + w.cd4 * (CD4_vec[k] - CD4_sp)^2 + w.cd5 * (CD5_vec[k] - CD5_sp)^2 for k = 1:N+1)


    ISC = sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 + w.f5 * (F5[k] - F5_sp)^2 +
              w.f6 * (F6[k] - F6_sp)^2 + w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 + w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 + w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
              w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2 for k = 1:N+1)

    # ISE = sum((V1_vec[k] - V1_sp)^2 + (V2_vec[k] - V2_sp)^2 + (V3_vec[k] - V3_sp)^2 + (V4_vec[k] - V4_sp)^2 + (V5_vec[k] - V5_sp)^2 +
    #           (T1_vec[k] - T1_sp)^2 + (T2_vec[k] - T2_sp)^2 + (T3_vec[k] - T3_sp)^2 + (T4_vec[k] - T4_sp)^2 + (T5_vec[k] - T5_sp)^2 +
    #           (CA1_vec[k] - CA1_sp)^2 + (CA2_vec[k] - CA2_sp)^2 + (CA3_vec[k] - CA3_sp)^2 + (CA4_vec[k] - CA4_sp)^2 + (CA5_vec[k] - CA5_sp)^2 +
    #           (CB1_vec[k] - CB1_sp)^2 + (CB2_vec[k] - CB2_sp)^2 + (CB3_vec[k] - CB3_sp)^2 + (CB4_vec[k] - CB4_sp)^2 + (CB5_vec[k] - CB5_sp)^2 +
    #           (CC1_vec[k] - CC1_sp)^2 + (CC2_vec[k] - CC2_sp)^2 + (CC3_vec[k] - CC3_sp)^2 + (CC4_vec[k] - CC4_sp)^2 + (CC5_vec[k] - CC5_sp)^2 +
    #           (CD1_vec[k] - CD1_sp)^2 + (CD2_vec[k] - CD2_sp)^2 + (CD3_vec[k] - CD3_sp)^2 + (CD4_vec[k] - CD4_sp)^2 + (CD5_vec[k] - CD5_sp)^2 for k = 1:N+1)



    # ISC = sum((F1[k] - F1_sp)^2 + (F2[k] - F2_sp)^2 + (F3[k] - F3_sp)^2 + (F4[k] - F4_sp)^2 + (F5[k] - F5_sp)^2 +
    #           (F6[k] - F6_sp)^2 + (F7[k] - F7_sp)^2 + (F8[k] - F8_sp)^2 + (F9[k] - F9_sp)^2 + (F10[k] - F10_sp)^2 + (Fr1[k] - Fr1_sp)^2 + (Fr2[k] - Fr2_sp)^2 +
    #           (Q1[k] - Q1_sp)^2 + (Q2[k] - Q2_sp)^2 + (Q3[k] - Q3_sp)^2 + (Q4[k] - Q4_sp)^2 + (Q5[k] - Q5_sp)^2 for k = 1:N+1)

    PI = ISE + ISC

    if isnan(PI)
        PI = 1e12
    end

    return ISE, ISC, PI
end

function takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

    # We are using a 30 second sampling time for the system, just with a fine discretization. Therefore, the next timestep is at k = 61

    global V1_init = V1[dtd+1]
    global V2_init = V2[dtd+1]
    global V3_init = V3[dtd+1]
    global V4_init = V4[dtd+1]
    global V5_init = V5[dtd+1]

    global T1_init = T1[dtd+1]
    global T2_init = T2[dtd+1]
    global T3_init = T3[dtd+1]
    global T4_init = T4[dtd+1]
    global T5_init = T5[dtd+1]

    global CA1_init = CA1[dtd+1]
    global CA2_init = CA2[dtd+1]
    global CA3_init = CA3[dtd+1]
    global CA4_init = CA4[dtd+1]
    global CA5_init = CA5[dtd+1]

    global CB1_init = CB1[dtd+1]
    global CB2_init = CB2[dtd+1]
    global CB3_init = CB3[dtd+1]
    global CB4_init = CB4[dtd+1]
    global CB5_init = CB5[dtd+1]

    global CC1_init = CC1[dtd+1]
    global CC2_init = CC2[dtd+1]
    global CC3_init = CC3[dtd+1]
    global CC4_init = CC4[dtd+1]
    global CC5_init = CC5[dtd+1]

    global CD1_init = CD1[dtd+1]
    global CD2_init = CD2[dtd+1]
    global CD3_init = CD3[dtd+1]
    global CD4_init = CD4[dtd+1]
    global CD5_init = CD5[dtd+1]
end

predict = Decomposition_Trajectory(zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P))

centr_PI_vec = zeros(P)
centr_ISE_vec = zeros(P)
centr_ISC_vec = zeros(P)
centr_time = zeros(P)

distr_PI_vec = zeros(P)
distr_ISE_vec = zeros(P)
distr_ISC_vec = zeros(P)
distr_time = zeros(P)

function mhmpc_sim()

    for mpc_step = 1:P

        println("MPC Step: $(mpc_step)")
        println()
        predict.V1[mpc_step] = V1_init
        predict.V2[mpc_step] = V2_init
        predict.V3[mpc_step] = V3_init
        predict.V4[mpc_step] = V4_init
        predict.V5[mpc_step] = V5_init

        predict.T1[mpc_step] = T1_init
        predict.T2[mpc_step] = T2_init
        predict.T3[mpc_step] = T3_init
        predict.T4[mpc_step] = T4_init
        predict.T5[mpc_step] = T5_init

        predict.CA1[mpc_step] = CA1_init
        predict.CA2[mpc_step] = CA2_init
        predict.CA3[mpc_step] = CA3_init
        predict.CA4[mpc_step] = CA4_init
        predict.CA5[mpc_step] = CA5_init

        predict.CB1[mpc_step] = CB1_init
        predict.CB2[mpc_step] = CB2_init
        predict.CB3[mpc_step] = CB3_init
        predict.CB4[mpc_step] = CB4_init
        predict.CB5[mpc_step] = CB5_init

        predict.CC1[mpc_step] = CC1_init
        predict.CC2[mpc_step] = CC2_init
        predict.CC3[mpc_step] = CC3_init
        predict.CC4[mpc_step] = CC4_init
        predict.CC5[mpc_step] = CC5_init

        predict.CD1[mpc_step] = CD1_init
        predict.CD2[mpc_step] = CD2_init
        predict.CD3[mpc_step] = CD3_init
        predict.CD4[mpc_step] = CD4_init
        predict.CD5[mpc_step] = CD5_init

        ex.T1[mpc_step] = T1_init
        ex.T2[mpc_step] = T2_init
        ex.T3[mpc_step] = T3_init
        ex.T4[mpc_step] = T4_init
        ex.T5[mpc_step] = T5_init

        ex.V1[mpc_step] = V1_init
        ex.V2[mpc_step] = V2_init
        ex.V3[mpc_step] = V3_init
        ex.V4[mpc_step] = V4_init
        ex.V5[mpc_step] = V5_init

        ex.CA1[mpc_step] = CA1_init
        ex.CA2[mpc_step] = CA2_init
        ex.CA3[mpc_step] = CA3_init
        ex.CA4[mpc_step] = CA4_init
        ex.CA5[mpc_step] = CA5_init

        ex.CB1[mpc_step] = CB1_init
        ex.CB2[mpc_step] = CB2_init
        ex.CB3[mpc_step] = CB3_init
        ex.CB4[mpc_step] = CB4_init
        ex.CB5[mpc_step] = CB5_init

        ex.CC1[mpc_step] = CC1_init
        ex.CC2[mpc_step] = CC2_init
        ex.CC3[mpc_step] = CC3_init
        ex.CC4[mpc_step] = CC4_init
        ex.CC5[mpc_step] = CC5_init

        ex.CD1[mpc_step] = CD1_init
        ex.CD2[mpc_step] = CD2_init
        ex.CD3[mpc_step] = CD3_init
        ex.CD4[mpc_step] = CD4_init
        ex.CD5[mpc_step] = CD5_init

        # Initialize fix structure for decomp
        for set_traj = 1

            # Set control variables to nominal steady state values
            fix.F1 .= F1_sp
            fix.F2 .= F2_sp
            fix.F3 .= F3_sp
            fix.F4 .= F4_sp
            fix.F5 .= F5_sp
            fix.F6 .= F6_sp
            fix.F7 .= F7_sp
            fix.F8 .= F8_sp
            fix.F9 .= F9_sp
            fix.F10 .= F10_sp
            fix.Fr1 .= Fr1_sp
            fix.Fr2 .= Fr2_sp

            fix.Q1 .= Q1_sp
            fix.Q2 .= Q2_sp
            fix.Q3 .= Q3_sp
            fix.Q4 .= Q4_sp
            fix.Q5 .= Q5_sp

            fix.V1[1] = V1_sp
            fix.V2[1] = V2_sp
            fix.V3[1] = V3_sp
            fix.V4[1] = V4_sp
            fix.V5[1] = V5_sp
            fix.T1[1] = T1_init
            fix.T2[1] = T2_init
            fix.T3[1] = T3_init
            fix.T4[1] = T4_init
            fix.T5[1] = T5_init

            fix.CA1[1] = CA1_init
            fix.CA2[1] = CA2_init
            fix.CA3[1] = CA3_init
            fix.CA4[1] = CA4_init
            fix.CA5[1] = CA5_init
            fix.CB1[1] = CB1_init
            fix.CB2[1] = CB2_init
            fix.CB3[1] = CB3_init
            fix.CB4[1] = CB4_init
            fix.CB5[1] = CB5_init
            fix.CC1[1] = CC1_init
            fix.CC2[1] = CC2_init
            fix.CC3[1] = CC3_init
            fix.CC4[1] = CC4_init
            fix.CC5[1] = CC5_init
            fix.CD1[1] = CD1_init
            fix.CD2[1] = CD2_init
            fix.CD3[1] = CD3_init
            fix.CD4[1] = CD4_init
            fix.CD5[1] = CD5_init

            for i = 1:N
                global k = i

                F1 = fix.F1[k]
                F2 = fix.F2[k]
                F3 = fix.F3[k]
                F4 = fix.F4[k]
                F5 = fix.F5[k]
                F6 = fix.F6[k]
                F7 = fix.F7[k]
                F8 = fix.F8[k]
                F9 = fix.F9[k]
                F10 = fix.F10[k]
                Fr1 = fix.Fr1[k]
                Fr2 = fix.Fr2[k]
                Q1 = fix.Q1[k]
                Q2 = fix.Q2[k]
                Q3 = fix.Q3[k]
                Q4 = fix.Q4[k]
                Q5 = fix.Q5[k]

                V1 = fix.V1[k]
                V2 = fix.V2[k]
                V3 = fix.V3[k]
                V4 = fix.V4[k]
                V5 = fix.V5[k]

                T1 = fix.T1[k]
                T2 = fix.T2[k]
                T3 = fix.T3[k]
                T4 = fix.T4[k]
                T5 = fix.T5[k]

                CA1 = fix.CA1[k]
                CB1 = fix.CB1[k]
                CC1 = fix.CC1[k]
                CD1 = fix.CD1[k]

                CA2 = fix.CA2[k]
                CB2 = fix.CB2[k]
                CC2 = fix.CC2[k]
                CD2 = fix.CD2[k]

                CA3 = fix.CA3[k]
                CB3 = fix.CB3[k]
                CC3 = fix.CC3[k]
                CD3 = fix.CD3[k]

                CA4 = fix.CA4[k]
                CB4 = fix.CB4[k]
                CC4 = fix.CC4[k]
                CD4 = fix.CD4[k]

                CA5 = fix.CA5[k]
                CB5 = fix.CB5[k]
                CC5 = fix.CC5[k]
                CD5 = fix.CD5[k]

                x0 = [fix.V1[k], fix.V2[k], fix.V3[k], fix.V4[k], fix.V5[k], fix.T1[k], fix.T2[k], fix.T3[k], fix.T4[k], fix.T5[k], fix.CA1[k], fix.CB1[k], fix.CC1[k], fix.CD1[k], fix.CA2[k], fix.CB2[k], fix.CC2[k], fix.CD2[k], fix.CA3[k], fix.CB3[k], fix.CC3[k], fix.CD3[k], fix.CA4[k], fix.CB4[k], fix.CC4[k], fix.CD4[k], fix.CA5[k], fix.CB5[k], fix.CC5[k], fix.CD5[k]]

                # Define the ODE function
                function f(y, p, t)
                    # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
                    V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

                    dV1 = F1 + F2 + Fr2 - F3
                    dV2 = F3 + F4 - F5
                    dV3 = F5 + F6 - F7
                    dV4 = F7 + F9 - F8 - Fr1 - Fr2
                    dV5 = F10 + Fr1 - F9

                    dT1 = (
                        ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
                          (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
                          (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
                          (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
                          (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
                         /
                         (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
                        (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
                        (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

                    dT2 = (
                        (Q2 + F4 * CB0 * H_B(TB0) +
                         (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
                         (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
                         (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
                         (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
                        /
                        (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
                        +
                        (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
                        /
                        (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

                    dT3 = (
                        ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
                          (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
                          (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
                          (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
                         /
                         (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
                        +
                        (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
                        /
                        (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

                    dT4 = ((Q4
                            + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
                            + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
                            + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
                            + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
                           /
                           (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
                    dT5 = (
                        ((Q5 +
                          F10 * CD0 * H_D(TD0)
                          + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
                          + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
                          + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
                          + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
                         /
                         (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
                        +
                        ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
                         /
                         (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


                    dCA1 = (
                        ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CA1)
                         /
                         V1) - r1(T1, CA1, CB1))

                    dCB1 = (
                        ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CB1) /
                         V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

                    dCC1 = (
                        ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CC1) /
                         V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

                    dCD1 = (
                        ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CD1) /
                         V1) + r2(T1, CB1, CC1, CD1))

                    dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
                    dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
                    dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
                    dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

                    dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
                    dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
                    dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
                    dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

                    dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
                    dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
                    dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
                    dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

                    dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
                    dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
                    dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
                    dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

                    return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
                end

                # Solve ODE 
                tspan = (0.0, dt)
                prob = ODEProblem(f, x0, tspan)
                soln = solve(prob, Rosenbrock23())

                # Obtain the next values for each element
                fix.V1[k+1] = soln.u[end][1]
                fix.V2[k+1] = soln.u[end][2]
                fix.V3[k+1] = soln.u[end][3]
                fix.V4[k+1] = soln.u[end][4]
                fix.V5[k+1] = soln.u[end][5]

                fix.T1[k+1] = soln.u[end][6]
                fix.T2[k+1] = soln.u[end][7]
                fix.T3[k+1] = soln.u[end][8]
                fix.T4[k+1] = soln.u[end][9]
                fix.T5[k+1] = soln.u[end][10]

                fix.CA1[k+1] = soln.u[end][11]
                fix.CB1[k+1] = soln.u[end][12]
                fix.CC1[k+1] = soln.u[end][13]
                fix.CD1[k+1] = soln.u[end][14]

                fix.CA2[k+1] = soln.u[end][15]
                fix.CB2[k+1] = soln.u[end][16]
                fix.CC2[k+1] = soln.u[end][17]
                fix.CD2[k+1] = soln.u[end][18]

                fix.CA3[k+1] = soln.u[end][19]
                fix.CB3[k+1] = soln.u[end][20]
                fix.CC3[k+1] = soln.u[end][21]
                fix.CD3[k+1] = soln.u[end][22]

                fix.CA4[k+1] = soln.u[end][23]
                fix.CB4[k+1] = soln.u[end][24]
                fix.CC4[k+1] = soln.u[end][25]
                fix.CD4[k+1] = soln.u[end][26]

                fix.CA5[k+1] = soln.u[end][27]
                fix.CB5[k+1] = soln.u[end][28]
                fix.CC5[k+1] = soln.u[end][29]
                fix.CD5[k+1] = soln.u[end][30]
            end

        end

        # Initialize ig structure for centralized
        for set_traj = 1

            # Set control variables to nominal steady state values
            ig.F1 .= F1_sp
            ig.F2 .= F2_sp
            ig.F3 .= F3_sp
            ig.F4 .= F4_sp
            ig.F5 .= F5_sp
            ig.F6 .= F6_sp
            ig.F7 .= F7_sp
            ig.F8 .= F8_sp
            ig.F9 .= F9_sp
            ig.F10 .= F10_sp
            ig.Fr1 .= Fr1_sp
            ig.Fr2 .= Fr2_sp

            ig.Q1 .= Q1_sp
            ig.Q2 .= Q2_sp
            ig.Q3 .= Q3_sp
            ig.Q4 .= Q4_sp
            ig.Q5 .= Q5_sp

            ig.V1[1] = V1_init
            ig.V2[1] = V2_init
            ig.V3[1] = V3_init
            ig.V4[1] = V4_init
            ig.V5[1] = V5_init
            ig.T1[1] = T1_init
            ig.T2[1] = T2_init
            ig.T3[1] = T3_init
            ig.T4[1] = T4_init
            ig.T5[1] = T5_init

            ig.CA1[1] = CA1_init
            ig.CA2[1] = CA2_init
            ig.CA3[1] = CA3_init
            ig.CA4[1] = CA4_init
            ig.CA5[1] = CA5_init
            ig.CB1[1] = CB1_init
            ig.CB2[1] = CB2_init
            ig.CB3[1] = CB3_init
            ig.CB4[1] = CB4_init
            ig.CB5[1] = CB5_init
            ig.CC1[1] = CC1_init
            ig.CC2[1] = CC2_init
            ig.CC3[1] = CC3_init
            ig.CC4[1] = CC4_init
            ig.CC5[1] = CC5_init
            ig.CD1[1] = CD1_init
            ig.CD2[1] = CD2_init
            ig.CD3[1] = CD3_init
            ig.CD4[1] = CD4_init
            ig.CD5[1] = CD5_init

            for i = 1:N
                global k = i

                F1 = ig.F1[k]
                F2 = ig.F2[k]
                F3 = ig.F3[k]
                F4 = ig.F4[k]
                F5 = ig.F5[k]
                F6 = ig.F6[k]
                F7 = ig.F7[k]
                F8 = ig.F8[k]
                F9 = ig.F9[k]
                F10 = ig.F10[k]
                Fr1 = ig.Fr1[k]
                Fr2 = ig.Fr2[k]
                Q1 = ig.Q1[k]
                Q2 = ig.Q2[k]
                Q3 = ig.Q3[k]
                Q4 = ig.Q4[k]
                Q5 = ig.Q5[k]

                V1 = ig.V1[k]
                V2 = ig.V2[k]
                V3 = ig.V3[k]
                V4 = ig.V4[k]
                V5 = ig.V5[k]

                T1 = ig.T1[k]
                T2 = ig.T2[k]
                T3 = ig.T3[k]
                T4 = ig.T4[k]
                T5 = ig.T5[k]

                CA1 = ig.CA1[k]
                CB1 = ig.CB1[k]
                CC1 = ig.CC1[k]
                CD1 = ig.CD1[k]

                CA2 = ig.CA2[k]
                CB2 = ig.CB2[k]
                CC2 = ig.CC2[k]
                CD2 = ig.CD2[k]

                CA3 = ig.CA3[k]
                CB3 = ig.CB3[k]
                CC3 = ig.CC3[k]
                CD3 = ig.CD3[k]

                CA4 = ig.CA4[k]
                CB4 = ig.CB4[k]
                CC4 = ig.CC4[k]
                CD4 = ig.CD4[k]

                CA5 = ig.CA5[k]
                CB5 = ig.CB5[k]
                CC5 = ig.CC5[k]
                CD5 = ig.CD5[k]

                x0 = [ig.V1[k], ig.V2[k], ig.V3[k], ig.V4[k], ig.V5[k], ig.T1[k], ig.T2[k], ig.T3[k], ig.T4[k], ig.T5[k], ig.CA1[k], ig.CB1[k], ig.CC1[k], ig.CD1[k], ig.CA2[k], ig.CB2[k], ig.CC2[k], ig.CD2[k], ig.CA3[k], ig.CB3[k], ig.CC3[k], ig.CD3[k], ig.CA4[k], ig.CB4[k], ig.CC4[k], ig.CD4[k], ig.CA5[k], ig.CB5[k], ig.CC5[k], ig.CD5[k]]

                # Define the ODE function
                function f(y, p, t)
                    # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
                    V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

                    dV1 = F1 + F2 + Fr2 - F3
                    dV2 = F3 + F4 - F5
                    dV3 = F5 + F6 - F7
                    dV4 = F7 + F9 - F8 - Fr1 - Fr2
                    dV5 = F10 + Fr1 - F9

                    dT1 = (
                        ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
                          (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
                          (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
                          (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
                          (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
                         /
                         (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
                        (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
                        (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

                    dT2 = (
                        (Q2 + F4 * CB0 * H_B(TB0) +
                         (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
                         (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
                         (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
                         (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
                        /
                        (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
                        +
                        (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
                        /
                        (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

                    dT3 = (
                        ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
                          (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
                          (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
                          (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
                         /
                         (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
                        +
                        (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
                        /
                        (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

                    dT4 = ((Q4
                            + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
                            + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
                            + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
                            + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
                           /
                           (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
                    dT5 = (
                        ((Q5 +
                          F10 * CD0 * H_D(TD0)
                          + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
                          + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
                          + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
                          + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
                         /
                         (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
                        +
                        ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
                         /
                         (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


                    dCA1 = (
                        ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CA1)
                         /
                         V1) - r1(T1, CA1, CB1))

                    dCB1 = (
                        ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CB1) /
                         V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

                    dCC1 = (
                        ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CC1) /
                         V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

                    dCD1 = (
                        ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
                          -
                          F3 * CD1) /
                         V1) + r2(T1, CB1, CC1, CD1))

                    dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
                    dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
                    dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
                    dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

                    dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
                    dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
                    dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
                    dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

                    dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
                    dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
                    dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
                    dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

                    dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
                    dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
                    dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
                    dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

                    return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
                end

                # Solve ODE 
                tspan = (0.0, dt)
                prob = ODEProblem(f, x0, tspan)
                soln = solve(prob, Rosenbrock23())

                # Obtain the next values for each element
                ig.V1[k+1] = soln.u[end][1]
                ig.V2[k+1] = soln.u[end][2]
                ig.V3[k+1] = soln.u[end][3]
                ig.V4[k+1] = soln.u[end][4]
                ig.V5[k+1] = soln.u[end][5]

                ig.T1[k+1] = soln.u[end][6]
                ig.T2[k+1] = soln.u[end][7]
                ig.T3[k+1] = soln.u[end][8]
                ig.T4[k+1] = soln.u[end][9]
                ig.T5[k+1] = soln.u[end][10]

                ig.CA1[k+1] = soln.u[end][11]
                ig.CB1[k+1] = soln.u[end][12]
                ig.CC1[k+1] = soln.u[end][13]
                ig.CD1[k+1] = soln.u[end][14]

                ig.CA2[k+1] = soln.u[end][15]
                ig.CB2[k+1] = soln.u[end][16]
                ig.CC2[k+1] = soln.u[end][17]
                ig.CD2[k+1] = soln.u[end][18]

                ig.CA3[k+1] = soln.u[end][19]
                ig.CB3[k+1] = soln.u[end][20]
                ig.CC3[k+1] = soln.u[end][21]
                ig.CD3[k+1] = soln.u[end][22]

                ig.CA4[k+1] = soln.u[end][23]
                ig.CB4[k+1] = soln.u[end][24]
                ig.CC4[k+1] = soln.u[end][25]
                ig.CD4[k+1] = soln.u[end][26]

                ig.CA5[k+1] = soln.u[end][27]
                ig.CB5[k+1] = soln.u[end][28]
                ig.CC5[k+1] = soln.u[end][29]
                ig.CD5[k+1] = soln.u[end][30]
            end

        end


        global cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5 = cmpc()
        centr_ISE, centr_ISC, centr_PI = getPI(cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5)

        ex.F1c[mpc_step] = cF1[1]
        ex.F2c[mpc_step] = cF2[1]
        ex.F3c[mpc_step] = cF3[1]
        ex.F4c[mpc_step] = cF4[1]
        ex.F5c[mpc_step] = cF5[1]
        ex.F6c[mpc_step] = cF6[1]
        ex.F7c[mpc_step] = cF7[1]
        ex.F8c[mpc_step] = cF8[1]
        ex.F9c[mpc_step] = cF9[1]
        ex.F10c[mpc_step] = cF10[1]
        ex.Fr1c[mpc_step] = cFr1[1]
        ex.Fr2c[mpc_step] = cFr2[1]

        ex.Q1c[mpc_step] = cQ1[1]
        ex.Q2c[mpc_step] = cQ2[1]
        ex.Q3c[mpc_step] = cQ3[1]
        ex.Q4c[mpc_step] = cQ4[1]
        ex.Q5c[mpc_step] = cQ5[1]


        centr_PI_vec[mpc_step] = centr_PI
        centr_ISE_vec[mpc_step] = centr_ISE
        centr_ISC_vec[mpc_step] = centr_ISC
        centr_time[mpc_step] = cmpc_solve_time

        println("cPI = $(centr_PI)")
        println("cISE = $(centr_ISE)")
        println("cISC = $(centr_ISC)")
        println("time = $(cmpc_solve_time)")
        println()

        global dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5 = dmpc()
        distr_ISE, distr_ISC, distr_PI = getPI(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)

        ex.F1d[mpc_step] = dF1[1]
        ex.F2d[mpc_step] = dF2[1]
        ex.F3d[mpc_step] = dF3[1]
        ex.F4d[mpc_step] = dF4[1]
        ex.F5d[mpc_step] = dF5[1]
        ex.F6d[mpc_step] = dF6[1]
        ex.F7d[mpc_step] = dF7[1]
        ex.F8d[mpc_step] = dF8[1]
        ex.F9d[mpc_step] = dF9[1]
        ex.F10d[mpc_step] = dF10[1]
        ex.Fr1d[mpc_step] = dFr1[1]
        ex.Fr2d[mpc_step] = dFr2[1]

        ex.Q1d[mpc_step] = dQ1[1]
        ex.Q2d[mpc_step] = dQ2[1]
        ex.Q3d[mpc_step] = dQ3[1]
        ex.Q4d[mpc_step] = dQ4[1]
        ex.Q5d[mpc_step] = dQ5[1]

        distr_PI_vec[mpc_step] = distr_PI
        distr_ISE_vec[mpc_step] = distr_ISE
        distr_ISC_vec[mpc_step] = distr_ISC
        distr_time[mpc_step] = dmpc_solve_time

        println("dPI = $(distr_PI)")
        println("dISE = $(distr_ISE)")
        println("dISC = $(distr_ISC)")
        println("time = $(dmpc_solve_time)")
        println()

        ex.PIc[mpc_step] = centr_PI
        ex.PId[mpc_step] = distr_PI
        ex.ISEc[mpc_step] = centr_ISE
        ex.ISEd[mpc_step] = distr_ISE
        ex.ISCc[mpc_step] = centr_ISC
        ex.ISCd[mpc_step] = distr_ISC

        if centr_PI < distr_PI
            println("Centralized chosen")
            println()
            V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5)
            takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

            predict.F1[mpc_step] = cF1[1]
            predict.F2[mpc_step] = cF2[1]
            predict.F3[mpc_step] = cF3[1]
            predict.F4[mpc_step] = cF4[1]
            predict.F5[mpc_step] = cF5[1]
            predict.F6[mpc_step] = cF6[1]
            predict.F7[mpc_step] = cF7[1]
            predict.F8[mpc_step] = cF8[1]
            predict.F9[mpc_step] = cF9[1]
            predict.F10[mpc_step] = cF10[1]
            predict.Fr1[mpc_step] = cFr1[1]
            predict.Fr2[mpc_step] = cFr2[1]

            predict.Q1[mpc_step] = cQ1[1]
            predict.Q2[mpc_step] = cQ2[1]
            predict.Q3[mpc_step] = cQ3[1]
            predict.Q4[mpc_step] = cQ4[1]
            predict.Q5[mpc_step] = cQ5[1]
        else
            println("Distributed chosen")
            println()
            V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)
            takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

            predict.F1[mpc_step] = dF1[1]
            predict.F2[mpc_step] = dF2[1]
            predict.F3[mpc_step] = dF3[1]
            predict.F4[mpc_step] = dF4[1]
            predict.F5[mpc_step] = dF5[1]
            predict.F6[mpc_step] = dF6[1]
            predict.F7[mpc_step] = dF7[1]
            predict.F8[mpc_step] = dF8[1]
            predict.F9[mpc_step] = dF9[1]
            predict.F10[mpc_step] = dF10[1]
            predict.Fr1[mpc_step] = dFr1[1]
            predict.Fr2[mpc_step] = dFr2[1]

            predict.Q1[mpc_step] = dQ1[1]
            predict.Q2[mpc_step] = dQ2[1]
            predict.Q3[mpc_step] = dQ3[1]
            predict.Q4[mpc_step] = dQ4[1]
            predict.Q5[mpc_step] = dQ5[1]
        end

    end

end

global dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5 = dmpc(1)
V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)
distr_ISE, distr_ISC, distr_PI = getPI(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)

println("PI: $distr_PI")
println("ISC: $distr_ISC")
println("ISE: $distr_ISE")


plot(V1[1:end-1])
hline!([V1_sp])
