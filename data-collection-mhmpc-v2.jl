# Benzene alkylation with ethlyene
using JuMP, Ipopt, Plots, OrdinaryDiffEq, LinearAlgebra, DataFrames, CSV

# global mhmpc_src_instance = 261

# while mhmpc_src_instance <= 299

N = 1200  # Control horizon
dt = 1 # Sampling time of the system
Nd = 40
dtd = 30
P = 80

global s_path = 0.01
global cpu_max = 5.0 # Maximum cpu time for Ipopt
global dual_inf_tol = Float64(1 * 10^(0))
global opt_tol = Float64(1 * 10^(0))
global constr_viol_tol = Float64(1 * 10^(0))
global compl_inf_tol = Float64(1 * 10^(0))

global c_cpu_max = 5.0 # Maximum cpu time for Ipopt
global c_dual_inf_tol = Float64(1 * 10^(0))
global c_opt_tol = Float64(1 * 10^(0))
global c_constr_viol_tol = Float64(1 * 10^(0))
global c_compl_inf_tol = Float64(1 * 10^(0))

block_size = Int(dtd / 1) - 1
k_indices = Int[]  # Initialize an empty array

# Loop to generate the pattern until N - 1
for start_k in 0:dtd:N-1
    append!(k_indices, start_k:start_k+block_size-1)
end

# append!(k_indices, N-1)
# Ensure indices do not exceed N - 1
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


data = CSV.read("C:\\Users\\escooper\\git\\bae\\final-formulation\\src-data.csv", DataFrame)
instance = data[mhmpc_src_instance, :]

# Declare setpoint, initial condition, parameters
function init_sp_and_params()

    global V1_init = instance[1]
    global V2_init = instance[2]
    global V3_init = instance[3]
    global V4_init = instance[4]
    global V5_init = instance[5]

    global T1_init = instance[6]
    global T2_init = instance[7]
    global T3_init = instance[8]
    global T4_init = instance[9]
    global T5_init = instance[10]

    global CA1_init = instance[11]
    global CB1_init = instance[16]
    global CC1_init = instance[21]
    global CD1_init = instance[26]

    global CA2_init = instance[12]
    global CB2_init = instance[17]
    global CC2_init = instance[22]
    global CD2_init = instance[27]

    global CA3_init = instance[13]
    global CB3_init = instance[18]
    global CC3_init = instance[23]
    global CD3_init = instance[28]

    global CA4_init = instance[14]
    global CB4_init = instance[19]
    global CC4_init = instance[24]
    global CD4_init = instance[29]

    global CA5_init = instance[15]
    global CB5_init = instance[20]
    global CC5_init = instance[25]
    global CD5_init = instance[30]

    global V1_sp = instance[1]
    global V2_sp = instance[2]
    global V3_sp = instance[3]
    global V4_sp = instance[4]
    global V5_sp = instance[5]

    global T1_sp = instance[31]
    global T2_sp = instance[32]
    global T3_sp = instance[33]
    global T4_sp = instance[34]
    global T5_sp = instance[35]

    global CA1_sp = instance[36]
    global CA2_sp = instance[37]
    global CA3_sp = instance[38]
    global CA4_sp = instance[39]
    global CA5_sp = instance[40]

    global CB1_sp = instance[41]
    global CB2_sp = instance[42]
    global CB3_sp = instance[43]
    global CB4_sp = instance[44]
    global CB5_sp = instance[45]

    global CC1_sp = instance[46]
    global CC2_sp = instance[47]
    global CC3_sp = instance[48]
    global CC4_sp = instance[49]
    global CC5_sp = instance[50]

    global CD1_sp = instance[51]
    global CD2_sp = instance[52]
    global CD3_sp = instance[53]
    global CD4_sp = instance[54]
    global CD5_sp = instance[55]


    global F1_sp = instance[56]
    global F2_sp = instance[57]
    global F3_sp = instance[58]
    global F4_sp = instance[59]
    global F5_sp = instance[60]
    global F6_sp = instance[61]
    global F7_sp = instance[62]
    global F8_sp = instance[63]
    global F9_sp = instance[64]
    global F10_sp = instance[65]
    global Fr1_sp = instance[66]
    global Fr2_sp = instance[67]

    global Q1_sp = instance[68]
    global Q2_sp = instance[69]
    global Q3_sp = instance[70]
    global Q4_sp = instance[71]
    global Q5_sp = instance[72]

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
    global TA0 = 473
    global TB0 = 473
    global TD0 = 473


    global delH_r1 = -1.53e5
    global delH_r2 = -1.118e5
    global delH_r3 = 4.141e5

    global R = 8.314
end


init_sp_and_params()

w = Weights(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

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
ub_f = 3.0
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

fix = Decomposition_Trajectory(zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1))
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

    for i = 1:Nd
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
        tspan = (0.0, dtd)
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
        V1[k=0:N], (lower_bound=0.2 * V1_sp, upper_bound=1.8 * V1_sp)
        V2[k=0:N], (lower_bound=0.2 * V2_sp, upper_bound=1.8 * V2_sp)
        V3[k=0:N], (lower_bound=0.2 * V3_sp, upper_bound=1.8 * V3_sp)
        V4[k=0:N], (lower_bound=0.2 * V4_sp, upper_bound=1.8 * V4_sp)
        V5[k=0:N], (lower_bound=0.2 * V5_sp, upper_bound=1.8 * V5_sp)

        # Temperature, state, [=] K
        T1[k=0:N], (lower_bound=0.2 * T1_sp, upper_bound=1.8 * T1_sp)
        T2[k=0:N], (lower_bound=0.2 * T2_sp, upper_bound=1.8 * T2_sp)
        T3[k=0:N], (lower_bound=0.2 * T3_sp, upper_bound=1.8 * T3_sp)
        T4[k=0:N], (lower_bound=0.2 * T4_sp, upper_bound=1.8 * T4_sp)
        T5[k=0:N], (lower_bound=0.2 * T5_sp, upper_bound=1.8 * T5_sp)

        # Concentration, state, [=] mol/m3
        CA1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CA2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CA3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CA4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CA5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

        CB1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CB2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CB3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CB4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CB5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

        CC1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CC2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CC3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CC4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CC5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

        CD1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CD2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CD3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CD4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
        CD5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

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
        JuMP.fix(Q1[k], Q1_sp; force=true)
        JuMP.fix(Q2[k], Q2_sp; force=true)
        JuMP.fix(Q3[k], Q3_sp; force=true)
        JuMP.fix(Q4[k], Q4_sp; force=true)
        JuMP.fix(Q5[k], Q5_sp; force=true)

        JuMP.fix(F7[k], F7_sp; force=true)
        # JuMP.fix(F8[k], F8_sp; force=true)
        # JuMP.fix(F9[k], F9_sp; force=true)
        # JuMP.fix(F10[k], F10_sp; force=true)
        # JuMP.fix(Fr1[k], Fr1_sp; force=true)
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
        dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + Fr2[k] - F3[k]) * dt == V1_sp

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

        dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - F5[k]) * dt == V2_sp

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

        dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - F7[k]) * dt == V3_sp


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

        dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4_sp

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

        dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5_sp

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

        # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + Fr2[k] - F3[k] == -(V1[k] - V1_sp) / 200
        # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / 200
        # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / 200
        # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
        # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200

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
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    set_silent(mpc)

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

function dmpc_1()

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
        V1[k=0:Nd], (lower_bound=0.2 * V1_sp, upper_bound=1.8 * V1_sp, start=V1_sp)
        V2[k=0:Nd], (lower_bound=0.2 * V2_sp, upper_bound=1.8 * V2_sp, start=V2_sp)

        # Temperature, state, [=] K
        T1[k=0:Nd], (lower_bound=0.2 * T1_sp, upper_bound=1.8 * T1_sp, start=T1_sp)
        T2[k=0:Nd], (lower_bound=0.2 * T2_sp, upper_bound=1.8 * T2_sp, start=T2_sp)

        # Concentration, state, [=] mol/m3
        CA1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA1_sp)
        CA2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA2_sp)

        CB1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB1_sp)
        CB2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB2_sp)

        CC1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC1_sp)
        CC2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC2_sp)

        CD1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD1_sp)
        CD2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD2_sp)

        F1[k=0:Nd], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
        F2[k=0:Nd], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)
        F3[k=0:Nd], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
        F4[k=0:Nd], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)

        Q1[k=0:Nd], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
        Q2[k=0:Nd], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)

    end


    # for k = 0:N
    #     JuMP.fix(F1[k], F1_sp; force=true)
    #     JuMP.fix(F2[k], F2_sp; force=true)
    #     JuMP.fix(F3[k], F3_sp; force=true)
    #     JuMP.fix(F4[k], F4_sp; force=true)
    # end


    for k = 0:Nd
        JuMP.fix(Q1[k], Q1_sp; force=true)
        JuMP.fix(Q2[k], Q2_sp; force=true)
        # JuMP.fix(Q3[k], Q3_sp; force=true)
        # JuMP.fix(Q4[k], Q4_sp; force=true)
        # JuMP.fix(Q5[k], Q5_sp; force=true)
    end

    for k = 0:Nd
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

    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system
        dV1_dt[k=0:Nd-1], V1[k] + (F1[k] + F2[k] + fix.Fr2[k+1] - F3[k]) * dtd == V1_sp

        dT1_dt[k=0:Nd-1], T1[k] + (
            ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
              (fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_A(fix.T4[k+1]) - F3[k] * CA1[k] * H_A(T1[k])) +
              (fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_B(fix.T4[k+1]) - F3[k] * CB1[k] * H_B(T1[k])) +
              (fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_C(fix.T4[k+1]) - F3[k] * CC1[k] * H_C(T1[k])) +
              (fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_D(fix.T4[k+1]) - F3[k] * CD1[k] * H_D(T1[k])))
             /
             (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
            (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
            (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dtd == T1[k+1]

        dCA1_dt[k=0:Nd-1], CA1[k] + (
            ((F1[k] * CA0 + fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CA1[k])
             /
             V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dtd == CA1[k+1]

        dCB1_dt[k=0:Nd-1], CB1[k] + (
            ((F2[k] * CB0 + fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CB1[k]) /
             V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CB1[k+1]

        dCC1_dt[k=0:Nd-1], CC1[k] + (
            ((fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CC1[k]) /
             V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CC1[k+1]

        dCD1_dt[k=0:Nd-1], CD1[k] + (
            ((fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
              -
              F3[k] * CD1[k]) /
             V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CD1[k+1]

        dV2_dt[k=0:Nd-1], V2[k] + (F3[k] + F4[k] - fix.F5[k+1]) * dtd == V2_sp

        dT2_dt[k=0:Nd-1], T2[k] + (
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
            (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dtd == T2[k+1]


        dCA2_dt[k=0:Nd-1], CA2[k] + (((F3[k] * CA1[k] - fix.F5[k+1] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dtd == CA2[k+1]
        dCB2_dt[k=0:Nd-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - fix.F5[k+1] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CB2[k+1]
        dCC2_dt[k=0:Nd-1], CC2[k] + ((F3[k] * CC1[k] - fix.F5[k+1] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CC2[k+1]
        dCD2_dt[k=0:Nd-1], CD2[k] + ((F3[k] * CD1[k] - fix.F5[k+1] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CD2[k+1]

        # volHoldUp11[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] <= (-(V1[k] - V1_sp) / 200) + s_path
        # volHoldUp12[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] >= (-(V1[k] - V1_sp) / 200) - s_path
        # volHoldUp21[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] <= (-(V2[k] - V2_sp) / 200) + s_path
        # volHoldUp22[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] >= (-(V2[k] - V2_sp) / 200) - s_path

        # volHoldUp1[k=0:Nd-1], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / 200
        # volHoldUp2[k=0:Nd-1], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / 200

        # volDec1[k=0:N-1], (V1[k+1] - V1_sp) <= 0.8 * (V1[k] - V1_sp)
        # volDec2[k=0:N-1], (V2[k+1] - V2_sp) <= 0.8 * (V2[k] - V2_sp)

        # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / 200
        # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / 200

    end



    @NLobjective(mpc, Min, 1e-5 * sum(
        w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 +
        w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 +
        w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 +
        w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 +
        w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 +
        w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2
        for k = 0:Nd) +
                           1e-5 * sum(
        w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
        w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 for k = 0:Nd
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    set_silent(mpc)

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

function dmpc_2()

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
        V3[k=0:Nd], (lower_bound=0.2 * V3_sp, upper_bound=1.8 * V3_sp, start=V3_sp)

        # Temperature, state, [=] K
        T3[k=0:Nd], (lower_bound=0.2 * T3_sp, upper_bound=1.8 * T3_sp, start=T3_sp)

        # Concentration, state, [=] mol/m3
        CA3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA3_sp)

        CB3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB3_sp)

        CC3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC3_sp)

        CD3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD3_sp)

        F5[k=0:Nd], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
        F6[k=0:Nd], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)

        Q3[k=0:Nd], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)


    end

    for k = 0:Nd
        # JuMP.fix(F5[k], F5_sp; force=true)
        # JuMP.fix(F6[k], F6_sp; force=true)
    end

    for k = 0:Nd
        # JuMP.fix(Q1[k], Q1_sp; force=true)
        # JuMP.fix(Q2[k], Q2_sp; force=true)
        JuMP.fix(Q3[k], Q3_sp; force=true)
        # JuMP.fix(Q4[k], Q4_sp; force=true)
        # JuMP.fix(Q5[k], Q5_sp; force=true)
    end

    for k = 0:Nd
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


        # volDec3[k=0:N-1], (V3[k+1] - V3_sp) <= 0.8 * (V3[k] - V3_sp)

    end

    @NLconstraints mpc begin

        dV3_dt[k=0:Nd-1], V3[k] + (F5[k] + F6[k] - fix.F7[k+1]) * dtd == V3_sp

        dT3_dt[k=0:Nd-1], T3[k] + (
            ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * fix.CA2[k+1] * H_A(fix.T2[k+1]) - fix.F7[k+1] * CA3[k] * H_A(T3[k])) +
              (F5[k] * fix.CB2[k+1] * H_B(fix.T2[k+1]) - fix.F7[k+1] * CB3[k] * H_B(T3[k])) +
              (F5[k] * fix.CC2[k+1] * H_C(fix.T2[k+1]) - fix.F7[k+1] * CC3[k] * H_C(T3[k])) +
              (F5[k] * fix.CD2[k+1] * H_D(fix.T2[k+1]) - fix.F7[k+1] * CD3[k] * H_D(T3[k])))
             /
             (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
            +
            (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
            /
            (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dtd == T3[k+1]

        dCA3_dt[k=0:Nd-1], CA3[k] + (((F5[k] * fix.CA2[k+1] - fix.F7[k+1] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dtd == CA3[k+1]
        dCB3_dt[k=0:Nd-1], CB3[k] + (((F5[k] * fix.CB2[k+1] + F6[k] * CB0 - fix.F7[k+1] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CB3[k+1]
        dCC3_dt[k=0:Nd-1], CC3[k] + ((F5[k] * fix.CC2[k+1] - fix.F7[k+1] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CC3[k+1]
        dCD3_dt[k=0:Nd-1], CD3[k] + ((F5[k] * fix.CD2[k+1] - fix.F7[k+1] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CD3[k+1]
        # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / 200




        # volHoldUp3[k=0:Nd-1], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / 200
        # volHoldUp31[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] <= (-(V3[k] - V3_sp) / 200) + s_path
        # volHoldUp32[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] >= (-(V3[k] - V3_sp) / 200) - s_path
    end


    @NLobjective(
        mpc,
        Min,
        1e-5 * sum(
            w.v * (V3[k] - V3_sp)^2 +
            w.t3 * (T3[k] - T3_sp)^2 +
            w.ca3 * (CA3[k] - CA3_sp)^2 +
            w.cb3 * (CB3[k] - CB3_sp)^2 +
            w.cc3 * (CC3[k] - CC3_sp)^2 +
            w.cd3 * (CD3[k] - CD3_sp)^2
            for k = 0:Nd) +
        1e-5 * sum(
            w.f5 * (F5[k] - F5_sp)^2 +
            w.f6 * (F6[k] - F6_sp)^2 +
            w.q3 * (Q3[k] - Q3_sp)^2
            for k = 0:Nd
        )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    set_silent(mpc)

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

function dmpc_3()

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
        # State variables
        # Volume, state, [=] m3
        V4[k=0:Nd], (lower_bound=0.2 * V4_sp, upper_bound=1.8 * V4_sp, start=V4_sp)
        V5[k=0:Nd], (lower_bound=0.2 * V5_sp, upper_bound=1.8 * V5_sp, start=V5_sp)

        # Temperature, state, [=] K
        T4[k=0:Nd], (lower_bound=0.2 * T4_sp, upper_bound=1.8 * T4_sp, start=T4_sp)
        T5[k=0:Nd], (lower_bound=0.2 * T5_sp, upper_bound=1.8 * T5_sp, start=T5_sp)

        # Concentration, state, [=] mol/m3
        CA4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA4_sp)
        CA5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA5_sp)

        CB4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB4_sp)
        CB5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB5_sp)

        CC4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC4_sp)
        CC5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC5_sp)

        CD4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD4_sp)
        CD5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD5_sp)

        # Control variables
        # Flow, control [=] m3/s
        F7[k=0:Nd], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
        F8[k=0:Nd], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
        F9[k=0:Nd], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
        F10[k=0:Nd], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
        Fr1[k=0:Nd], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
        Fr2[k=0:Nd], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)



        Q4[k=0:Nd], (start = Q4_sp)
        Q5[k=0:Nd], (start = Q5_sp)
    end

    for k = 0:Nd
        JuMP.fix(F7[k], F7_sp; force=true)
        # JuMP.fix(F8[k], F8_sp; force=true)
        # JuMP.fix(F9[k], F9_sp; force=true)
        # JuMP.fix(F10[k], F10_sp; force=true)
        # JuMP.fix(Fr1[k], Fr1_sp; force=true)
        JuMP.fix(Fr2[k], Fr2_sp; force=true)
    end

    for k = 0:Nd
        # JuMP.fix(Q1[k], Q1_sp; force=true)
        # JuMP.fix(Q2[k], Q2_sp; force=true)
        # JuMP.fix(Q3[k], Q3_sp; force=true)
        JuMP.fix(Q4[k], Q4_sp; force=true)
        JuMP.fix(Q5[k], Q5_sp; force=true)
    end

    for k = 0:Nd
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

        # volDec4[k=0:N-1], (V4[k+1] - V4_sp) <= 0.8 * (V4[k] - V4_sp)
        # volDec5[k=0:N-1], (V5[k+1] - V5_sp) <= 0.8 * (V5[k] - V5_sp)

    end

    @NLconstraints mpc begin
        # NLconstraints are the differential equations that describe the dynamics of the system

        dV4_dt[k=0:Nd-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dtd == V4_sp
        dT4_dt[k=0:Nd-1], T4[k] +
                          ((Q4[k]
                            + (F7[k] * fix.CA3[k+1] * H_A(fix.T3[k+1]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
                            + (F7[k] * fix.CB3[k+1] * H_B(fix.T3[k+1]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
                            + (F7[k] * fix.CC3[k+1] * H_C(fix.T3[k+1]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
                            + (F7[k] * fix.CD3[k+1] * H_D(fix.T3[k+1]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
                           /
                           (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dtd == T4[k+1]

        dCA4_dt[k=0:Nd-1], CA4[k] + ((F7[k] * fix.CA3[k+1] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dtd == CA4[k+1]
        dCB4_dt[k=0:Nd-1], CB4[k] + ((F7[k] * fix.CB3[k+1] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dtd == CB4[k+1]
        dCC4_dt[k=0:Nd-1], CC4[k] + ((F7[k] * fix.CC3[k+1] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dtd == CC4[k+1]
        dCD4_dt[k=0:Nd-1], CD4[k] + ((F7[k] * fix.CD3[k+1] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dtd == CD4[k+1]

        dV5_dt[k=0:Nd-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dtd == V5_sp

        dT5_dt[k=0:Nd-1], T5[k] + (
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
             (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dtd == T5[k+1]

        dCA5_dt[k=0:Nd-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dtd == CA5[k+1]
        dCB5_dt[k=0:Nd-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dtd == CB5[k+1]
        dCC5_dt[k=0:Nd-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dtd == CC5[k+1]
        dCD5_dt[k=0:Nd-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dtd == CD5[k+1]
        # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
        # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200

        # volHoldUp4[k=0:N-1], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
        # volHoldUp5[k=0:N-1], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200


        # volHoldUp41[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] <= -(V4[k] - V4_sp) / 200 + s_path
        # volHoldUp42[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] >= -(V4[k] - V4_sp) / 200 - s_path

        # volHoldUp51[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] <= -(V5[k] - V5_sp) / 200 + s_path
        # volHoldUp52[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] >= -(V5[k] - V5_sp) / 200 - s_path


    end


    @NLobjective(mpc, Min, sum(
        1e-5 * w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
        w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
        w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
        w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
        w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
        w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
        for k = 0:Nd) +
                           1e-5 * sum(
        w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
        w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
        w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
        w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
        for k = 0:Nd
    )
    )

    set_optimizer_attribute(mpc, "max_iter", 100000)
    set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
    set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
    set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
    set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
    # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
    # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
    # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
    # set_optimizer_attribute(mpc, "mu_init", 1e-3)
    # set_optimizer_attribute(mpc, "linear_solver", "ma57")

    set_silent(mpc)

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

function dmpc()

    max_steps = 15
    global dmpc_solve_time = 0
    for steps = 1:max_steps

        F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2 = dmpc_1()
        F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3 = dmpc_2()
        F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5 = dmpc_3()

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


        global dmpc_solve_time = dmpc_solve_time + max(dmpc1_solve_time, dmpc2_solve_time, dmpc3_solve_time)

        if dmpc_solve_time > c_cpu_max
            break
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

    Q1 = ones(dtd) .* fix.Q1[1]
    Q2 = ones(dtd) .* fix.Q2[1]
    Q3 = ones(dtd) .* fix.Q3[1]
    Q4 = ones(dtd) .* fix.Q4[1]
    Q5 = ones(dtd) .* fix.Q5[1]

    F1 = ones(dtd) .* fix.F1[1]
    F2 = ones(dtd) .* fix.F2[1]
    F3 = ones(dtd) .* fix.F3[1]
    F4 = ones(dtd) .* fix.F4[1]
    F5 = ones(dtd) .* fix.F5[1]
    F6 = ones(dtd) .* fix.F6[1]
    F7 = ones(dtd) .* fix.F7[1]
    F8 = ones(dtd) .* fix.F8[1]
    F9 = ones(dtd) .* fix.F9[1]
    F10 = ones(dtd) .* fix.F10[1]
    Fr1 = ones(dtd) .* fix.Fr1[1]
    Fr2 = ones(dtd) .* fix.Fr2[1]

    for i = 2:Nd
        append!(F1, ones(dtd) .* fix.F1[i])
        append!(F2, ones(dtd) .* fix.F2[i])
        append!(F3, ones(dtd) .* fix.F3[i])
        append!(F4, ones(dtd) .* fix.F4[i])
        append!(F5, ones(dtd) .* fix.F5[i])
        append!(F6, ones(dtd) .* fix.F6[i])
        append!(F7, ones(dtd) .* fix.F7[i])
        append!(F8, ones(dtd) .* fix.F8[i])
        append!(F9, ones(dtd) .* fix.F9[i])
        append!(F10, ones(dtd) .* fix.F10[i])
        append!(Fr1, ones(dtd) .* fix.Fr1[i])
        append!(Fr2, ones(dtd) .* fix.Fr2[i])

        append!(Q1, ones(dtd) .* fix.Q1[i])
        append!(Q2, ones(dtd) .* fix.Q2[i])
        append!(Q3, ones(dtd) .* fix.Q3[i])
        append!(Q4, ones(dtd) .* fix.Q4[i])
        append!(Q5, ones(dtd) .* fix.Q5[i])
    end


    append!(F1, F1_sp)
    append!(F2, F2_sp)
    append!(F3, F3_sp)
    append!(F4, F4_sp)
    append!(F5, F5_sp)
    append!(F6, F6_sp)
    append!(F7, F7_sp)
    append!(F8, F8_sp)
    append!(F9, F9_sp)
    append!(F10, F10_sp)
    append!(Fr1, Fr1_sp)
    append!(Fr2, Fr2_sp)

    append!(Q1, Q1_sp)
    append!(Q2, Q2_sp)
    append!(Q3, Q3_sp)
    append!(Q4, Q4_sp)
    append!(Q5, Q5_sp)


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

            # Set state variable guesses to behavior associated with these
            for i = 1:Nd
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
                tspan = (0.0, dtd)
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


mhmpc_sim()

decision = zeros(P)
for i = 1:P
    if distr_PI_vec[i] <= centr_PI_vec[i]
        decision[i] = 1
    end
end

print(decision)
plot(decision)
plot(log10.(centr_PI_vec[1:P]), label="Log10 Centralized PI")
plot!(log10.(distr_PI_vec[1:P]), label="Distributed", ylabel="Log10 PI")

plot(predict.T3)
hline!([T3_sp])

PI_diff = centr_PI_vec - distr_PI_vec
plot(centr_time, label="Centralized time", ylabel="Seconds")
plot!(distr_time, label="Distributed time")

df = DataFrame(PIc=ex.PIc, ISEc=ex.ISEc, ISCc=ex.ISCc,
    PId=ex.PId, ISEd=ex.ISEd, ISCd=ex.ISCd,
    V1=ex.V1, V2=ex.V2, V3=ex.V3, V4=ex.V4, V5=ex.V5,
    T1=ex.T1, T2=ex.T2, T3=ex.T3, T4=ex.T4, T5=ex.T5,
    CA1=ex.CA1, CA2=ex.CA2, CA3=ex.CA3, CA4=ex.CA4, CA5=ex.CA5,
    CB1=ex.CB1, CB2=ex.CB2, CB3=ex.CB3, CB4=ex.CB4, CB5=ex.CB5,
    CC1=ex.CC1, CC2=ex.CC2, CC3=ex.CC3, CC4=ex.CC4, CC5=ex.CC5,
    CD1=ex.CD1, CD2=ex.CD2, CD3=ex.CD3, CD4=ex.CD4, CD5=ex.CD5,
    F1d=ex.F1d, F2d=ex.F2d, F3d=ex.F3d, F4d=ex.F4d, F5d=ex.F5d,
    F6d=ex.F6d, F7d=ex.F7d, F8d=ex.F8d, F9d=ex.F9d, F10d=ex.F10d,
    Fr1d=ex.Fr1d, Fr2d=ex.Fr2d,
    Q1d=ex.Q1d, Q2d=ex.Q2d, Q3d=ex.Q3d, Q4d=ex.Q4d, Q5d=ex.Q5d,
    F1c=ex.F1c, F2c=ex.F2c, F3c=ex.F3c, F4c=ex.F4c, F5c=ex.F5c,
    F6c=ex.F6c, F7c=ex.F7c, F8c=ex.F8c, F9c=ex.F9c, F10c=ex.F10c,
    Fr1c=ex.Fr1c, Fr2c=ex.Fr2c,
    Q1c=ex.Q1c, Q2c=ex.Q2c, Q3c=ex.Q3c, Q4c=ex.Q4c, Q5c=ex.Q5c
)
# CSV.write("mhmpc-src-results-$(mhmpc_src_instance)-t-5-sec.csv", df)

# global mhmpc_src_instance = mhmpc_src_instance + 1

# end

# global mhmpc_src_instance = 1

# while mhmpc_src_instance <= 104

#     N = 1200  # Control horizon
#     dt = 1 # Sampling time of the system
#     Nd = 40
#     dtd = 30
#     P = 80

#     global s_path = 0.01
#     global cpu_max = 7.0 # Maximum cpu time for Ipopt
#     global dual_inf_tol = Float64(1 * 10^(0))
#     global opt_tol = Float64(1 * 10^(0))
#     global constr_viol_tol = Float64(1 * 10^(0))
#     global compl_inf_tol = Float64(1 * 10^(0))

#     global c_cpu_max = 7.0 # Maximum cpu time for Ipopt
#     global c_dual_inf_tol = Float64(1 * 10^(0))
#     global c_opt_tol = Float64(1 * 10^(0))
#     global c_constr_viol_tol = Float64(1 * 10^(0))
#     global c_compl_inf_tol = Float64(1 * 10^(0))

#     block_size = Int(dtd/1) - 1
#     k_indices = Int[]  # Initialize an empty array

#     # Loop to generate the pattern until N - 1
#     for start_k in 0:dtd:N-1
#         append!(k_indices, start_k:start_k+block_size-1)
#     end

#     # append!(k_indices, N-1)
#     # Ensure indices do not exceed N - 1
#     global k_indices = filter(x -> x < N, k_indices)

#     mutable struct Weights
#         v::Float64

#         t1::Float64
#         t2::Float64
#         t3::Float64
#         t4::Float64
#         t5::Float64

#         ca1::Float64
#         ca2::Float64
#         ca3::Float64
#         ca4::Float64
#         ca5::Float64

#         cb1::Float64
#         cb2::Float64
#         cb3::Float64
#         cb4::Float64
#         cb5::Float64

#         cc1::Float64
#         cc2::Float64
#         cc3::Float64
#         cc4::Float64
#         cc5::Float64

#         cd1::Float64
#         cd2::Float64
#         cd3::Float64
#         cd4::Float64
#         cd5::Float64

#         f1::Float64
#         f2::Float64
#         f3::Float64
#         f4::Float64
#         f5::Float64
#         f6::Float64
#         f7::Float64
#         f8::Float64
#         f9::Float64
#         f10::Float64
#         fr1::Float64
#         fr2::Float64

#         q1::Float64
#         q2::Float64
#         q3::Float64
#         q4::Float64
#         q5::Float64

#     end

#     mutable struct Decomposition_Trajectory


#         V1::Vector{Float64}
#         V2::Vector{Float64}
#         V3::Vector{Float64}
#         V4::Vector{Float64}
#         V5::Vector{Float64}

#         T1::Vector{Float64}
#         T2::Vector{Float64}
#         T3::Vector{Float64}
#         T4::Vector{Float64}
#         T5::Vector{Float64}

#         CA1::Vector{Float64}
#         CB1::Vector{Float64}
#         CC1::Vector{Float64}
#         CD1::Vector{Float64}

#         CA2::Vector{Float64}
#         CB2::Vector{Float64}
#         CC2::Vector{Float64}
#         CD2::Vector{Float64}

#         CA3::Vector{Float64}
#         CB3::Vector{Float64}
#         CC3::Vector{Float64}
#         CD3::Vector{Float64}

#         CA4::Vector{Float64}
#         CB4::Vector{Float64}
#         CC4::Vector{Float64}
#         CD4::Vector{Float64}

#         CA5::Vector{Float64}
#         CB5::Vector{Float64}
#         CC5::Vector{Float64}
#         CD5::Vector{Float64}

#         F1::Vector{Float64}
#         F2::Vector{Float64}
#         F3::Vector{Float64}
#         F4::Vector{Float64}
#         F5::Vector{Float64}
#         F6::Vector{Float64}
#         F7::Vector{Float64}
#         F8::Vector{Float64}
#         F9::Vector{Float64}
#         F10::Vector{Float64}
#         Fr1::Vector{Float64}
#         Fr2::Vector{Float64}
#         Q1::Vector{Float64}
#         Q2::Vector{Float64}
#         Q3::Vector{Float64}
#         Q4::Vector{Float64}
#         Q5::Vector{Float64}

#     end

#     mutable struct Solve_Strat_Comparison_Trajectory

#         V1::Vector{Float64}
#         V2::Vector{Float64}
#         V3::Vector{Float64}
#         V4::Vector{Float64}
#         V5::Vector{Float64}

#         T1::Vector{Float64}
#         T2::Vector{Float64}
#         T3::Vector{Float64}
#         T4::Vector{Float64}
#         T5::Vector{Float64}

#         CA1::Vector{Float64}
#         CB1::Vector{Float64}
#         CC1::Vector{Float64}
#         CD1::Vector{Float64}

#         CA2::Vector{Float64}
#         CB2::Vector{Float64}
#         CC2::Vector{Float64}
#         CD2::Vector{Float64}

#         CA3::Vector{Float64}
#         CB3::Vector{Float64}
#         CC3::Vector{Float64}
#         CD3::Vector{Float64}

#         CA4::Vector{Float64}
#         CB4::Vector{Float64}
#         CC4::Vector{Float64}
#         CD4::Vector{Float64}

#         CA5::Vector{Float64}
#         CB5::Vector{Float64}
#         CC5::Vector{Float64}
#         CD5::Vector{Float64}

#         F1c::Vector{Float64}
#         F2c::Vector{Float64}
#         F3c::Vector{Float64}
#         F4c::Vector{Float64}
#         F5c::Vector{Float64}
#         F6c::Vector{Float64}
#         F7c::Vector{Float64}
#         F8c::Vector{Float64}
#         F9c::Vector{Float64}
#         F10c::Vector{Float64}
#         Fr1c::Vector{Float64}
#         Fr2c::Vector{Float64}
#         Q1c::Vector{Float64}
#         Q2c::Vector{Float64}
#         Q3c::Vector{Float64}
#         Q4c::Vector{Float64}
#         Q5c::Vector{Float64}

#         F1d::Vector{Float64}
#         F2d::Vector{Float64}
#         F3d::Vector{Float64}
#         F4d::Vector{Float64}
#         F5d::Vector{Float64}
#         F6d::Vector{Float64}
#         F7d::Vector{Float64}
#         F8d::Vector{Float64}
#         F9d::Vector{Float64}
#         F10d::Vector{Float64}
#         Fr1d::Vector{Float64}
#         Fr2d::Vector{Float64}
#         Q1d::Vector{Float64}
#         Q2d::Vector{Float64}
#         Q3d::Vector{Float64}
#         Q4d::Vector{Float64}
#         Q5d::Vector{Float64}

#         PIc::Vector{Float64}
#         ISEc::Vector{Float64}
#         ISCc::Vector{Float64}

#         PId::Vector{Float64}
#         ISEd::Vector{Float64}
#         ISCd::Vector{Float64}
#     end

#     ex = Solve_Strat_Comparison_Trajectory(zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P))


#     data = CSV.read("C:\\Users\\escooper\\git\\bae\\final-formulation\\src-data.csv", DataFrame)
#     instance = data[mhmpc_src_instance, :]

#     # Declare setpoint, initial condition, parameters
#     function init_sp_and_params()

#         global V1_init = instance[1]
#         global V2_init = instance[2]
#         global V3_init = instance[3]
#         global V4_init = instance[4]
#         global V5_init = instance[5]

#         global T1_init = instance[6]
#         global T2_init = instance[7]
#         global T3_init = instance[8]
#         global T4_init = instance[9]
#         global T5_init = instance[10]

#         global CA1_init = instance[11]
#         global CB1_init = instance[16]
#         global CC1_init = instance[21]
#         global CD1_init = instance[26]

#         global CA2_init = instance[12]
#         global CB2_init = instance[17]
#         global CC2_init = instance[22]
#         global CD2_init = instance[27]

#         global CA3_init = instance[13]
#         global CB3_init = instance[18]
#         global CC3_init = instance[23]
#         global CD3_init = instance[28]

#         global CA4_init = instance[14]
#         global CB4_init = instance[19]
#         global CC4_init = instance[24]
#         global CD4_init = instance[29]

#         global CA5_init = instance[15]
#         global CB5_init = instance[20]
#         global CC5_init = instance[25]
#         global CD5_init = instance[30]

#         global V1_sp = instance[1]
#         global V2_sp = instance[2]
#         global V3_sp = instance[3]
#         global V4_sp = instance[4]
#         global V5_sp = instance[5]

#         global T1_sp = instance[31]
#         global T2_sp = instance[32]
#         global T3_sp = instance[33]
#         global T4_sp = instance[34]
#         global T5_sp = instance[35]

#         global CA1_sp = instance[36]
#         global CA2_sp = instance[37]
#         global CA3_sp = instance[38]
#         global CA4_sp = instance[39]
#         global CA5_sp = instance[40]

#         global CB1_sp = instance[41]
#         global CB2_sp = instance[42]
#         global CB3_sp = instance[43]
#         global CB4_sp = instance[44]
#         global CB5_sp = instance[45]

#         global CC1_sp = instance[46]
#         global CC2_sp = instance[47]
#         global CC3_sp = instance[48]
#         global CC4_sp = instance[49]
#         global CC5_sp = instance[50]

#         global CD1_sp = instance[51]
#         global CD2_sp = instance[52]
#         global CD3_sp = instance[53]
#         global CD4_sp = instance[54]
#         global CD5_sp = instance[55]


#         global F1_sp = instance[56]
#         global F2_sp = instance[57]
#         global F3_sp = instance[58]
#         global F4_sp = instance[59]
#         global F5_sp = instance[60]
#         global F6_sp = instance[61]
#         global F7_sp = instance[62]
#         global F8_sp = instance[63]
#         global F9_sp = instance[64]
#         global F10_sp = instance[65]
#         global Fr1_sp = instance[66]
#         global Fr2_sp = instance[67]

#         global Q1_sp = instance[68]
#         global Q2_sp = instance[69]
#         global Q3_sp = instance[70]
#         global Q4_sp = instance[71]
#         global Q5_sp = instance[72]

#         global H_vap_A = 3.073e4
#         global H_vap_B = 1.35e4
#         global H_vap_C = 4.226e4
#         global H_vap_D = 4.55e4

#         global H_ref_A = 7.44e4
#         global H_ref_B = 5.91e4
#         global H_ref_C = 2.01e4
#         global H_ref_D = -2.89e4

#         global Cp_A = 184.6
#         global Cp_B = 59.1
#         global Cp_C = 247
#         global Cp_D = 301.3

#         global CA0 = 1.126e4
#         global CB0 = 2.028e4
#         global CC0 = 8174
#         global CD0 = 6485

#         global Tref = 450
#         global TA0 = 473
#         global TB0 = 473
#         global TD0 = 473


#         global delH_r1 = -1.53e5
#         global delH_r2 = -1.118e5
#         global delH_r3 = 4.141e5

#         global R = 8.314
#     end


#     init_sp_and_params()

#     w = Weights(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

#     # Linear dependence of enthalpies
#     H_A(T) = H_ref_A + Cp_A * (T - Tref)
#     H_B(T) = H_ref_B + Cp_B * (T - Tref)
#     H_C(T) = H_ref_C + Cp_C * (T - Tref)
#     H_D(T) = H_ref_D + Cp_D * (T - Tref)

#     kEB2(T) = 0.152 * exp(-3933 / (R * T))
#     kEB3(T) = 0.490 * exp(-50870 / (R * T))

#     # Volatilities of the species in the seperator
#     alpha_A(T) = 0.0449 * T + 10
#     alpha_B(T) = 0.0260 * T + 10
#     alpha_C(T) = 0.0065 * T + 0.5
#     alpha_D(T) = 0.0058 * T + 0.25

#     # Debugging parameters for dual feasibility:
#     ub_f = 3.0
#     lb_f = 0.1

#     # Reaction rate expressions
#     r1(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * CA^(0.32) * CB^(1.5)
#     r2(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * CB^(2.5) * CC^(0.5)) / (1 + kEB2(T) * CD)
#     r3(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * CA^(1.0218) * CD) / (1 + kEB3(T) * CA)

#     # Reaction rate expressions
#     r1_(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * max(1e-6, CA)^(0.32) * max(1e-6, CB)^(1.5)
#     r2_(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * max(1e-6, CB)^(2.5) * max(1e-6, CC)^(0.5)) / (1 + kEB2(T) * CD)
#     r3_(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * max(1e-6, CA)^(1.0218) * CD) / (1 + kEB3(T) * CA)

#     # Molar flow in the overhead stream
#     MA(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_A(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
#     MB(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_B(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
#     MC(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_C(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
#     MD(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_D(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))

#     # Concentration of species in the recycle stream
#     CAr(MA, MB, MC, MD) = MA / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
#     CBr(MA, MB, MC, MD) = MB / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
#     CCr(MA, MB, MC, MD) = MC / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
#     CDr(MA, MB, MC, MD) = MD / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))

#     fix = Decomposition_Trajectory(zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1))
#     ig = Decomposition_Trajectory(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))

#     # Initialize fix structure for decomp
#     for set_traj = 1

#         # Set control variables to nominal steady state values
#         fix.F1 .= F1_sp
#         fix.F2 .= F2_sp
#         fix.F3 .= F3_sp
#         fix.F4 .= F4_sp
#         fix.F5 .= F5_sp
#         fix.F6 .= F6_sp
#         fix.F7 .= F7_sp
#         fix.F8 .= F8_sp
#         fix.F9 .= F9_sp
#         fix.F10 .= F10_sp
#         fix.Fr1 .= Fr1_sp
#         fix.Fr2 .= Fr2_sp

#         fix.Q1 .= Q1_sp
#         fix.Q2 .= Q2_sp
#         fix.Q3 .= Q3_sp
#         fix.Q4 .= Q4_sp
#         fix.Q5 .= Q5_sp

#         fix.V1[1] = V1_init
#         fix.V2[1] = V2_init
#         fix.V3[1] = V3_init
#         fix.V4[1] = V4_init
#         fix.V5[1] = V5_init
#         fix.T1[1] = T1_init
#         fix.T2[1] = T2_init
#         fix.T3[1] = T3_init
#         fix.T4[1] = T4_init
#         fix.T5[1] = T5_init

#         fix.CA1[1] = CA1_init
#         fix.CA2[1] = CA2_init
#         fix.CA3[1] = CA3_init
#         fix.CA4[1] = CA4_init
#         fix.CA5[1] = CA5_init
#         fix.CB1[1] = CB1_init
#         fix.CB2[1] = CB2_init
#         fix.CB3[1] = CB3_init
#         fix.CB4[1] = CB4_init
#         fix.CB5[1] = CB5_init
#         fix.CC1[1] = CC1_init
#         fix.CC2[1] = CC2_init
#         fix.CC3[1] = CC3_init
#         fix.CC4[1] = CC4_init
#         fix.CC5[1] = CC5_init
#         fix.CD1[1] = CD1_init
#         fix.CD2[1] = CD2_init
#         fix.CD3[1] = CD3_init
#         fix.CD4[1] = CD4_init
#         fix.CD5[1] = CD5_init

#         for i = 1:Nd
#             global k = i

#             F1 = fix.F1[k]
#             F2 = fix.F2[k]
#             F3 = fix.F3[k]
#             F4 = fix.F4[k]
#             F5 = fix.F5[k]
#             F6 = fix.F6[k]
#             F7 = fix.F7[k]
#             F8 = fix.F8[k]
#             F9 = fix.F9[k]
#             F10 = fix.F10[k]
#             Fr1 = fix.Fr1[k]
#             Fr2 = fix.Fr2[k]
#             Q1 = fix.Q1[k]
#             Q2 = fix.Q2[k]
#             Q3 = fix.Q3[k]
#             Q4 = fix.Q4[k]
#             Q5 = fix.Q5[k]

#             V1 = fix.V1[k]
#             V2 = fix.V2[k]
#             V3 = fix.V3[k]
#             V4 = fix.V4[k]
#             V5 = fix.V5[k]

#             T1 = fix.T1[k]
#             T2 = fix.T2[k]
#             T3 = fix.T3[k]
#             T4 = fix.T4[k]
#             T5 = fix.T5[k]

#             CA1 = fix.CA1[k]
#             CB1 = fix.CB1[k]
#             CC1 = fix.CC1[k]
#             CD1 = fix.CD1[k]

#             CA2 = fix.CA2[k]
#             CB2 = fix.CB2[k]
#             CC2 = fix.CC2[k]
#             CD2 = fix.CD2[k]

#             CA3 = fix.CA3[k]
#             CB3 = fix.CB3[k]
#             CC3 = fix.CC3[k]
#             CD3 = fix.CD3[k]

#             CA4 = fix.CA4[k]
#             CB4 = fix.CB4[k]
#             CC4 = fix.CC4[k]
#             CD4 = fix.CD4[k]

#             CA5 = fix.CA5[k]
#             CB5 = fix.CB5[k]
#             CC5 = fix.CC5[k]
#             CD5 = fix.CD5[k]

#             x0 = [fix.V1[k], fix.V2[k], fix.V3[k], fix.V4[k], fix.V5[k], fix.T1[k], fix.T2[k], fix.T3[k], fix.T4[k], fix.T5[k], fix.CA1[k], fix.CB1[k], fix.CC1[k], fix.CD1[k], fix.CA2[k], fix.CB2[k], fix.CC2[k], fix.CD2[k], fix.CA3[k], fix.CB3[k], fix.CC3[k], fix.CD3[k], fix.CA4[k], fix.CB4[k], fix.CC4[k], fix.CD4[k], fix.CA5[k], fix.CB5[k], fix.CC5[k], fix.CD5[k]]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1 + F2 + Fr2 - F3
#                 dV2 = F3 + F4 - F5
#                 dV3 = F5 + F6 - F7
#                 dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                 dV5 = F10 + Fr1 - F9

#                 dT1 = (
#                     ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                       (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                       (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                       (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                       (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2 + F4 * CB0 * H_B(TB0) +
#                      (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                      (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                      (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                      (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

#                 dT3 = (
#                     ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                       (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                       (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                       (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4
#                         + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5 +
#                       F10 * CD0 * H_D(TD0)
#                       + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                       + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                       + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                       + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CA1)
#                      /
#                      V1) - r1(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CB1) /
#                      V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CC1) /
#                      V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CD1) /
#                      V1) + r2(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                 dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                 dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                 dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                 dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                 dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                 dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                 dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                 dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dtd)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23())

#             # Obtain the next values for each element
#             fix.V1[k+1] = soln.u[end][1]
#             fix.V2[k+1] = soln.u[end][2]
#             fix.V3[k+1] = soln.u[end][3]
#             fix.V4[k+1] = soln.u[end][4]
#             fix.V5[k+1] = soln.u[end][5]

#             fix.T1[k+1] = soln.u[end][6]
#             fix.T2[k+1] = soln.u[end][7]
#             fix.T3[k+1] = soln.u[end][8]
#             fix.T4[k+1] = soln.u[end][9]
#             fix.T5[k+1] = soln.u[end][10]

#             fix.CA1[k+1] = soln.u[end][11]
#             fix.CB1[k+1] = soln.u[end][12]
#             fix.CC1[k+1] = soln.u[end][13]
#             fix.CD1[k+1] = soln.u[end][14]

#             fix.CA2[k+1] = soln.u[end][15]
#             fix.CB2[k+1] = soln.u[end][16]
#             fix.CC2[k+1] = soln.u[end][17]
#             fix.CD2[k+1] = soln.u[end][18]

#             fix.CA3[k+1] = soln.u[end][19]
#             fix.CB3[k+1] = soln.u[end][20]
#             fix.CC3[k+1] = soln.u[end][21]
#             fix.CD3[k+1] = soln.u[end][22]

#             fix.CA4[k+1] = soln.u[end][23]
#             fix.CB4[k+1] = soln.u[end][24]
#             fix.CC4[k+1] = soln.u[end][25]
#             fix.CD4[k+1] = soln.u[end][26]

#             fix.CA5[k+1] = soln.u[end][27]
#             fix.CB5[k+1] = soln.u[end][28]
#             fix.CC5[k+1] = soln.u[end][29]
#             fix.CD5[k+1] = soln.u[end][30]
#         end

#     end

#     # Initialize ig structure for centralized
#     for set_traj = 1

#         # Set control variables to nominal steady state values
#         ig.F1 .= F1_sp
#         ig.F2 .= F2_sp
#         ig.F3 .= F3_sp
#         ig.F4 .= F4_sp
#         ig.F5 .= F5_sp
#         ig.F6 .= F6_sp
#         ig.F7 .= F7_sp
#         ig.F8 .= F8_sp
#         ig.F9 .= F9_sp
#         ig.F10 .= F10_sp
#         ig.Fr1 .= Fr1_sp
#         ig.Fr2 .= Fr2_sp

#         ig.Q1 .= Q1_sp
#         ig.Q2 .= Q2_sp
#         ig.Q3 .= Q3_sp
#         ig.Q4 .= Q4_sp
#         ig.Q5 .= Q5_sp

#         ig.V1[1] = V1_init
#         ig.V2[1] = V2_init
#         ig.V3[1] = V3_init
#         ig.V4[1] = V4_init
#         ig.V5[1] = V5_init
#         ig.T1[1] = T1_init
#         ig.T2[1] = T2_init
#         ig.T3[1] = T3_init
#         ig.T4[1] = T4_init
#         ig.T5[1] = T5_init

#         ig.CA1[1] = CA1_init
#         ig.CA2[1] = CA2_init
#         ig.CA3[1] = CA3_init
#         ig.CA4[1] = CA4_init
#         ig.CA5[1] = CA5_init
#         ig.CB1[1] = CB1_init
#         ig.CB2[1] = CB2_init
#         ig.CB3[1] = CB3_init
#         ig.CB4[1] = CB4_init
#         ig.CB5[1] = CB5_init
#         ig.CC1[1] = CC1_init
#         ig.CC2[1] = CC2_init
#         ig.CC3[1] = CC3_init
#         ig.CC4[1] = CC4_init
#         ig.CC5[1] = CC5_init
#         ig.CD1[1] = CD1_init
#         ig.CD2[1] = CD2_init
#         ig.CD3[1] = CD3_init
#         ig.CD4[1] = CD4_init
#         ig.CD5[1] = CD5_init

#         for i = 1:N
#             global k = i

#             F1 = ig.F1[k]
#             F2 = ig.F2[k]
#             F3 = ig.F3[k]
#             F4 = ig.F4[k]
#             F5 = ig.F5[k]
#             F6 = ig.F6[k]
#             F7 = ig.F7[k]
#             F8 = ig.F8[k]
#             F9 = ig.F9[k]
#             F10 = ig.F10[k]
#             Fr1 = ig.Fr1[k]
#             Fr2 = ig.Fr2[k]
#             Q1 = ig.Q1[k]
#             Q2 = ig.Q2[k]
#             Q3 = ig.Q3[k]
#             Q4 = ig.Q4[k]
#             Q5 = ig.Q5[k]

#             V1 = ig.V1[k]
#             V2 = ig.V2[k]
#             V3 = ig.V3[k]
#             V4 = ig.V4[k]
#             V5 = ig.V5[k]

#             T1 = ig.T1[k]
#             T2 = ig.T2[k]
#             T3 = ig.T3[k]
#             T4 = ig.T4[k]
#             T5 = ig.T5[k]

#             CA1 = ig.CA1[k]
#             CB1 = ig.CB1[k]
#             CC1 = ig.CC1[k]
#             CD1 = ig.CD1[k]

#             CA2 = ig.CA2[k]
#             CB2 = ig.CB2[k]
#             CC2 = ig.CC2[k]
#             CD2 = ig.CD2[k]

#             CA3 = ig.CA3[k]
#             CB3 = ig.CB3[k]
#             CC3 = ig.CC3[k]
#             CD3 = ig.CD3[k]

#             CA4 = ig.CA4[k]
#             CB4 = ig.CB4[k]
#             CC4 = ig.CC4[k]
#             CD4 = ig.CD4[k]

#             CA5 = ig.CA5[k]
#             CB5 = ig.CB5[k]
#             CC5 = ig.CC5[k]
#             CD5 = ig.CD5[k]

#             x0 = [ig.V1[k], ig.V2[k], ig.V3[k], ig.V4[k], ig.V5[k], ig.T1[k], ig.T2[k], ig.T3[k], ig.T4[k], ig.T5[k], ig.CA1[k], ig.CB1[k], ig.CC1[k], ig.CD1[k], ig.CA2[k], ig.CB2[k], ig.CC2[k], ig.CD2[k], ig.CA3[k], ig.CB3[k], ig.CC3[k], ig.CD3[k], ig.CA4[k], ig.CB4[k], ig.CC4[k], ig.CD4[k], ig.CA5[k], ig.CB5[k], ig.CC5[k], ig.CD5[k]]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1 + F2 + Fr2 - F3
#                 dV2 = F3 + F4 - F5
#                 dV3 = F5 + F6 - F7
#                 dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                 dV5 = F10 + Fr1 - F9

#                 dT1 = (
#                     ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                       (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                       (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                       (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                       (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2 + F4 * CB0 * H_B(TB0) +
#                      (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                      (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                      (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                      (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

#                 dT3 = (
#                     ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                       (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                       (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                       (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4
#                         + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5 +
#                       F10 * CD0 * H_D(TD0)
#                       + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                       + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                       + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                       + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CA1)
#                      /
#                      V1) - r1(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CB1) /
#                      V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CC1) /
#                      V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CD1) /
#                      V1) + r2(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                 dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                 dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                 dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                 dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                 dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                 dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                 dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                 dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dt)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23())

#             # Obtain the next values for each element
#             ig.V1[k+1] = soln.u[end][1]
#             ig.V2[k+1] = soln.u[end][2]
#             ig.V3[k+1] = soln.u[end][3]
#             ig.V4[k+1] = soln.u[end][4]
#             ig.V5[k+1] = soln.u[end][5]

#             ig.T1[k+1] = soln.u[end][6]
#             ig.T2[k+1] = soln.u[end][7]
#             ig.T3[k+1] = soln.u[end][8]
#             ig.T4[k+1] = soln.u[end][9]
#             ig.T5[k+1] = soln.u[end][10]

#             ig.CA1[k+1] = soln.u[end][11]
#             ig.CB1[k+1] = soln.u[end][12]
#             ig.CC1[k+1] = soln.u[end][13]
#             ig.CD1[k+1] = soln.u[end][14]

#             ig.CA2[k+1] = soln.u[end][15]
#             ig.CB2[k+1] = soln.u[end][16]
#             ig.CC2[k+1] = soln.u[end][17]
#             ig.CD2[k+1] = soln.u[end][18]

#             ig.CA3[k+1] = soln.u[end][19]
#             ig.CB3[k+1] = soln.u[end][20]
#             ig.CC3[k+1] = soln.u[end][21]
#             ig.CD3[k+1] = soln.u[end][22]

#             ig.CA4[k+1] = soln.u[end][23]
#             ig.CB4[k+1] = soln.u[end][24]
#             ig.CC4[k+1] = soln.u[end][25]
#             ig.CD4[k+1] = soln.u[end][26]

#             ig.CA5[k+1] = soln.u[end][27]
#             ig.CB5[k+1] = soln.u[end][28]
#             ig.CC5[k+1] = soln.u[end][29]
#             ig.CD5[k+1] = soln.u[end][30]
#         end

#     end

#     function cmpc()

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin
#             # State variables

#             # Volume, state, [=] m3
#             V1[k=0:N], (lower_bound=0.2 * V1_sp, upper_bound=1.8 * V1_sp)
#             V2[k=0:N], (lower_bound=0.2 * V2_sp, upper_bound=1.8 * V2_sp)
#             V3[k=0:N], (lower_bound=0.2 * V3_sp, upper_bound=1.8 * V3_sp)
#             V4[k=0:N], (lower_bound=0.2 * V4_sp, upper_bound=1.8 * V4_sp)
#             V5[k=0:N], (lower_bound=0.2 * V5_sp, upper_bound=1.8 * V5_sp)

#             # Temperature, state, [=] K
#             T1[k=0:N], (lower_bound=0.2 * T1_sp, upper_bound=1.8 * T1_sp)
#             T2[k=0:N], (lower_bound=0.2 * T2_sp, upper_bound=1.8 * T2_sp)
#             T3[k=0:N], (lower_bound=0.2 * T3_sp, upper_bound=1.8 * T3_sp)
#             T4[k=0:N], (lower_bound=0.2 * T4_sp, upper_bound=1.8 * T4_sp)
#             T5[k=0:N], (lower_bound=0.2 * T5_sp, upper_bound=1.8 * T5_sp)

#             # Concentration, state, [=] mol/m3
#             CA1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             CB1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             CC1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             CD1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp)
#             F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp)
#             F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp)
#             F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp)
#             F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp)
#             F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp)
#             F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp)
#             F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp)
#             F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp)
#             F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp)

#             Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp)
#             Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp)

#             Q1[k=0:N], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
#             Q2[k=0:N], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)
#             Q3[k=0:N], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)
#             Q4[k=0:N], (start=Q4_sp, upper_bound=1.8 * Q4_sp, lower_bound=1e-6)
#             Q5[k=0:N], (start=Q5_sp, upper_bound=1.8 * Q5_sp, lower_bound=1e-6)

#         end

#         for k = 0:N
#             set_start_value(V1[k], ig.V1[k+1])
#             set_start_value(V2[k], ig.V2[k+1])
#             set_start_value(V3[k], ig.V3[k+1])
#             set_start_value(V4[k], ig.V4[k+1])
#             set_start_value(V5[k], ig.V5[k+1])

#             set_start_value(T1[k], ig.T1[k+1])
#             set_start_value(T2[k], ig.T2[k+1])
#             set_start_value(T3[k], ig.T3[k+1])
#             set_start_value(T4[k], ig.T4[k+1])
#             set_start_value(T5[k], ig.T5[k+1])

#             set_start_value(CA1[k], ig.CA1[k+1])
#             set_start_value(CA2[k], ig.CA2[k+1])
#             set_start_value(CA3[k], ig.CA3[k+1])
#             set_start_value(CA4[k], ig.CA4[k+1])
#             set_start_value(CA5[k], ig.CA5[k+1])

#             set_start_value(CB1[k], ig.CB1[k+1])
#             set_start_value(CB2[k], ig.CB2[k+1])
#             set_start_value(CB3[k], ig.CB3[k+1])
#             set_start_value(CB4[k], ig.CB4[k+1])
#             set_start_value(CB5[k], ig.CB5[k+1])

#             set_start_value(CC1[k], ig.CC1[k+1])
#             set_start_value(CC2[k], ig.CC2[k+1])
#             set_start_value(CC3[k], ig.CC3[k+1])
#             set_start_value(CC4[k], ig.CC4[k+1])
#             set_start_value(CC5[k], ig.CC5[k+1])

#             set_start_value(CD1[k], ig.CD1[k+1])
#             set_start_value(CD2[k], ig.CD2[k+1])
#             set_start_value(CD3[k], ig.CD3[k+1])
#             set_start_value(CD4[k], ig.CD4[k+1])
#             set_start_value(CD5[k], ig.CD5[k+1])

#             set_start_value(F1[k], F1_sp)
#             set_start_value(F2[k], F2_sp)
#             set_start_value(F3[k], F3_sp)
#             set_start_value(F4[k], F4_sp)
#             set_start_value(F5[k], F5_sp)
#             set_start_value(F6[k], F6_sp)
#             set_start_value(F7[k], F7_sp)
#             set_start_value(F8[k], F8_sp)
#             set_start_value(F9[k], F9_sp)
#             set_start_value(F10[k], F10_sp)
#             set_start_value(Fr1[k], Fr1_sp)
#             set_start_value(Fr2[k], Fr2_sp)

#         end

#         for k = 0:N
#             JuMP.fix(Q1[k], Q1_sp; force=true)
#             JuMP.fix(Q2[k], Q2_sp; force=true)
#             JuMP.fix(Q3[k], Q3_sp; force=true)
#             JuMP.fix(Q4[k], Q4_sp; force=true)
#             JuMP.fix(Q5[k], Q5_sp; force=true)

#             JuMP.fix(F7[k], F7_sp; force=true)
#             # JuMP.fix(F8[k], F8_sp; force=true)
#             # JuMP.fix(F9[k], F9_sp; force=true)
#             # JuMP.fix(F10[k], F10_sp; force=true)
#             # JuMP.fix(Fr1[k], Fr1_sp; force=true)
#             JuMP.fix(Fr2[k], Fr2_sp; force=true)
#         end

#         @constraints mpc begin
#             # Initial condition
#             V1_inital, V1[0] == V1_init
#             V2_inital, V2[0] == V2_init
#             V3_inital, V3[0] == V3_init
#             V4_inital, V4[0] == V4_init
#             V5_inital, V5[0] == V5_init

#             T1_inital, T1[0] == T1_init
#             T2_inital, T2[0] == T2_init
#             T3_inital, T3[0] == T3_init
#             T4_inital, T4[0] == T4_init
#             T5_inital, T5[0] == T5_init

#             CA1_initial, CA1[0] == CA1_init
#             CA2_initial, CA2[0] == CA2_init
#             CA3_initial, CA3[0] == CA3_init
#             CA4_initial, CA4[0] == CA4_init
#             CA5_initial, CA5[0] == CA5_init

#             CB1_initial, CB1[0] == CB1_init
#             CB2_initial, CB2[0] == CB2_init
#             CB3_initial, CB3[0] == CB3_init
#             CB4_initial, CB4[0] == CB4_init
#             CB5_initial, CB5[0] == CB5_init

#             CC1_initial, CC1[0] == CC1_init
#             CC2_initial, CC2[0] == CC2_init
#             CC3_initial, CC3[0] == CC3_init
#             CC4_initial, CC4[0] == CC4_init
#             CC5_initial, CC5[0] == CC5_init

#             CD1_initial, CD1[0] == CD1_init
#             CD2_initial, CD2[0] == CD2_init
#             CD3_initial, CD3[0] == CD3_init
#             CD4_initial, CD4[0] == CD4_init
#             CD5_initial, CD5[0] == CD5_init

#             F1_hold[k in k_indices], F1[k] == F1[k+1]
#             F2_hold[k in k_indices], F2[k] == F2[k+1]
#             F3_hold[k in k_indices], F3[k] == F3[k+1]
#             F4_hold[k in k_indices], F4[k] == F4[k+1]
#             F5_hold[k in k_indices], F5[k] == F5[k+1]
#             F6_hold[k in k_indices], F6[k] == F6[k+1]
#             F7_hold[k in k_indices], F7[k] == F7[k+1]
#             F8_hold[k in k_indices], F8[k] == F8[k+1]
#             F9_hold[k in k_indices], F9[k] == F9[k+1]
#             F10_hold[k in k_indices], F10[k] == F10[k+1]
#             Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
#             Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

#             Q1_hold[k in k_indices], Q1[k] == Q1[k+1]
#             Q2_hold[k in k_indices], Q2[k] == Q2[k+1]
#             Q3_hold[k in k_indices], Q3[k] == Q3[k+1]
#             Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
#             Q5_hold[k in k_indices], Q5[k] == Q5[k+1]


#         end

#         @NLconstraints mpc begin
#             # NLconstraints are the differential equations that describe the dynamics of the system
#             dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + Fr2[k] - F3[k]) * dt == V1_sp

#             dT1_dt[k=0:N-1], T1[k] + (
#                 ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                   (Fr2[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F3[k] * CA1[k] * H_A(T1[k])) +
#                   (Fr2[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F3[k] * CB1[k] * H_B(T1[k])) +
#                   (Fr2[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F3[k] * CC1[k] * H_C(T1[k])) +
#                   (Fr2[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F3[k] * CD1[k] * H_D(T1[k])))
#                  /
#                  (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
#                 (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
#                 (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

#             dCA1_dt[k=0:N-1], CA1[k] + (
#                 ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CA1[k])
#                  /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

#             dCB1_dt[k=0:N-1], CB1[k] + (
#                 ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CB1[k]) /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

#             dCC1_dt[k=0:N-1], CC1[k] + (
#                 ((Fr2[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CC1[k]) /
#                  V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

#             dCD1_dt[k=0:N-1], CD1[k] + (
#                 ((Fr2[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CD1[k]) /
#                  V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

#             dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - F5[k]) * dt == V2_sp

#             dT2_dt[k=0:N-1], T2[k] + (
#                 (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                  (F3[k] * CA1[k] * H_A(T1[k]) - F5[k] * CA2[k] * H_A(T2[k])) +
#                  (F3[k] * CB1[k] * H_B(T1[k]) - F5[k] * CB2[k] * H_B(T2[k])) +
#                  (F3[k] * CC1[k] * H_C(T1[k]) - F5[k] * CC2[k] * H_C(T2[k])) +
#                  (F3[k] * CD1[k] * H_D(T1[k]) - F5[k] * CD2[k] * H_D(T2[k])))
#                 /
#                 (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
#                 +
#                 (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
#                 /
#                 (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


#             dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * CA1[k] - F5[k] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
#             dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - F5[k] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
#             dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * CC1[k] - F5[k] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
#             dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * CD1[k] - F5[k] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]

#             dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - F7[k]) * dt == V3_sp


#             dT3_dt[k=0:N-1], T3[k] + (
#                 ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2[k] * H_A(T2[k]) - F7[k] * CA3[k] * H_A(T3[k])) +
#                   (F5[k] * CB2[k] * H_B(T2[k]) - F7[k] * CB3[k] * H_B(T3[k])) +
#                   (F5[k] * CC2[k] * H_C(T2[k]) - F7[k] * CC3[k] * H_C(T3[k])) +
#                   (F5[k] * CD2[k] * H_D(T2[k]) - F7[k] * CD3[k] * H_D(T3[k])))
#                  /
#                  (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
#                 +
#                 (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
#                 /
#                 (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

#             dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * CA2[k] - F7[k] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
#             dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * CB2[k] + F6[k] * CB0 - F7[k] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
#             dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * CC2[k] - F7[k] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
#             dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * CD2[k] - F7[k] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

#             dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4_sp

#             dT4_dt[k=0:N-1], T4[k] +
#                              ((Q4[k]
#                                + (F7[k] * CA3[k] * H_A(T3[k]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
#                                + (F7[k] * CB3[k] * H_B(T3[k]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
#                                + (F7[k] * CC3[k] * H_C(T3[k]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
#                                + (F7[k] * CD3[k] * H_D(T3[k]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
#                               /
#                               (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

#             dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * CA3[k] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
#             dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * CB3[k] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
#             dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * CC3[k] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
#             dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * CD3[k] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

#             dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5_sp

#             dT5_dt[k=0:N-1], T5[k] + (
#                 ((Q5[k] +
#                   F10[k] * CD0 * H_D(TD0)
#                   + (Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
#                   + (Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
#                   + (Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
#                   + (Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
#                  /
#                  (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
#                 +
#                 ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
#                  /
#                  (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

#             dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
#             dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
#             dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
#             dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]

#             # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + Fr2[k] - F3[k] == -(V1[k] - V1_sp) / 200
#             # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / 200
#             # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / 200
#             # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
#             # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200

#         end


#         @NLobjective(mpc, Min, sum(
#             w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 + w.v * (V3[k] - V3_sp)^2 + w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
#             w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
#             w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
#             w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 + w.cb3 * (CB3[k] - CB3_sp)^2 + w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
#             w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 + w.cc3 * (CC3[k] - CC3_sp)^2 + w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
#             w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2 + w.cd3 * (CD3[k] - CD3_sp)^2 + w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
#             for k = 0:N)
#                                +
#                                sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 + w.f5 * (F5[k] - F5_sp)^2 +
#                                    w.f6 * (F6[k] - F6_sp)^2 + w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 + w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 + w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
#                                    w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2 for k = 0:N)
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", c_opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", c_dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", c_constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", c_compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", c_cpu_max)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global cmpc_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

#         F1 = Vector(JuMP.value.(F1))
#         F2 = Vector(JuMP.value.(F2))
#         F3 = Vector(JuMP.value.(F3))
#         F4 = Vector(JuMP.value.(F4))
#         Q1 = Vector(JuMP.value.(Q1))
#         Q2 = Vector(JuMP.value.(Q2))

#         F5 = Vector(JuMP.value.(F5))
#         F6 = Vector(JuMP.value.(F6))
#         Q3 = Vector(JuMP.value.(Q3))

#         F7 = Vector(JuMP.value.(F7))
#         F8 = Vector(JuMP.value.(F8))
#         F9 = Vector(JuMP.value.(F9))
#         F10 = Vector(JuMP.value.(F10))
#         Fr1 = Vector(JuMP.value.(Fr1))
#         Fr2 = Vector(JuMP.value.(Fr2))
#         Q4 = Vector(JuMP.value.(Q4))
#         Q5 = Vector(JuMP.value.(Q5))

#         return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5

#     end

#     function dmpc_1()

#         # Control variables: F1, F2, F3, F4, Q1, Q2
#         # State variables: All state variables associated with CSTR-1 and CSTR-2

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin

#             # State variables

#             # Volume, state, [=] m3
#             V1[k=0:Nd], (lower_bound=0.2 * V1_sp, upper_bound=1.8 * V1_sp, start=V1_sp)
#             V2[k=0:Nd], (lower_bound=0.2 * V2_sp, upper_bound=1.8 * V2_sp, start=V2_sp)

#             # Temperature, state, [=] K
#             T1[k=0:Nd], (lower_bound=0.2 * T1_sp, upper_bound=1.8 * T1_sp, start=T1_sp)
#             T2[k=0:Nd], (lower_bound=0.2 * T2_sp, upper_bound=1.8 * T2_sp, start=T2_sp)

#             # Concentration, state, [=] mol/m3
#             CA1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA1_sp)
#             CA2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA2_sp)

#             CB1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB1_sp)
#             CB2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB2_sp)

#             CC1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC1_sp)
#             CC2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC2_sp)

#             CD1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD1_sp)
#             CD2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD2_sp)

#             F1[k=0:Nd], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
#             F2[k=0:Nd], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)
#             F3[k=0:Nd], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
#             F4[k=0:Nd], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)

#             Q1[k=0:Nd], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
#             Q2[k=0:Nd], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)

#         end


#         # for k = 0:N
#         #     JuMP.fix(F1[k], F1_sp; force=true)
#         #     JuMP.fix(F2[k], F2_sp; force=true)
#         #     JuMP.fix(F3[k], F3_sp; force=true)
#         #     JuMP.fix(F4[k], F4_sp; force=true)
#         # end


#         for k = 0:Nd
#             JuMP.fix(Q1[k], Q1_sp; force=true)
#             JuMP.fix(Q2[k], Q2_sp; force=true)
#             # JuMP.fix(Q3[k], Q3_sp; force=true)
#             # JuMP.fix(Q4[k], Q4_sp; force=true)
#             # JuMP.fix(Q5[k], Q5_sp; force=true)
#         end

#         for k = 0:Nd
#             set_start_value(V1[k], fix.V1[k+1])
#             set_start_value(V2[k], fix.V2[k+1])

#             set_start_value(T1[k], fix.T1[k+1])
#             set_start_value(T2[k], fix.T2[k+1])

#             set_start_value(CA1[k], fix.CA1[k+1])
#             set_start_value(CA2[k], fix.CA2[k+1])

#             set_start_value(CB1[k], fix.CB1[k+1])
#             set_start_value(CB2[k], fix.CB2[k+1])

#             set_start_value(CC1[k], fix.CC1[k+1])
#             set_start_value(CC2[k], fix.CC2[k+1])

#             set_start_value(CD1[k], fix.CD1[k+1])
#             set_start_value(CD2[k], fix.CD2[k+1])

#             set_start_value(F1[k], fix.F1[k+1])
#             set_start_value(F2[k], fix.F2[k+1])
#             set_start_value(F3[k], fix.F3[k+1])
#             set_start_value(F4[k], fix.F4[k+1])

#             set_start_value(Q1[k], fix.Q1[k+1])
#             set_start_value(Q2[k], fix.Q2[k+1])
#         end

#         @constraints mpc begin

#             # Initial condition
#             V1_inital, V1[0] == V1_init
#             V2_inital, V2[0] == V2_init

#             T1_inital, T1[0] == T1_init
#             T2_inital, T2[0] == T2_init

#             CA1_initial, CA1[0] == CA1_init
#             CA2_initial, CA2[0] == CA2_init

#             CB1_initial, CB1[0] == CB1_init
#             CB2_initial, CB2[0] == CB2_init

#             CC1_initial, CC1[0] == CC1_init
#             CC2_initial, CC2[0] == CC2_init

#             CD1_initial, CD1[0] == CD1_init
#             CD2_initial, CD2[0] == CD2_init

#         end

#         @NLconstraints mpc begin
#             # NLconstraints are the differential equations that describe the dynamics of the system
#             dV1_dt[k=0:Nd-1], V1[k] + (F1[k] + F2[k] + fix.Fr2[k+1] - F3[k]) * dtd == V1_sp

#             dT1_dt[k=0:Nd-1], T1[k] + (
#                 ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                   (fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_A(fix.T4[k+1]) - F3[k] * CA1[k] * H_A(T1[k])) +
#                   (fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_B(fix.T4[k+1]) - F3[k] * CB1[k] * H_B(T1[k])) +
#                   (fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_C(fix.T4[k+1]) - F3[k] * CC1[k] * H_C(T1[k])) +
#                   (fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_D(fix.T4[k+1]) - F3[k] * CD1[k] * H_D(T1[k])))
#                  /
#                  (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
#                 (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
#                 (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dtd == T1[k+1]

#             dCA1_dt[k=0:Nd-1], CA1[k] + (
#                 ((F1[k] * CA0 + fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CA1[k])
#                  /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dtd == CA1[k+1]

#             dCB1_dt[k=0:Nd-1], CB1[k] + (
#                 ((F2[k] * CB0 + fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CB1[k]) /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CB1[k+1]

#             dCC1_dt[k=0:Nd-1], CC1[k] + (
#                 ((fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CC1[k]) /
#                  V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CC1[k+1]

#             dCD1_dt[k=0:Nd-1], CD1[k] + (
#                 ((fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CD1[k]) /
#                  V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CD1[k+1]

#             dV2_dt[k=0:Nd-1], V2[k] + (F3[k] + F4[k] - fix.F5[k+1]) * dtd == V2_sp

#             dT2_dt[k=0:Nd-1], T2[k] + (
#                 (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                  (F3[k] * CA1[k] * H_A(T1[k]) - fix.F5[k+1] * CA2[k] * H_A(T2[k])) +
#                  (F3[k] * CB1[k] * H_B(T1[k]) - fix.F5[k+1] * CB2[k] * H_B(T2[k])) +
#                  (F3[k] * CC1[k] * H_C(T1[k]) - fix.F5[k+1] * CC2[k] * H_C(T2[k])) +
#                  (F3[k] * CD1[k] * H_D(T1[k]) - fix.F5[k+1] * CD2[k] * H_D(T2[k])))
#                 /
#                 (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
#                 +
#                 (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
#                 /
#                 (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dtd == T2[k+1]


#             dCA2_dt[k=0:Nd-1], CA2[k] + (((F3[k] * CA1[k] - fix.F5[k+1] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dtd == CA2[k+1]
#             dCB2_dt[k=0:Nd-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - fix.F5[k+1] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CB2[k+1]
#             dCC2_dt[k=0:Nd-1], CC2[k] + ((F3[k] * CC1[k] - fix.F5[k+1] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CC2[k+1]
#             dCD2_dt[k=0:Nd-1], CD2[k] + ((F3[k] * CD1[k] - fix.F5[k+1] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CD2[k+1]

#             # volHoldUp11[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] <= (-(V1[k] - V1_sp) / 200) + s_path
#             # volHoldUp12[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] >= (-(V1[k] - V1_sp) / 200) - s_path
#             # volHoldUp21[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] <= (-(V2[k] - V2_sp) / 200) + s_path
#             # volHoldUp22[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] >= (-(V2[k] - V2_sp) / 200) - s_path

#             # volHoldUp1[k=0:Nd-1], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / 200
#             # volHoldUp2[k=0:Nd-1], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / 200

#             # volDec1[k=0:N-1], (V1[k+1] - V1_sp) <= 0.8 * (V1[k] - V1_sp)
#             # volDec2[k=0:N-1], (V2[k+1] - V2_sp) <= 0.8 * (V2[k] - V2_sp)

#             # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / 200
#             # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / 200

#         end



#         @NLobjective(mpc, Min, 1e-5 * sum(
#             w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 +
#             w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 +
#             w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 +
#             w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 +
#             w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 +
#             w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2
#             for k = 0:Nd) +
#                                1e-5 * sum(
#             w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
#             w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 for k = 0:Nd
#         )
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
#         # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
#         # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
#         # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
#         # set_optimizer_attribute(mpc, "mu_init", 1e-3)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global dmpc1_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

#         F1 = Vector(JuMP.value.(F1))
#         F2 = Vector(JuMP.value.(F2))
#         F3 = Vector(JuMP.value.(F3))
#         F4 = Vector(JuMP.value.(F4))
#         Q1 = Vector(JuMP.value.(Q1))
#         Q2 = Vector(JuMP.value.(Q2))

#         V1 = Vector(JuMP.value.(V1))
#         V2 = Vector(JuMP.value.(V2))

#         T1 = Vector(JuMP.value.(T1))
#         T2 = Vector(JuMP.value.(T2))

#         CA1 = Vector(JuMP.value.(CA1))
#         CA2 = Vector(JuMP.value.(CA2))

#         CB1 = Vector(JuMP.value.(CB1))
#         CB2 = Vector(JuMP.value.(CB2))

#         CC1 = Vector(JuMP.value.(CC1))
#         CC2 = Vector(JuMP.value.(CC2))

#         CD1 = Vector(JuMP.value.(CD1))
#         CD2 = Vector(JuMP.value.(CD2))

#         return F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2
#     end

#     function dmpc_2()

#         # Control variables: F5, F6, Q3
#         # State variables: All state variables associated with CSTR-3

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin
#             # State variables

#             # Volume, state, [=] m3    
#             V3[k=0:Nd], (lower_bound=0.2 * V3_sp, upper_bound=1.8 * V3_sp, start=V3_sp)

#             # Temperature, state, [=] K
#             T3[k=0:Nd], (lower_bound=0.2 * T3_sp, upper_bound=1.8 * T3_sp, start=T3_sp)

#             # Concentration, state, [=] mol/m3
#             CA3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA3_sp)

#             CB3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB3_sp)

#             CC3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC3_sp)

#             CD3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD3_sp)

#             F5[k=0:Nd], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
#             F6[k=0:Nd], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)

#             Q3[k=0:Nd], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)


#         end

#         for k = 0:Nd
#             # JuMP.fix(F5[k], F5_sp; force=true)
#             # JuMP.fix(F6[k], F6_sp; force=true)
#         end

#         for k = 0:Nd
#             # JuMP.fix(Q1[k], Q1_sp; force=true)
#             # JuMP.fix(Q2[k], Q2_sp; force=true)
#             JuMP.fix(Q3[k], Q3_sp; force=true)
#             # JuMP.fix(Q4[k], Q4_sp; force=true)
#             # JuMP.fix(Q5[k], Q5_sp; force=true)
#         end

#         for k = 0:Nd
#             set_start_value(V3[k], fix.V3[k+1])

#             set_start_value(T3[k], fix.T3[k+1])

#             set_start_value(CA3[k], fix.CA3[k+1])

#             set_start_value(CB3[k], fix.CB3[k+1])

#             set_start_value(CC3[k], fix.CC3[k+1])

#             set_start_value(CD3[k], fix.CD3[k+1])

#             set_start_value(F5[k], fix.F5[k+1])
#             set_start_value(F6[k], fix.F6[k+1])
#         end

#         @constraints mpc begin
#             # Initial condition
#             V3_inital, V3[0] == V3_init

#             T3_inital, T3[0] == T3_init

#             CA3_initial, CA3[0] == CA3_init

#             CB3_initial, CB3[0] == CB3_init

#             CC3_initial, CC3[0] == CC3_init

#             CD3_initial, CD3[0] == CD3_init


#             # volDec3[k=0:N-1], (V3[k+1] - V3_sp) <= 0.8 * (V3[k] - V3_sp)

#         end

#         @NLconstraints mpc begin

#             dV3_dt[k=0:Nd-1], V3[k] + (F5[k] + F6[k] - fix.F7[k+1]) * dtd == V3_sp

#             dT3_dt[k=0:Nd-1], T3[k] + (
#                 ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * fix.CA2[k+1] * H_A(fix.T2[k+1]) - fix.F7[k+1] * CA3[k] * H_A(T3[k])) +
#                   (F5[k] * fix.CB2[k+1] * H_B(fix.T2[k+1]) - fix.F7[k+1] * CB3[k] * H_B(T3[k])) +
#                   (F5[k] * fix.CC2[k+1] * H_C(fix.T2[k+1]) - fix.F7[k+1] * CC3[k] * H_C(T3[k])) +
#                   (F5[k] * fix.CD2[k+1] * H_D(fix.T2[k+1]) - fix.F7[k+1] * CD3[k] * H_D(T3[k])))
#                  /
#                  (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
#                 +
#                 (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
#                 /
#                 (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dtd == T3[k+1]

#             dCA3_dt[k=0:Nd-1], CA3[k] + (((F5[k] * fix.CA2[k+1] - fix.F7[k+1] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dtd == CA3[k+1]
#             dCB3_dt[k=0:Nd-1], CB3[k] + (((F5[k] * fix.CB2[k+1] + F6[k] * CB0 - fix.F7[k+1] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CB3[k+1]
#             dCC3_dt[k=0:Nd-1], CC3[k] + ((F5[k] * fix.CC2[k+1] - fix.F7[k+1] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CC3[k+1]
#             dCD3_dt[k=0:Nd-1], CD3[k] + ((F5[k] * fix.CD2[k+1] - fix.F7[k+1] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CD3[k+1]
#             # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / 200




#             # volHoldUp3[k=0:Nd-1], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / 200
#             # volHoldUp31[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] <= (-(V3[k] - V3_sp) / 200) + s_path
#             # volHoldUp32[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] >= (-(V3[k] - V3_sp) / 200) - s_path
#         end


#         @NLobjective(
#             mpc,
#             Min,
#             1e-5 * sum(
#                 w.v * (V3[k] - V3_sp)^2 +
#                 w.t3 * (T3[k] - T3_sp)^2 +
#                 w.ca3 * (CA3[k] - CA3_sp)^2 +
#                 w.cb3 * (CB3[k] - CB3_sp)^2 +
#                 w.cc3 * (CC3[k] - CC3_sp)^2 +
#                 w.cd3 * (CD3[k] - CD3_sp)^2
#                 for k = 0:Nd) +
#             1e-5 * sum(
#                 w.f5 * (F5[k] - F5_sp)^2 +
#                 w.f6 * (F6[k] - F6_sp)^2 +
#                 w.q3 * (Q3[k] - Q3_sp)^2
#                 for k = 0:Nd
#             )
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
#         # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
#         # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
#         # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
#         # set_optimizer_attribute(mpc, "mu_init", 1e-3)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())


#         F5 = Vector(JuMP.value.(F5))
#         F6 = Vector(JuMP.value.(F6))
#         Q3 = Vector(JuMP.value.(Q3))

#         V3 = Vector(JuMP.value.(V3))

#         T3 = Vector(JuMP.value.(T3))

#         CA3 = Vector(JuMP.value.(CA3))

#         CB3 = Vector(JuMP.value.(CB3))

#         CC3 = Vector(JuMP.value.(CC3))

#         CD3 = Vector(JuMP.value.(CD3))

#         return F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3

#     end

#     function dmpc_3()

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin
#             # Volume, state, [=] m3
#             # State variables
#             # Volume, state, [=] m3
#             V4[k=0:Nd], (lower_bound=0.2 * V4_sp, upper_bound=1.8 * V4_sp, start=V4_sp)
#             V5[k=0:Nd], (lower_bound=0.2 * V5_sp, upper_bound=1.8 * V5_sp, start=V5_sp)

#             # Temperature, state, [=] K
#             T4[k=0:Nd], (lower_bound=0.2 * T4_sp, upper_bound=1.8 * T4_sp, start=T4_sp)
#             T5[k=0:Nd], (lower_bound=0.2 * T5_sp, upper_bound=1.8 * T5_sp, start=T5_sp)

#             # Concentration, state, [=] mol/m3
#             CA4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA4_sp)
#             CA5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA5_sp)

#             CB4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB4_sp)
#             CB5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB5_sp)

#             CC4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC4_sp)
#             CC5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC5_sp)

#             CD4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD4_sp)
#             CD5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD5_sp)

#             # Control variables
#             # Flow, control [=] m3/s
#             F7[k=0:Nd], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
#             F8[k=0:Nd], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
#             F9[k=0:Nd], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
#             F10[k=0:Nd], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
#             Fr1[k=0:Nd], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
#             Fr2[k=0:Nd], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)



#             Q4[k=0:Nd], (start = Q4_sp)
#             Q5[k=0:Nd], (start = Q5_sp)
#         end

#         for k = 0:Nd
#             JuMP.fix(F7[k], F7_sp; force=true)
#             # JuMP.fix(F8[k], F8_sp; force=true)
#             # JuMP.fix(F9[k], F9_sp; force=true)
#             # JuMP.fix(F10[k], F10_sp; force=true)
#             # JuMP.fix(Fr1[k], Fr1_sp; force=true)
#             JuMP.fix(Fr2[k], Fr2_sp; force=true)
#         end

#         for k = 0:Nd
#             # JuMP.fix(Q1[k], Q1_sp; force=true)
#             # JuMP.fix(Q2[k], Q2_sp; force=true)
#             # JuMP.fix(Q3[k], Q3_sp; force=true)
#             JuMP.fix(Q4[k], Q4_sp; force=true)
#             JuMP.fix(Q5[k], Q5_sp; force=true)
#         end

#         for k = 0:Nd
#             set_start_value(V4[k], fix.V4[k+1])
#             set_start_value(V5[k], fix.V5[k+1])

#             set_start_value(T4[k], fix.T4[k+1])
#             set_start_value(T5[k], fix.T5[k+1])

#             set_start_value(CA4[k], fix.CA4[k+1])
#             set_start_value(CA5[k], fix.CA5[k+1])

#             set_start_value(CB4[k], fix.CB4[k+1])
#             set_start_value(CB5[k], fix.CB5[k+1])

#             set_start_value(CC4[k], fix.CC4[k+1])
#             set_start_value(CC5[k], fix.CC5[k+1])

#             set_start_value(CD4[k], fix.CD4[k+1])
#             set_start_value(CD5[k], fix.CD5[k+1])

#             set_start_value(F7[k], fix.F7[k+1])
#             set_start_value(F8[k], fix.F8[k+1])
#             set_start_value(F9[k], fix.F9[k+1])
#             set_start_value(F10[k], fix.F10[k+1])
#             set_start_value(Fr1[k], fix.Fr1[k+1])
#             set_start_value(Fr2[k], fix.Fr2[k+1])

#         end

#         @constraints mpc begin

#             # Initial condition
#             V4_inital, V4[0] == V4_init
#             V5_inital, V5[0] == V5_init

#             T4_inital, T4[0] == T4_init
#             T5_inital, T5[0] == T5_init

#             CA4_initial, CA4[0] == CA4_init
#             CA5_initial, CA5[0] == CA5_init

#             CB4_initial, CB4[0] == CB4_init
#             CB5_initial, CB5[0] == CB5_init

#             CC4_initial, CC4[0] == CC4_init
#             CC5_initial, CC5[0] == CC5_init

#             CD4_initial, CD4[0] == CD4_init
#             CD5_initial, CD5[0] == CD5_init

#             # volDec4[k=0:N-1], (V4[k+1] - V4_sp) <= 0.8 * (V4[k] - V4_sp)
#             # volDec5[k=0:N-1], (V5[k+1] - V5_sp) <= 0.8 * (V5[k] - V5_sp)

#         end

#         @NLconstraints mpc begin
#             # NLconstraints are the differential equations that describe the dynamics of the system

#             dV4_dt[k=0:Nd-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dtd == V4_sp
#             dT4_dt[k=0:Nd-1], T4[k] +
#                               ((Q4[k]
#                                 + (F7[k] * fix.CA3[k+1] * H_A(fix.T3[k+1]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
#                                 + (F7[k] * fix.CB3[k+1] * H_B(fix.T3[k+1]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
#                                 + (F7[k] * fix.CC3[k+1] * H_C(fix.T3[k+1]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
#                                 + (F7[k] * fix.CD3[k+1] * H_D(fix.T3[k+1]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
#                                /
#                                (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dtd == T4[k+1]

#             dCA4_dt[k=0:Nd-1], CA4[k] + ((F7[k] * fix.CA3[k+1] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dtd == CA4[k+1]
#             dCB4_dt[k=0:Nd-1], CB4[k] + ((F7[k] * fix.CB3[k+1] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dtd == CB4[k+1]
#             dCC4_dt[k=0:Nd-1], CC4[k] + ((F7[k] * fix.CC3[k+1] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dtd == CC4[k+1]
#             dCD4_dt[k=0:Nd-1], CD4[k] + ((F7[k] * fix.CD3[k+1] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dtd == CD4[k+1]

#             dV5_dt[k=0:Nd-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dtd == V5_sp

#             dT5_dt[k=0:Nd-1], T5[k] + (
#                 ((Q5[k] +
#                   F10[k] * CD0 * H_D(TD0)
#                   + (Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
#                   + (Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
#                   + (Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
#                   + (Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
#                  /
#                  (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
#                 +
#                 ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
#                  /
#                  (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dtd == T5[k+1]

#             dCA5_dt[k=0:Nd-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dtd == CA5[k+1]
#             dCB5_dt[k=0:Nd-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dtd == CB5[k+1]
#             dCC5_dt[k=0:Nd-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dtd == CC5[k+1]
#             dCD5_dt[k=0:Nd-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dtd == CD5[k+1]
#             # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
#             # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200

#             # volHoldUp4[k=0:N-1], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
#             # volHoldUp5[k=0:N-1], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200


#             # volHoldUp41[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] <= -(V4[k] - V4_sp) / 200 + s_path
#             # volHoldUp42[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] >= -(V4[k] - V4_sp) / 200 - s_path

#             # volHoldUp51[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] <= -(V5[k] - V5_sp) / 200 + s_path
#             # volHoldUp52[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] >= -(V5[k] - V5_sp) / 200 - s_path


#         end


#         @NLobjective(mpc, Min, sum(
#             1e-5 * w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
#             w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
#             w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
#             w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
#             w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
#             w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
#             for k = 0:Nd) +
#                                1e-5 * sum(
#             w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
#             w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
#             w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
#             w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
#             for k = 0:Nd
#         )
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
#         # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
#         # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
#         # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
#         # set_optimizer_attribute(mpc, "mu_init", 1e-3)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global dmpc3_solve_time = MOI.get(mpc, MOI.SolveTimeSec())
#         F7 = Vector(JuMP.value.(F7))
#         F8 = Vector(JuMP.value.(F8))
#         F9 = Vector(JuMP.value.(F9))
#         F10 = Vector(JuMP.value.(F10))
#         Fr1 = Vector(JuMP.value.(Fr1))
#         Fr2 = Vector(JuMP.value.(Fr2))

#         Q4 = Vector(JuMP.value.(Q4))
#         Q5 = Vector(JuMP.value.(Q5))

#         V4 = Vector(JuMP.value.(V4))
#         V5 = Vector(JuMP.value.(V5))

#         T4 = Vector(JuMP.value.(T4))
#         T5 = Vector(JuMP.value.(T5))

#         CA4 = Vector(JuMP.value.(CA4))
#         CA5 = Vector(JuMP.value.(CA5))

#         CB4 = Vector(JuMP.value.(CB4))
#         CB5 = Vector(JuMP.value.(CB5))

#         CC4 = Vector(JuMP.value.(CC4))
#         CC5 = Vector(JuMP.value.(CC5))

#         CD4 = Vector(JuMP.value.(CD4))
#         CD5 = Vector(JuMP.value.(CD5))

#         return F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5
#     end

#     function dmpc()

#         max_steps = 15
#         global dmpc_solve_time = 0
#         for steps = 1:max_steps

#             F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2 = dmpc_1()
#             F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3 = dmpc_2()
#             F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5 = dmpc_3()

#             fix.F1 = F1
#             fix.F2 = F2
#             fix.F3 = F3
#             fix.F4 = F4
#             fix.F5 = F5
#             fix.F6 = F6
#             fix.F7 = F7
#             fix.F8 = F8
#             fix.F9 = F9
#             fix.F10 = F10
#             fix.Fr1 = Fr1
#             fix.Fr2 = Fr2

#             fix.Q1 = Q1
#             fix.Q2 = Q2
#             fix.Q3 = Q3
#             fix.Q4 = Q4
#             fix.Q5 = Q5


#             global dmpc_solve_time = dmpc_solve_time + max(dmpc1_solve_time, dmpc2_solve_time, dmpc3_solve_time)

#             if dmpc_solve_time > c_cpu_max
#                 break
#             end


#             fix.T1 = T1
#             fix.T2 = T2
#             fix.T3 = T3
#             fix.T4 = T4
#             fix.T5 = T5

#             fix.V1 = V1
#             fix.V2 = V2
#             fix.V3 = V3
#             fix.V4 = V4
#             fix.V5 = V5

#             fix.CA1 = CA1
#             fix.CA2 = CA2
#             fix.CA3 = CA3
#             fix.CA4 = CA4
#             fix.CA5 = CA5
#             fix.CB1 = CB1
#             fix.CB2 = CB2
#             fix.CB3 = CB3
#             fix.CB4 = CB4
#             fix.CB5 = CB5
#             fix.CC1 = CC1
#             fix.CC2 = CC2
#             fix.CC3 = CC3
#             fix.CC4 = CC4
#             fix.CC5 = CC5
#             fix.CD1 = CD1
#             fix.CD2 = CD2
#             fix.CD3 = CD3
#             fix.CD4 = CD4
#             fix.CD5 = CD5
#         end

#         F1 = fix.F1
#         F2 = fix.F2
#         F3 = fix.F3
#         F4 = fix.F4
#         F5 = fix.F5
#         F6 = fix.F6
#         F7 = fix.F7
#         F8 = fix.F8
#         F9 = fix.F9
#         F10 = fix.F10
#         Fr1 = fix.Fr1
#         Fr2 = fix.Fr2
#         Q1 = fix.Q1
#         Q2 = fix.Q2
#         Q3 = fix.Q3
#         Q4 = fix.Q4
#         Q5 = fix.Q5

#         Q1 = ones(dtd) .* fix.Q1[1]
#         Q2 = ones(dtd) .* fix.Q2[1]
#         Q3 = ones(dtd) .* fix.Q3[1]
#         Q4 = ones(dtd) .* fix.Q4[1]
#         Q5 = ones(dtd) .* fix.Q5[1]

#         F1 = ones(dtd) .* fix.F1[1]
#         F2 = ones(dtd) .* fix.F2[1]
#         F3 = ones(dtd) .* fix.F3[1]
#         F4 = ones(dtd) .* fix.F4[1]
#         F5 = ones(dtd) .* fix.F5[1]
#         F6 = ones(dtd) .* fix.F6[1]
#         F7 = ones(dtd) .* fix.F7[1]
#         F8 = ones(dtd) .* fix.F8[1]
#         F9 = ones(dtd) .* fix.F9[1]
#         F10 = ones(dtd) .* fix.F10[1]
#         Fr1 = ones(dtd) .* fix.Fr1[1]
#         Fr2 = ones(dtd) .* fix.Fr2[1]

#         for i = 2:Nd
#             append!(F1, ones(dtd) .* fix.F1[i])
#             append!(F2, ones(dtd) .* fix.F2[i])
#             append!(F3, ones(dtd) .* fix.F3[i])
#             append!(F4, ones(dtd) .* fix.F4[i])
#             append!(F5, ones(dtd) .* fix.F5[i])
#             append!(F6, ones(dtd) .* fix.F6[i])
#             append!(F7, ones(dtd) .* fix.F7[i])
#             append!(F8, ones(dtd) .* fix.F8[i])
#             append!(F9, ones(dtd) .* fix.F9[i])
#             append!(F10, ones(dtd) .* fix.F10[i])
#             append!(Fr1, ones(dtd) .* fix.Fr1[i])
#             append!(Fr2, ones(dtd) .* fix.Fr2[i])

#             append!(Q1, ones(dtd) .* fix.Q1[i])
#             append!(Q2, ones(dtd) .* fix.Q2[i])
#             append!(Q3, ones(dtd) .* fix.Q3[i])
#             append!(Q4, ones(dtd) .* fix.Q4[i])
#             append!(Q5, ones(dtd) .* fix.Q5[i])
#         end


#         append!(F1, F1_sp)
#         append!(F2, F2_sp)
#         append!(F3, F3_sp)
#         append!(F4, F4_sp)
#         append!(F5, F5_sp)
#         append!(F6, F6_sp)
#         append!(F7, F7_sp)
#         append!(F8, F8_sp)
#         append!(F9, F9_sp)
#         append!(F10, F10_sp)
#         append!(Fr1, Fr1_sp)
#         append!(Fr2, Fr2_sp)

#         append!(Q1, Q1_sp)
#         append!(Q2, Q2_sp)
#         append!(Q3, Q3_sp)
#         append!(Q4, Q4_sp)
#         append!(Q5, Q5_sp)


#         return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5


#     end

#     function getTraj(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5)

#         V1_vec = zeros(N + 1)
#         V2_vec = zeros(N + 1)
#         V3_vec = zeros(N + 1)
#         V4_vec = zeros(N + 1)
#         V5_vec = zeros(N + 1)
#         T1_vec = zeros(N + 1)
#         T2_vec = zeros(N + 1)
#         T3_vec = zeros(N + 1)
#         T4_vec = zeros(N + 1)
#         T5_vec = zeros(N + 1)

#         CA1_vec = zeros(N + 1)
#         CA2_vec = zeros(N + 1)
#         CA3_vec = zeros(N + 1)
#         CA4_vec = zeros(N + 1)
#         CA5_vec = zeros(N + 1)
#         CB1_vec = zeros(N + 1)
#         CB2_vec = zeros(N + 1)
#         CB3_vec = zeros(N + 1)
#         CB4_vec = zeros(N + 1)
#         CB5_vec = zeros(N + 1)
#         CC1_vec = zeros(N + 1)
#         CC2_vec = zeros(N + 1)
#         CC3_vec = zeros(N + 1)
#         CC4_vec = zeros(N + 1)
#         CC5_vec = zeros(N + 1)
#         CD1_vec = zeros(N + 1)
#         CD2_vec = zeros(N + 1)
#         CD3_vec = zeros(N + 1)
#         CD4_vec = zeros(N + 1)
#         CD5_vec = zeros(N + 1)



#         V1_vec[1] = V1_init
#         V2_vec[1] = V2_init
#         V3_vec[1] = V3_init
#         V4_vec[1] = V4_init
#         V5_vec[1] = V5_init
#         T1_vec[1] = T1_init
#         T2_vec[1] = T2_init
#         T3_vec[1] = T3_init
#         T4_vec[1] = T4_init
#         T5_vec[1] = T5_init

#         CA1_vec[1] = CA1_init
#         CA2_vec[1] = CA2_init
#         CA3_vec[1] = CA3_init
#         CA4_vec[1] = CA4_init
#         CA5_vec[1] = CA5_init
#         CB1_vec[1] = CB1_init
#         CB2_vec[1] = CB2_init
#         CB3_vec[1] = CB3_init
#         CB4_vec[1] = CB4_init
#         CB5_vec[1] = CB5_init
#         CC1_vec[1] = CC1_init
#         CC2_vec[1] = CC2_init
#         CC3_vec[1] = CC3_init
#         CC4_vec[1] = CC4_init
#         CC5_vec[1] = CC5_init
#         CD1_vec[1] = CD1_init
#         CD2_vec[1] = CD2_init
#         CD3_vec[1] = CD3_init
#         CD4_vec[1] = CD4_init
#         CD5_vec[1] = CD5_init

#         for j = 1:N
#             global k = j

#             V1 = V1_vec[k]
#             V2 = V2_vec[k]
#             V3 = V3_vec[k]
#             V4 = V4_vec[k]
#             V5 = V5_vec[k]

#             T1 = T1_vec[k]
#             T2 = T2_vec[k]
#             T3 = T3_vec[k]
#             T4 = T4_vec[k]
#             T5 = T5_vec[k]

#             CA1 = CA1_vec[k]
#             CB1 = CB1_vec[k]
#             CC1 = CC1_vec[k]
#             CD1 = CD1_vec[k]

#             CA2 = CA2_vec[k]
#             CB2 = CB2_vec[k]
#             CC2 = CC2_vec[k]
#             CD2 = CD2_vec[k]

#             CA3 = CA3_vec[k]
#             CB3 = CB3_vec[k]
#             CC3 = CC3_vec[k]
#             CD3 = CD3_vec[k]

#             CA4 = CA4_vec[k]
#             CB4 = CB4_vec[k]
#             CC4 = CC4_vec[k]
#             CD4 = CD4_vec[k]

#             CA5 = CA5_vec[k]
#             CB5 = CB5_vec[k]
#             CC5 = CC5_vec[k]
#             CD5 = CD5_vec[k]

#             x0 = [V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1[k] + F2[k] + Fr2[k] - F3[k]
#                 dV2 = F3[k] + F4[k] - F5[k]
#                 dV3 = F5[k] + F6[k] - F7[k]
#                 dV4 = F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]
#                 dV5 = F10[k] + Fr1[k] - F9[k]

#                 dT1 = (
#                     ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                       (Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3[k] * CA1 * H_A(T1)) +
#                       (Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3[k] * CB1 * H_B(T1)) +
#                       (Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3[k] * CC1 * H_C(T1)) +
#                       (Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3[k] * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1_(T1, CA1, CB1) - delH_r2 * r2_(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                      (F3[k] * CA1 * H_A(T1) - F5[k] * CA2 * H_A(T2)) +
#                      (F3[k] * CB1 * H_B(T1) - F5[k] * CB2 * H_B(T2)) +
#                      (F3[k] * CC1 * H_C(T1) - F5[k] * CC2 * H_C(T2)) +
#                      (F3[k] * CD1 * H_D(T1) - F5[k] * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1_(T2, CA2, CB2) - delH_r2 * r2_(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


#                 dT3 = (
#                     ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2 * H_A(T2) - F7[k] * CA3 * H_A(T3)) +
#                       (F5[k] * CB2 * H_B(T2) - F7[k] * CB3 * H_B(T3)) +
#                       (F5[k] * CC2 * H_C(T2) - F7[k] * CC3 * H_C(T3)) +
#                       (F5[k] * CD2 * H_D(T2) - F7[k] * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1_(T3, CA3, CB3) - delH_r2 * r2_(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4[k]
#                         + (F7[k] * CA3 * H_A(T3) + F9[k] * CA5 * H_A(T5) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8[k] * CA4 * H_A(T4) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7[k] * CB3 * H_B(T3) + F9[k] * CB5 * H_B(T5) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8[k] * CB4 * H_B(T4) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7[k] * CC3 * H_C(T3) + F9[k] * CC5 * H_C(T5) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8[k] * CC4 * H_C(T4) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7[k] * CD3 * H_D(T3) + F9[k] * CD5 * H_D(T5) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8[k] * CD4 * H_D(T4) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5[k] +
#                       F10[k] * CD0 * H_D(TD0)
#                       + (Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9[k] * CA5 * H_A(T5))
#                       + (Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9[k] * CB5 * H_B(T5))
#                       + (Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9[k] * CC5 * H_C(T5))
#                       + (Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9[k] * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2_(T5, CB5, CC5, CD5) - delH_r3 * r3_(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CA1)
#                      /
#                      V1) - r1_(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CB1) /
#                      V1) - r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CC1) /
#                      V1) + r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CD1) /
#                      V1) + r2_(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3[k] * CA1 - F5[k] * CA2) / V2) - r1_(T2, CA2, CB2))
#                 dCB2 = ((F3[k] * CB1 + F4[k] * CB0 - F5[k] * CB2) / V2 - r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3[k] * CC1 - F5[k] * CC2) / V2 + r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3[k] * CD1 - F5[k] * CD2) / V2 + r2_(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5[k] * CA2 - F7[k] * CA3) / V3) - r1_(T3, CA3, CB3))
#                 dCB3 = (((F5[k] * CB2 + F6[k] * CB0 - F7[k] * CB3) / V3) - r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5[k] * CC2 - F7[k] * CC3) / V3 + r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5[k] * CD2 - F7[k] * CD3) / V3 + r2_(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7[k] * CA3 + F9[k] * CA5 - (Fr1[k] + Fr2[k]) * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CA4) / V4)
#                 dCB4 = ((F7[k] * CB3 + F9[k] * CB5 - (Fr1[k] + Fr2[k]) * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CB4) / V4)
#                 dCC4 = ((F7[k] * CC3 + F9[k] * CC5 - (Fr1[k] + Fr2[k]) * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CC4) / V4)
#                 dCD4 = ((F7[k] * CD3 + F9[k] * CD5 - (Fr1[k] + Fr2[k]) * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CD4) / V4)

#                 dCA5 = ((Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CA5) / V5 - r3_(T5, CA5, CD5))
#                 dCB5 = ((Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CB5) / V5 - r2_(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CC5) / V5 - r2_(T5, CB5, CC5, CD5) + 2 * r3_(T5, CA5, CD5))
#                 dCD5 = ((Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10[k] * CD0 - F9[k] * CD5) / V5 + r2_(T5, CB5, CC5, CD5) - r3_(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dt)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23())

#             # Obtain the next values for each element
#             V1_vec[k+1] = max(1e-6, soln.u[end][1])
#             V2_vec[k+1] = max(1e-6, soln.u[end][2])
#             V3_vec[k+1] = max(1e-6, soln.u[end][3])
#             V4_vec[k+1] = max(1e-6, soln.u[end][4])
#             V5_vec[k+1] = max(1e-6, soln.u[end][5])

#             T1_vec[k+1] = max(1e-6, soln.u[end][6])
#             T2_vec[k+1] = max(1e-6, soln.u[end][7])
#             T3_vec[k+1] = max(1e-6, soln.u[end][8])
#             T4_vec[k+1] = max(1e-6, soln.u[end][9])
#             T5_vec[k+1] = max(1e-6, soln.u[end][10])

#             CA1_vec[k+1] = max(1e-6, soln.u[end][11])
#             CB1_vec[k+1] = max(1e-6, soln.u[end][12])
#             CC1_vec[k+1] = max(1e-6, soln.u[end][13])
#             CD1_vec[k+1] = max(1e-6, soln.u[end][14])

#             CA2_vec[k+1] = max(1e-6, soln.u[end][15])
#             CB2_vec[k+1] = max(1e-6, soln.u[end][16])
#             CC2_vec[k+1] = max(1e-6, soln.u[end][17])
#             CD2_vec[k+1] = max(1e-6, soln.u[end][18])

#             CA3_vec[k+1] = max(1e-6, soln.u[end][19])
#             CB3_vec[k+1] = max(1e-6, soln.u[end][20])
#             CC3_vec[k+1] = max(1e-6, soln.u[end][21])
#             CD3_vec[k+1] = max(1e-6, soln.u[end][22])

#             CA4_vec[k+1] = max(1e-6, soln.u[end][23])
#             CB4_vec[k+1] = max(1e-6, soln.u[end][24])
#             CC4_vec[k+1] = max(1e-6, soln.u[end][25])
#             CD4_vec[k+1] = max(1e-6, soln.u[end][26])

#             CA5_vec[k+1] = max(1e-6, soln.u[end][27])
#             CB5_vec[k+1] = max(1e-6, soln.u[end][28])
#             CC5_vec[k+1] = max(1e-6, soln.u[end][29])
#             CD5_vec[k+1] = max(1e-6, soln.u[end][30])

#         end

#         V1 = V1_vec
#         V2 = V2_vec
#         V3 = V3_vec
#         V4 = V4_vec
#         V5 = V5_vec

#         T1 = T1_vec
#         T2 = T2_vec
#         T3 = T3_vec
#         T4 = T4_vec
#         T5 = T5_vec

#         CA1 = CA1_vec
#         CB1 = CB1_vec
#         CC1 = CC1_vec
#         CD1 = CD1_vec

#         CA2 = CA2_vec
#         CB2 = CB2_vec
#         CC2 = CC2_vec
#         CD2 = CD2_vec

#         CA3 = CA3_vec
#         CB3 = CB3_vec
#         CC3 = CC3_vec
#         CD3 = CD3_vec

#         CA4 = CA4_vec
#         CB4 = CB4_vec
#         CC4 = CC4_vec
#         CD4 = CD4_vec

#         CA5 = CA5_vec
#         CB5 = CB5_vec
#         CC5 = CC5_vec
#         CD5 = CD5_vec

#         return V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5
#     end

#     function getPI(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5)

#         V1_vec = zeros(N + 1)
#         V2_vec = zeros(N + 1)
#         V3_vec = zeros(N + 1)
#         V4_vec = zeros(N + 1)
#         V5_vec = zeros(N + 1)
#         T1_vec = zeros(N + 1)
#         T2_vec = zeros(N + 1)
#         T3_vec = zeros(N + 1)
#         T4_vec = zeros(N + 1)
#         T5_vec = zeros(N + 1)

#         CA1_vec = zeros(N + 1)
#         CA2_vec = zeros(N + 1)
#         CA3_vec = zeros(N + 1)
#         CA4_vec = zeros(N + 1)
#         CA5_vec = zeros(N + 1)
#         CB1_vec = zeros(N + 1)
#         CB2_vec = zeros(N + 1)
#         CB3_vec = zeros(N + 1)
#         CB4_vec = zeros(N + 1)
#         CB5_vec = zeros(N + 1)
#         CC1_vec = zeros(N + 1)
#         CC2_vec = zeros(N + 1)
#         CC3_vec = zeros(N + 1)
#         CC4_vec = zeros(N + 1)
#         CC5_vec = zeros(N + 1)
#         CD1_vec = zeros(N + 1)
#         CD2_vec = zeros(N + 1)
#         CD3_vec = zeros(N + 1)
#         CD4_vec = zeros(N + 1)
#         CD5_vec = zeros(N + 1)

#         V1_vec[1] = V1_init
#         V2_vec[1] = V2_init
#         V3_vec[1] = V3_init
#         V4_vec[1] = V4_init
#         V5_vec[1] = V5_init
#         T1_vec[1] = T1_init
#         T2_vec[1] = T2_init
#         T3_vec[1] = T3_init
#         T4_vec[1] = T4_init
#         T5_vec[1] = T5_init

#         CA1_vec[1] = CA1_init
#         CA2_vec[1] = CA2_init
#         CA3_vec[1] = CA3_init
#         CA4_vec[1] = CA4_init
#         CA5_vec[1] = CA5_init
#         CB1_vec[1] = CB1_init
#         CB2_vec[1] = CB2_init
#         CB3_vec[1] = CB3_init
#         CB4_vec[1] = CB4_init
#         CB5_vec[1] = CB5_init
#         CC1_vec[1] = CC1_init
#         CC2_vec[1] = CC2_init
#         CC3_vec[1] = CC3_init
#         CC4_vec[1] = CC4_init
#         CC5_vec[1] = CC5_init
#         CD1_vec[1] = CD1_init
#         CD2_vec[1] = CD2_init
#         CD3_vec[1] = CD3_init
#         CD4_vec[1] = CD4_init
#         CD5_vec[1] = CD5_init

#         for j = 1:N
#             global k = j

#             V1 = V1_vec[k]
#             V2 = V2_vec[k]
#             V3 = V3_vec[k]
#             V4 = V4_vec[k]
#             V5 = V5_vec[k]

#             T1 = T1_vec[k]
#             T2 = T2_vec[k]
#             T3 = T3_vec[k]
#             T4 = T4_vec[k]
#             T5 = T5_vec[k]

#             CA1 = CA1_vec[k]
#             CB1 = CB1_vec[k]
#             CC1 = CC1_vec[k]
#             CD1 = CD1_vec[k]

#             CA2 = CA2_vec[k]
#             CB2 = CB2_vec[k]
#             CC2 = CC2_vec[k]
#             CD2 = CD2_vec[k]

#             CA3 = CA3_vec[k]
#             CB3 = CB3_vec[k]
#             CC3 = CC3_vec[k]
#             CD3 = CD3_vec[k]

#             CA4 = CA4_vec[k]
#             CB4 = CB4_vec[k]
#             CC4 = CC4_vec[k]
#             CD4 = CD4_vec[k]

#             CA5 = CA5_vec[k]
#             CB5 = CB5_vec[k]
#             CC5 = CC5_vec[k]
#             CD5 = CD5_vec[k]

#             x0 = [V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1[k] + F2[k] + Fr2[k] - F3[k]
#                 dV2 = F3[k] + F4[k] - F5[k]
#                 dV3 = F5[k] + F6[k] - F7[k]
#                 dV4 = F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]
#                 dV5 = F10[k] + Fr1[k] - F9[k]

#                 dT1 = (
#                     ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                       (Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3[k] * CA1 * H_A(T1)) +
#                       (Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3[k] * CB1 * H_B(T1)) +
#                       (Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3[k] * CC1 * H_C(T1)) +
#                       (Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3[k] * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1_(T1, CA1, CB1) - delH_r2 * r2_(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                      (F3[k] * CA1 * H_A(T1) - F5[k] * CA2 * H_A(T2)) +
#                      (F3[k] * CB1 * H_B(T1) - F5[k] * CB2 * H_B(T2)) +
#                      (F3[k] * CC1 * H_C(T1) - F5[k] * CC2 * H_C(T2)) +
#                      (F3[k] * CD1 * H_D(T1) - F5[k] * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1_(T2, CA2, CB2) - delH_r2 * r2_(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


#                 dT3 = (
#                     ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2 * H_A(T2) - F7[k] * CA3 * H_A(T3)) +
#                       (F5[k] * CB2 * H_B(T2) - F7[k] * CB3 * H_B(T3)) +
#                       (F5[k] * CC2 * H_C(T2) - F7[k] * CC3 * H_C(T3)) +
#                       (F5[k] * CD2 * H_D(T2) - F7[k] * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1_(T3, CA3, CB3) - delH_r2 * r2_(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4[k]
#                         + (F7[k] * CA3 * H_A(T3) + F9[k] * CA5 * H_A(T5) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8[k] * CA4 * H_A(T4) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7[k] * CB3 * H_B(T3) + F9[k] * CB5 * H_B(T5) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8[k] * CB4 * H_B(T4) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7[k] * CC3 * H_C(T3) + F9[k] * CC5 * H_C(T5) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8[k] * CC4 * H_C(T4) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7[k] * CD3 * H_D(T3) + F9[k] * CD5 * H_D(T5) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8[k] * CD4 * H_D(T4) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5[k] +
#                       F10[k] * CD0 * H_D(TD0)
#                       + (Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9[k] * CA5 * H_A(T5))
#                       + (Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9[k] * CB5 * H_B(T5))
#                       + (Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9[k] * CC5 * H_C(T5))
#                       + (Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9[k] * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2_(T5, CB5, CC5, CD5) - delH_r3 * r3_(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CA1)
#                      /
#                      V1) - r1_(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CB1) /
#                      V1) - r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CC1) /
#                      V1) + r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CD1) /
#                      V1) + r2_(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3[k] * CA1 - F5[k] * CA2) / V2) - r1_(T2, CA2, CB2))
#                 dCB2 = ((F3[k] * CB1 + F4[k] * CB0 - F5[k] * CB2) / V2 - r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3[k] * CC1 - F5[k] * CC2) / V2 + r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3[k] * CD1 - F5[k] * CD2) / V2 + r2_(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5[k] * CA2 - F7[k] * CA3) / V3) - r1_(T3, CA3, CB3))
#                 dCB3 = (((F5[k] * CB2 + F6[k] * CB0 - F7[k] * CB3) / V3) - r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5[k] * CC2 - F7[k] * CC3) / V3 + r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5[k] * CD2 - F7[k] * CD3) / V3 + r2_(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7[k] * CA3 + F9[k] * CA5 - (Fr1[k] + Fr2[k]) * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CA4) / V4)
#                 dCB4 = ((F7[k] * CB3 + F9[k] * CB5 - (Fr1[k] + Fr2[k]) * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CB4) / V4)
#                 dCC4 = ((F7[k] * CC3 + F9[k] * CC5 - (Fr1[k] + Fr2[k]) * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CC4) / V4)
#                 dCD4 = ((F7[k] * CD3 + F9[k] * CD5 - (Fr1[k] + Fr2[k]) * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CD4) / V4)

#                 dCA5 = ((Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CA5) / V5 - r3_(T5, CA5, CD5))
#                 dCB5 = ((Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CB5) / V5 - r2_(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CC5) / V5 - r2_(T5, CB5, CC5, CD5) + 2 * r3_(T5, CA5, CD5))
#                 dCD5 = ((Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10[k] * CD0 - F9[k] * CD5) / V5 + r2_(T5, CB5, CC5, CD5) - r3_(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dt)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23(), alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, save_everystep=false)

#             # Obtain the next values for each element
#             V1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][1]))
#             V2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][2]))
#             V3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][3]))
#             V4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][4]))
#             V5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][5]))

#             T1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][6]))
#             T2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][7]))
#             T3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][8]))
#             T4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][9]))
#             T5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][10]))

#             CA1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][11]))
#             CB1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][12]))
#             CC1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][13]))
#             CD1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][14]))

#             CA2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][15]))
#             CB2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][16]))
#             CC2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][17]))
#             CD2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][18]))

#             CA3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][19]))
#             CB3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][20]))
#             CC3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][21]))
#             CD3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][22]))

#             CA4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][23]))
#             CB4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][24]))
#             CC4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][25]))
#             CD4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][26]))

#             CA5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][27]))
#             CB5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][28]))
#             CC5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][29]))
#             CD5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][30]))


#         end

#         V1 = V1_vec
#         V2 = V2_vec
#         V3 = V3_vec
#         V4 = V4_vec
#         V5 = V5_vec

#         T1 = T1_vec
#         T2 = T2_vec
#         T3 = T3_vec
#         T4 = T4_vec
#         T5 = T5_vec

#         CA1 = CA1_vec
#         CB1 = CB1_vec
#         CC1 = CC1_vec
#         CD1 = CD1_vec

#         CA2 = CA2_vec
#         CB2 = CB2_vec
#         CC2 = CC2_vec
#         CD2 = CD2_vec

#         CA3 = CA3_vec
#         CB3 = CB3_vec
#         CC3 = CC3_vec
#         CD3 = CD3_vec

#         CA4 = CA4_vec
#         CB4 = CB4_vec
#         CC4 = CC4_vec
#         CD4 = CD4_vec

#         CA5 = CA5_vec
#         CB5 = CB5_vec
#         CC5 = CC5_vec
#         CD5 = CD5_vec

#         ISE = sum(w.v * (V1_vec[k] - V1_sp)^2 + w.v * (V2_vec[k] - V2_sp)^2 + w.v * (V3_vec[k] - V3_sp)^2 + w.v * (V4_vec[k] - V4_sp)^2 + w.v * (V5_vec[k] - V5_sp)^2 +
#                   w.t1 * (T1_vec[k] - T1_sp)^2 + w.t2 * (T2_vec[k] - T2_sp)^2 + w.t3 * (T3_vec[k] - T3_sp)^2 + w.t4 * (T4_vec[k] - T4_sp)^2 + w.t5 * (T5_vec[k] - T5_sp)^2 +
#                   w.ca1 * (CA1_vec[k] - CA1_sp)^2 + w.ca2 * (CA2_vec[k] - CA2_sp)^2 + w.ca3 * (CA3_vec[k] - CA3_sp)^2 + w.ca4 * (CA4_vec[k] - CA4_sp)^2 + w.ca5 * (CA5_vec[k] - CA5_sp)^2 +
#                   w.cb1 * (CB1_vec[k] - CB1_sp)^2 + w.cb2 * (CB2_vec[k] - CB2_sp)^2 + w.cb3 * (CB3_vec[k] - CB3_sp)^2 + w.cb4 * (CB4_vec[k] - CB4_sp)^2 + w.cb5 * (CB5_vec[k] - CB5_sp)^2 +
#                   w.cc1 * (CC1_vec[k] - CC1_sp)^2 + w.cc2 * (CC2_vec[k] - CC2_sp)^2 + w.cc3 * (CC3_vec[k] - CC3_sp)^2 + w.cc4 * (CC4_vec[k] - CC4_sp)^2 + w.cc5 * (CC5_vec[k] - CC5_sp)^2 +
#                   w.cd1 * (CD1_vec[k] - CD1_sp)^2 + w.cd2 * (CD2_vec[k] - CD2_sp)^2 + w.cd3 * (CD3_vec[k] - CD3_sp)^2 + w.cd4 * (CD4_vec[k] - CD4_sp)^2 + w.cd5 * (CD5_vec[k] - CD5_sp)^2 for k = 1:N+1)



#         ISC = sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 + w.f5 * (F5[k] - F5_sp)^2 +
#                   w.f6 * (F6[k] - F6_sp)^2 + w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 + w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 + w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
#                   w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2 for k = 1:N+1)

#         PI = ISE + ISC

#         if isnan(PI)
#             PI = 1e12
#         end

#         return ISE, ISC, PI
#     end

#     function takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

#         # We are using a 30 second sampling time for the system, just with a fine discretization. Therefore, the next timestep is at k = 61

#         global V1_init = V1[dtd+1]
#         global V2_init = V2[dtd+1]
#         global V3_init = V3[dtd+1]
#         global V4_init = V4[dtd+1]
#         global V5_init = V5[dtd+1]

#         global T1_init = T1[dtd+1]
#         global T2_init = T2[dtd+1]
#         global T3_init = T3[dtd+1]
#         global T4_init = T4[dtd+1]
#         global T5_init = T5[dtd+1]

#         global CA1_init = CA1[dtd+1]
#         global CA2_init = CA2[dtd+1]
#         global CA3_init = CA3[dtd+1]
#         global CA4_init = CA4[dtd+1]
#         global CA5_init = CA5[dtd+1]

#         global CB1_init = CB1[dtd+1]
#         global CB2_init = CB2[dtd+1]
#         global CB3_init = CB3[dtd+1]
#         global CB4_init = CB4[dtd+1]
#         global CB5_init = CB5[dtd+1]

#         global CC1_init = CC1[dtd+1]
#         global CC2_init = CC2[dtd+1]
#         global CC3_init = CC3[dtd+1]
#         global CC4_init = CC4[dtd+1]
#         global CC5_init = CC5[dtd+1]

#         global CD1_init = CD1[dtd+1]
#         global CD2_init = CD2[dtd+1]
#         global CD3_init = CD3[dtd+1]
#         global CD4_init = CD4[dtd+1]
#         global CD5_init = CD5[dtd+1]
#     end

#     predict = Decomposition_Trajectory(zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P))

#     centr_PI_vec = zeros(P)
#     centr_ISE_vec = zeros(P)
#     centr_ISC_vec = zeros(P)
#     centr_time = zeros(P)

#     distr_PI_vec = zeros(P)
#     distr_ISE_vec = zeros(P)
#     distr_ISC_vec = zeros(P)
#     distr_time = zeros(P)

#     function mhmpc_sim()

#         for mpc_step = 1:P

#             println("MPC Step: $(mpc_step)")
#             println()
#             predict.V1[mpc_step] = V1_init
#             predict.V2[mpc_step] = V2_init
#             predict.V3[mpc_step] = V3_init
#             predict.V4[mpc_step] = V4_init
#             predict.V5[mpc_step] = V5_init

#             predict.T1[mpc_step] = T1_init
#             predict.T2[mpc_step] = T2_init
#             predict.T3[mpc_step] = T3_init
#             predict.T4[mpc_step] = T4_init
#             predict.T5[mpc_step] = T5_init

#             predict.CA1[mpc_step] = CA1_init
#             predict.CA2[mpc_step] = CA2_init
#             predict.CA3[mpc_step] = CA3_init
#             predict.CA4[mpc_step] = CA4_init
#             predict.CA5[mpc_step] = CA5_init

#             predict.CB1[mpc_step] = CB1_init
#             predict.CB2[mpc_step] = CB2_init
#             predict.CB3[mpc_step] = CB3_init
#             predict.CB4[mpc_step] = CB4_init
#             predict.CB5[mpc_step] = CB5_init

#             predict.CC1[mpc_step] = CC1_init
#             predict.CC2[mpc_step] = CC2_init
#             predict.CC3[mpc_step] = CC3_init
#             predict.CC4[mpc_step] = CC4_init
#             predict.CC5[mpc_step] = CC5_init

#             predict.CD1[mpc_step] = CD1_init
#             predict.CD2[mpc_step] = CD2_init
#             predict.CD3[mpc_step] = CD3_init
#             predict.CD4[mpc_step] = CD4_init
#             predict.CD5[mpc_step] = CD5_init

#             ex.T1[mpc_step] = T1_init
#             ex.T2[mpc_step] = T2_init
#             ex.T3[mpc_step] = T3_init
#             ex.T4[mpc_step] = T4_init
#             ex.T5[mpc_step] = T5_init

#             ex.V1[mpc_step] = V1_init
#             ex.V2[mpc_step] = V2_init
#             ex.V3[mpc_step] = V3_init
#             ex.V4[mpc_step] = V4_init
#             ex.V5[mpc_step] = V5_init

#             ex.CA1[mpc_step] = CA1_init
#             ex.CA2[mpc_step] = CA2_init
#             ex.CA3[mpc_step] = CA3_init
#             ex.CA4[mpc_step] = CA4_init
#             ex.CA5[mpc_step] = CA5_init

#             ex.CB1[mpc_step] = CB1_init
#             ex.CB2[mpc_step] = CB2_init
#             ex.CB3[mpc_step] = CB3_init
#             ex.CB4[mpc_step] = CB4_init
#             ex.CB5[mpc_step] = CB5_init

#             ex.CC1[mpc_step] = CC1_init
#             ex.CC2[mpc_step] = CC2_init
#             ex.CC3[mpc_step] = CC3_init
#             ex.CC4[mpc_step] = CC4_init
#             ex.CC5[mpc_step] = CC5_init

#             ex.CD1[mpc_step] = CD1_init
#             ex.CD2[mpc_step] = CD2_init
#             ex.CD3[mpc_step] = CD3_init
#             ex.CD4[mpc_step] = CD4_init
#             ex.CD5[mpc_step] = CD5_init

#             # Initialize fix structure for decomp
#             for set_traj = 1
#                 # Set control variables to nominal steady state values
#                 fix.F1 .= F1_sp
#                 fix.F2 .= F2_sp
#                 fix.F3 .= F3_sp
#                 fix.F4 .= F4_sp
#                 fix.F5 .= F5_sp
#                 fix.F6 .= F6_sp
#                 fix.F7 .= F7_sp
#                 fix.F8 .= F8_sp
#                 fix.F9 .= F9_sp
#                 fix.F10 .= F10_sp
#                 fix.Fr1 .= Fr1_sp
#                 fix.Fr2 .= Fr2_sp

#                 fix.Q1 .= Q1_sp
#                 fix.Q2 .= Q2_sp
#                 fix.Q3 .= Q3_sp
#                 fix.Q4 .= Q4_sp
#                 fix.Q5 .= Q5_sp

#                 fix.V1[1] = V1_init
#                 fix.V2[1] = V2_init
#                 fix.V3[1] = V3_init
#                 fix.V4[1] = V4_init
#                 fix.V5[1] = V5_init

#                 fix.T1[1] = T1_init
#                 fix.T2[1] = T2_init
#                 fix.T3[1] = T3_init
#                 fix.T4[1] = T4_init
#                 fix.T5[1] = T5_init

#                 fix.CA1[1] = CA1_init
#                 fix.CA2[1] = CA2_init
#                 fix.CA3[1] = CA3_init
#                 fix.CA4[1] = CA4_init
#                 fix.CA5[1] = CA5_init
#                 fix.CB1[1] = CB1_init
#                 fix.CB2[1] = CB2_init
#                 fix.CB3[1] = CB3_init
#                 fix.CB4[1] = CB4_init
#                 fix.CB5[1] = CB5_init
#                 fix.CC1[1] = CC1_init
#                 fix.CC2[1] = CC2_init
#                 fix.CC3[1] = CC3_init
#                 fix.CC4[1] = CC4_init
#                 fix.CC5[1] = CC5_init
#                 fix.CD1[1] = CD1_init
#                 fix.CD2[1] = CD2_init
#                 fix.CD3[1] = CD3_init
#                 fix.CD4[1] = CD4_init
#                 fix.CD5[1] = CD5_init

#                 # Set state variable guesses to behavior associated with these
#                 for i = 1:Nd
#                     global k = i

#                     F1 = fix.F1[k]
#                     F2 = fix.F2[k]
#                     F3 = fix.F3[k]
#                     F4 = fix.F4[k]
#                     F5 = fix.F5[k]
#                     F6 = fix.F6[k]
#                     F7 = fix.F7[k]
#                     F8 = fix.F8[k]
#                     F9 = fix.F9[k]
#                     F10 = fix.F10[k]
#                     Fr1 = fix.Fr1[k]
#                     Fr2 = fix.Fr2[k]

#                     Q1 = fix.Q1[k]
#                     Q2 = fix.Q2[k]
#                     Q3 = fix.Q3[k]
#                     Q4 = fix.Q4[k]
#                     Q5 = fix.Q5[k]

#                     V1 = fix.V1[k]
#                     V2 = fix.V2[k]
#                     V3 = fix.V3[k]
#                     V4 = fix.V4[k]
#                     V5 = fix.V5[k]

#                     T1 = fix.T1[k]
#                     T2 = fix.T2[k]
#                     T3 = fix.T3[k]
#                     T4 = fix.T4[k]
#                     T5 = fix.T5[k]

#                     CA1 = fix.CA1[k]
#                     CB1 = fix.CB1[k]
#                     CC1 = fix.CC1[k]
#                     CD1 = fix.CD1[k]

#                     CA2 = fix.CA2[k]
#                     CB2 = fix.CB2[k]
#                     CC2 = fix.CC2[k]
#                     CD2 = fix.CD2[k]

#                     CA3 = fix.CA3[k]
#                     CB3 = fix.CB3[k]
#                     CC3 = fix.CC3[k]
#                     CD3 = fix.CD3[k]

#                     CA4 = fix.CA4[k]
#                     CB4 = fix.CB4[k]
#                     CC4 = fix.CC4[k]
#                     CD4 = fix.CD4[k]

#                     CA5 = fix.CA5[k]
#                     CB5 = fix.CB5[k]
#                     CC5 = fix.CC5[k]
#                     CD5 = fix.CD5[k]

#                     x0 = [fix.V1[k], fix.V2[k], fix.V3[k], fix.V4[k], fix.V5[k], fix.T1[k], fix.T2[k], fix.T3[k], fix.T4[k], fix.T5[k], fix.CA1[k], fix.CB1[k], fix.CC1[k], fix.CD1[k], fix.CA2[k], fix.CB2[k], fix.CC2[k], fix.CD2[k], fix.CA3[k], fix.CB3[k], fix.CC3[k], fix.CD3[k], fix.CA4[k], fix.CB4[k], fix.CC4[k], fix.CD4[k], fix.CA5[k], fix.CB5[k], fix.CC5[k], fix.CD5[k]]

#                     # Define the ODE function
#                     function f(y, p, t)
#                         # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                         V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y


#                         dV1 = F1 + F2 + Fr2 - F3
#                         dV2 = F3 + F4 - F5
#                         dV3 = F5 + F6 - F7
#                         dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                         dV5 = F10 + Fr1 - F9

#                         dT1 = (
#                             ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                               (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                               (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                               (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                               (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                              /
#                              (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                             (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                             (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                         dT2 = (
#                             (Q2 + F4 * CB0 * H_B(TB0) +
#                              (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                              (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                              (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                              (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                             /
#                             (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                             +
#                             (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                             /
#                             (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


#                         dT3 = (
#                             ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                               (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                               (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                               (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                              /
#                              (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                             +
#                             (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                             /
#                             (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                         dT4 = ((Q4
#                                 + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                                 + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                                 + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                                 + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                                /
#                                (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                         dT5 = (
#                             ((Q5 +
#                               F10 * CD0 * H_D(TD0)
#                               + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                               + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                               + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                               + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                              /
#                              (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                             +
#                             ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                              /
#                              (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                         dCA1 = (
#                             ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CA1)
#                              /
#                              V1) - r1(T1, CA1, CB1))

#                         dCB1 = (
#                             ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CB1) /
#                              V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCC1 = (
#                             ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CC1) /
#                              V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCD1 = (
#                             ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CD1) /
#                              V1) + r2(T1, CB1, CC1, CD1))

#                         dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                         dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                         dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                         dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                         dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                         dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                         dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                         dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                         dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                         dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                         dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                         dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                         return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#                     end

#                     # Solve ODE 
#                     tspan = (0.0, dtd)
#                     prob = ODEProblem(f, x0, tspan)
#                     soln = solve(prob, Rosenbrock23())

#                     # Obtain the next values for each element
#                     fix.V1[k+1] = soln.u[end][1]
#                     fix.V2[k+1] = soln.u[end][2]
#                     fix.V3[k+1] = soln.u[end][3]
#                     fix.V4[k+1] = soln.u[end][4]
#                     fix.V5[k+1] = soln.u[end][5]

#                     fix.T1[k+1] = soln.u[end][6]
#                     fix.T2[k+1] = soln.u[end][7]
#                     fix.T3[k+1] = soln.u[end][8]
#                     fix.T4[k+1] = soln.u[end][9]
#                     fix.T5[k+1] = soln.u[end][10]

#                     fix.CA1[k+1] = soln.u[end][11]
#                     fix.CB1[k+1] = soln.u[end][12]
#                     fix.CC1[k+1] = soln.u[end][13]
#                     fix.CD1[k+1] = soln.u[end][14]

#                     fix.CA2[k+1] = soln.u[end][15]
#                     fix.CB2[k+1] = soln.u[end][16]
#                     fix.CC2[k+1] = soln.u[end][17]
#                     fix.CD2[k+1] = soln.u[end][18]

#                     fix.CA3[k+1] = soln.u[end][19]
#                     fix.CB3[k+1] = soln.u[end][20]
#                     fix.CC3[k+1] = soln.u[end][21]
#                     fix.CD3[k+1] = soln.u[end][22]

#                     fix.CA4[k+1] = soln.u[end][23]
#                     fix.CB4[k+1] = soln.u[end][24]
#                     fix.CC4[k+1] = soln.u[end][25]
#                     fix.CD4[k+1] = soln.u[end][26]

#                     fix.CA5[k+1] = soln.u[end][27]
#                     fix.CB5[k+1] = soln.u[end][28]
#                     fix.CC5[k+1] = soln.u[end][29]
#                     fix.CD5[k+1] = soln.u[end][30]

#                 end
#             end

#             # Initialize ig structure for centralized
#             for set_traj = 1

#                 # Set control variables to nominal steady state values
#                 ig.F1 .= F1_sp
#                 ig.F2 .= F2_sp
#                 ig.F3 .= F3_sp
#                 ig.F4 .= F4_sp
#                 ig.F5 .= F5_sp
#                 ig.F6 .= F6_sp
#                 ig.F7 .= F7_sp
#                 ig.F8 .= F8_sp
#                 ig.F9 .= F9_sp
#                 ig.F10 .= F10_sp
#                 ig.Fr1 .= Fr1_sp
#                 ig.Fr2 .= Fr2_sp

#                 ig.Q1 .= Q1_sp
#                 ig.Q2 .= Q2_sp
#                 ig.Q3 .= Q3_sp
#                 ig.Q4 .= Q4_sp
#                 ig.Q5 .= Q5_sp

#                 ig.V1[1] = V1_init
#                 ig.V2[1] = V2_init
#                 ig.V3[1] = V3_init
#                 ig.V4[1] = V4_init
#                 ig.V5[1] = V5_init
#                 ig.T1[1] = T1_init
#                 ig.T2[1] = T2_init
#                 ig.T3[1] = T3_init
#                 ig.T4[1] = T4_init
#                 ig.T5[1] = T5_init

#                 ig.CA1[1] = CA1_init
#                 ig.CA2[1] = CA2_init
#                 ig.CA3[1] = CA3_init
#                 ig.CA4[1] = CA4_init
#                 ig.CA5[1] = CA5_init
#                 ig.CB1[1] = CB1_init
#                 ig.CB2[1] = CB2_init
#                 ig.CB3[1] = CB3_init
#                 ig.CB4[1] = CB4_init
#                 ig.CB5[1] = CB5_init
#                 ig.CC1[1] = CC1_init
#                 ig.CC2[1] = CC2_init
#                 ig.CC3[1] = CC3_init
#                 ig.CC4[1] = CC4_init
#                 ig.CC5[1] = CC5_init
#                 ig.CD1[1] = CD1_init
#                 ig.CD2[1] = CD2_init
#                 ig.CD3[1] = CD3_init
#                 ig.CD4[1] = CD4_init
#                 ig.CD5[1] = CD5_init

#                 for i = 1:N
#                     global k = i

#                     F1 = ig.F1[k]
#                     F2 = ig.F2[k]
#                     F3 = ig.F3[k]
#                     F4 = ig.F4[k]
#                     F5 = ig.F5[k]
#                     F6 = ig.F6[k]
#                     F7 = ig.F7[k]
#                     F8 = ig.F8[k]
#                     F9 = ig.F9[k]
#                     F10 = ig.F10[k]
#                     Fr1 = ig.Fr1[k]
#                     Fr2 = ig.Fr2[k]
#                     Q1 = ig.Q1[k]
#                     Q2 = ig.Q2[k]
#                     Q3 = ig.Q3[k]
#                     Q4 = ig.Q4[k]
#                     Q5 = ig.Q5[k]

#                     V1 = ig.V1[k]
#                     V2 = ig.V2[k]
#                     V3 = ig.V3[k]
#                     V4 = ig.V4[k]
#                     V5 = ig.V5[k]

#                     T1 = ig.T1[k]
#                     T2 = ig.T2[k]
#                     T3 = ig.T3[k]
#                     T4 = ig.T4[k]
#                     T5 = ig.T5[k]

#                     CA1 = ig.CA1[k]
#                     CB1 = ig.CB1[k]
#                     CC1 = ig.CC1[k]
#                     CD1 = ig.CD1[k]

#                     CA2 = ig.CA2[k]
#                     CB2 = ig.CB2[k]
#                     CC2 = ig.CC2[k]
#                     CD2 = ig.CD2[k]

#                     CA3 = ig.CA3[k]
#                     CB3 = ig.CB3[k]
#                     CC3 = ig.CC3[k]
#                     CD3 = ig.CD3[k]

#                     CA4 = ig.CA4[k]
#                     CB4 = ig.CB4[k]
#                     CC4 = ig.CC4[k]
#                     CD4 = ig.CD4[k]

#                     CA5 = ig.CA5[k]
#                     CB5 = ig.CB5[k]
#                     CC5 = ig.CC5[k]
#                     CD5 = ig.CD5[k]

#                     x0 = [ig.V1[k], ig.V2[k], ig.V3[k], ig.V4[k], ig.V5[k], ig.T1[k], ig.T2[k], ig.T3[k], ig.T4[k], ig.T5[k], ig.CA1[k], ig.CB1[k], ig.CC1[k], ig.CD1[k], ig.CA2[k], ig.CB2[k], ig.CC2[k], ig.CD2[k], ig.CA3[k], ig.CB3[k], ig.CC3[k], ig.CD3[k], ig.CA4[k], ig.CB4[k], ig.CC4[k], ig.CD4[k], ig.CA5[k], ig.CB5[k], ig.CC5[k], ig.CD5[k]]

#                     # Define the ODE function
#                     function f(y, p, t)
#                         # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                         V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                         dV1 = F1 + F2 + Fr2 - F3
#                         dV2 = F3 + F4 - F5
#                         dV3 = F5 + F6 - F7
#                         dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                         dV5 = F10 + Fr1 - F9

#                         dT1 = (
#                             ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                               (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                               (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                               (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                               (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                              /
#                              (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                             (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                             (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                         dT2 = (
#                             (Q2 + F4 * CB0 * H_B(TB0) +
#                              (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                              (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                              (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                              (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                             /
#                             (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                             +
#                             (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                             /
#                             (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

#                         dT3 = (
#                             ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                               (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                               (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                               (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                              /
#                              (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                             +
#                             (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                             /
#                             (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                         dT4 = ((Q4
#                                 + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                                 + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                                 + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                                 + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                                /
#                                (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                         dT5 = (
#                             ((Q5 +
#                               F10 * CD0 * H_D(TD0)
#                               + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                               + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                               + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                               + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                              /
#                              (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                             +
#                             ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                              /
#                              (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                         dCA1 = (
#                             ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CA1)
#                              /
#                              V1) - r1(T1, CA1, CB1))

#                         dCB1 = (
#                             ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CB1) /
#                              V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCC1 = (
#                             ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CC1) /
#                              V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCD1 = (
#                             ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CD1) /
#                              V1) + r2(T1, CB1, CC1, CD1))

#                         dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                         dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                         dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                         dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                         dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                         dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                         dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                         dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                         dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                         dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                         dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                         dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                         return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#                     end

#                     # Solve ODE 
#                     tspan = (0.0, dt)
#                     prob = ODEProblem(f, x0, tspan)
#                     soln = solve(prob, Rosenbrock23())

#                     # Obtain the next values for each element
#                     ig.V1[k+1] = soln.u[end][1]
#                     ig.V2[k+1] = soln.u[end][2]
#                     ig.V3[k+1] = soln.u[end][3]
#                     ig.V4[k+1] = soln.u[end][4]
#                     ig.V5[k+1] = soln.u[end][5]

#                     ig.T1[k+1] = soln.u[end][6]
#                     ig.T2[k+1] = soln.u[end][7]
#                     ig.T3[k+1] = soln.u[end][8]
#                     ig.T4[k+1] = soln.u[end][9]
#                     ig.T5[k+1] = soln.u[end][10]

#                     ig.CA1[k+1] = soln.u[end][11]
#                     ig.CB1[k+1] = soln.u[end][12]
#                     ig.CC1[k+1] = soln.u[end][13]
#                     ig.CD1[k+1] = soln.u[end][14]

#                     ig.CA2[k+1] = soln.u[end][15]
#                     ig.CB2[k+1] = soln.u[end][16]
#                     ig.CC2[k+1] = soln.u[end][17]
#                     ig.CD2[k+1] = soln.u[end][18]

#                     ig.CA3[k+1] = soln.u[end][19]
#                     ig.CB3[k+1] = soln.u[end][20]
#                     ig.CC3[k+1] = soln.u[end][21]
#                     ig.CD3[k+1] = soln.u[end][22]

#                     ig.CA4[k+1] = soln.u[end][23]
#                     ig.CB4[k+1] = soln.u[end][24]
#                     ig.CC4[k+1] = soln.u[end][25]
#                     ig.CD4[k+1] = soln.u[end][26]

#                     ig.CA5[k+1] = soln.u[end][27]
#                     ig.CB5[k+1] = soln.u[end][28]
#                     ig.CC5[k+1] = soln.u[end][29]
#                     ig.CD5[k+1] = soln.u[end][30]
#                 end

#             end


#             global cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5 = cmpc()
#             centr_ISE, centr_ISC, centr_PI = getPI(cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5)

#             ex.F1c[mpc_step] = cF1[1]
#             ex.F2c[mpc_step] = cF2[1]
#             ex.F3c[mpc_step] = cF3[1]
#             ex.F4c[mpc_step] = cF4[1]
#             ex.F5c[mpc_step] = cF5[1]
#             ex.F6c[mpc_step] = cF6[1]
#             ex.F7c[mpc_step] = cF7[1]
#             ex.F8c[mpc_step] = cF8[1]
#             ex.F9c[mpc_step] = cF9[1]
#             ex.F10c[mpc_step] = cF10[1]
#             ex.Fr1c[mpc_step] = cFr1[1]
#             ex.Fr2c[mpc_step] = cFr2[1]

#             ex.Q1c[mpc_step] = cQ1[1]
#             ex.Q2c[mpc_step] = cQ2[1]
#             ex.Q3c[mpc_step] = cQ3[1]
#             ex.Q4c[mpc_step] = cQ4[1]
#             ex.Q5c[mpc_step] = cQ5[1]


#             centr_PI_vec[mpc_step] = centr_PI
#             centr_ISE_vec[mpc_step] = centr_ISE
#             centr_ISC_vec[mpc_step] = centr_ISC
#             centr_time[mpc_step] = cmpc_solve_time

#             println("cPI = $(centr_PI)")
#             println("cISE = $(centr_ISE)")
#             println("cISC = $(centr_ISC)")
#             println("time = $(cmpc_solve_time)")
#             println()

#             global dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5 = dmpc()
#             distr_ISE, distr_ISC, distr_PI = getPI(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)

#             ex.F1d[mpc_step] = dF1[1]
#             ex.F2d[mpc_step] = dF2[1]
#             ex.F3d[mpc_step] = dF3[1]
#             ex.F4d[mpc_step] = dF4[1]
#             ex.F5d[mpc_step] = dF5[1]
#             ex.F6d[mpc_step] = dF6[1]
#             ex.F7d[mpc_step] = dF7[1]
#             ex.F8d[mpc_step] = dF8[1]
#             ex.F9d[mpc_step] = dF9[1]
#             ex.F10d[mpc_step] = dF10[1]
#             ex.Fr1d[mpc_step] = dFr1[1]
#             ex.Fr2d[mpc_step] = dFr2[1]

#             ex.Q1d[mpc_step] = dQ1[1]
#             ex.Q2d[mpc_step] = dQ2[1]
#             ex.Q3d[mpc_step] = dQ3[1]
#             ex.Q4d[mpc_step] = dQ4[1]
#             ex.Q5d[mpc_step] = dQ5[1]

#             distr_PI_vec[mpc_step] = distr_PI
#             distr_ISE_vec[mpc_step] = distr_ISE
#             distr_ISC_vec[mpc_step] = distr_ISC
#             distr_time[mpc_step] = dmpc_solve_time

#             println("dPI = $(distr_PI)")
#             println("dISE = $(distr_ISE)")
#             println("dISC = $(distr_ISC)")
#             println("time = $(dmpc_solve_time)")
#             println()

#             ex.PIc[mpc_step] = centr_PI
#             ex.PId[mpc_step] = distr_PI
#             ex.ISEc[mpc_step] = centr_ISE
#             ex.ISEd[mpc_step] = distr_ISE
#             ex.ISCc[mpc_step] = centr_ISC
#             ex.ISCd[mpc_step] = distr_ISC

#             if centr_PI < distr_PI
#                 println("Centralized chosen")
#                 println()
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5)
#                 takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

#                 predict.F1[mpc_step] = cF1[1]
#                 predict.F2[mpc_step] = cF2[1]
#                 predict.F3[mpc_step] = cF3[1]
#                 predict.F4[mpc_step] = cF4[1]
#                 predict.F5[mpc_step] = cF5[1]
#                 predict.F6[mpc_step] = cF6[1]
#                 predict.F7[mpc_step] = cF7[1]
#                 predict.F8[mpc_step] = cF8[1]
#                 predict.F9[mpc_step] = cF9[1]
#                 predict.F10[mpc_step] = cF10[1]
#                 predict.Fr1[mpc_step] = cFr1[1]
#                 predict.Fr2[mpc_step] = cFr2[1]

#                 predict.Q1[mpc_step] = cQ1[1]
#                 predict.Q2[mpc_step] = cQ2[1]
#                 predict.Q3[mpc_step] = cQ3[1]
#                 predict.Q4[mpc_step] = cQ4[1]
#                 predict.Q5[mpc_step] = cQ5[1]
#             else
#                 println("Distributed chosen")
#                 println()
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)
#                 takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

#                 predict.F1[mpc_step] = dF1[1]
#                 predict.F2[mpc_step] = dF2[1]
#                 predict.F3[mpc_step] = dF3[1]
#                 predict.F4[mpc_step] = dF4[1]
#                 predict.F5[mpc_step] = dF5[1]
#                 predict.F6[mpc_step] = dF6[1]
#                 predict.F7[mpc_step] = dF7[1]
#                 predict.F8[mpc_step] = dF8[1]
#                 predict.F9[mpc_step] = dF9[1]
#                 predict.F10[mpc_step] = dF10[1]
#                 predict.Fr1[mpc_step] = dFr1[1]
#                 predict.Fr2[mpc_step] = dFr2[1]

#                 predict.Q1[mpc_step] = dQ1[1]
#                 predict.Q2[mpc_step] = dQ2[1]
#                 predict.Q3[mpc_step] = dQ3[1]
#                 predict.Q4[mpc_step] = dQ4[1]
#                 predict.Q5[mpc_step] = dQ5[1]
#             end

#         end

#     end


#     mhmpc_sim()

#     decision = zeros(P)
#     for i = 1:P
#         if distr_PI_vec[i] <= centr_PI_vec[i]
#             decision[i] = 1
#         end
#     end

#     print(decision)
#     plot(decision)
#     plot(log10.(centr_PI_vec[1:P]), label="Log10 Centralized PI")
#     plot!(log10.(distr_PI_vec[1:P]), label="Distributed", ylabel="Log10 PI")

#     plot(predict.T3)
#     hline!([T3_sp])

#     PI_diff = centr_PI_vec - distr_PI_vec
#     plot(centr_time, label="Centralized time", ylabel="Seconds")
#     plot!(distr_time, label="Distributed time")

#     df = DataFrame(PIc=ex.PIc, ISEc=ex.ISEc, ISCc=ex.ISCc,
#         PId=ex.PId, ISEd=ex.ISEd, ISCd=ex.ISCd,
#         V1=ex.V1, V2=ex.V2, V3=ex.V3, V4=ex.V4, V5=ex.V5,
#         T1=ex.T1, T2=ex.T2, T3=ex.T3, T4=ex.T4, T5=ex.T5,
#         CA1=ex.CA1, CA2=ex.CA2, CA3=ex.CA3, CA4=ex.CA4, CA5=ex.CA5,
#         CB1=ex.CB1, CB2=ex.CB2, CB3=ex.CB3, CB4=ex.CB4, CB5=ex.CB5,
#         CC1=ex.CC1, CC2=ex.CC2, CC3=ex.CC3, CC4=ex.CC4, CC5=ex.CC5,
#         CD1=ex.CD1, CD2=ex.CD2, CD3=ex.CD3, CD4=ex.CD4, CD5=ex.CD5,
#         F1d=ex.F1d, F2d=ex.F2d, F3d=ex.F3d, F4d=ex.F4d, F5d=ex.F5d,
#         F6d=ex.F6d, F7d=ex.F7d, F8d=ex.F8d, F9d=ex.F9d, F10d=ex.F10d,
#         Fr1d=ex.Fr1d, Fr2d=ex.Fr2d,
#         Q1d=ex.Q1d, Q2d=ex.Q2d, Q3d=ex.Q3d, Q4d=ex.Q4d, Q5d=ex.Q5d,
#         F1c=ex.F1c, F2c=ex.F2c, F3c=ex.F3c, F4c=ex.F4c, F5c=ex.F5c,
#         F6c=ex.F6c, F7c=ex.F7c, F8c=ex.F8c, F9c=ex.F9c, F10c=ex.F10c,
#         Fr1c=ex.Fr1c, Fr2c=ex.Fr2c,
#         Q1c=ex.Q1c, Q2c=ex.Q2c, Q3c=ex.Q3c, Q4c=ex.Q4c, Q5c=ex.Q5c
#     )
#     CSV.write("mhmpc-src-results-$(mhmpc_src_instance)-t-7-sec.csv", df)

#     global mhmpc_src_instance = mhmpc_src_instance + 1

# end

# global mhmpc_src_instance = 1

# while mhmpc_src_instance <= 104

#     N = 1200  # Control horizon
#     dt = 1 # Sampling time of the system
#     Nd = 40
#     dtd = 30
#     P = 80

#     global s_path = 0.01
#     global cpu_max = 12.0 # Maximum cpu time for Ipopt
#     global dual_inf_tol = Float64(1 * 10^(0))
#     global opt_tol = Float64(1 * 10^(0))
#     global constr_viol_tol = Float64(1 * 10^(0))
#     global compl_inf_tol = Float64(1 * 10^(0))

#     global c_cpu_max = 12.0 # Maximum cpu time for Ipopt
#     global c_dual_inf_tol = Float64(1 * 10^(0))
#     global c_opt_tol = Float64(1 * 10^(0))
#     global c_constr_viol_tol = Float64(1 * 10^(0))
#     global c_compl_inf_tol = Float64(1 * 10^(0))

#     block_size = Int(dtd/1) - 1
#     k_indices = Int[]  # Initialize an empty array

#     # Loop to generate the pattern until N - 1
#     for start_k in 0:dtd:N-1
#         append!(k_indices, start_k:start_k+block_size-1)
#     end

#     # append!(k_indices, N-1)
#     # Ensure indices do not exceed N - 1
#     global k_indices = filter(x -> x < N, k_indices)

#     mutable struct Weights
#         v::Float64

#         t1::Float64
#         t2::Float64
#         t3::Float64
#         t4::Float64
#         t5::Float64

#         ca1::Float64
#         ca2::Float64
#         ca3::Float64
#         ca4::Float64
#         ca5::Float64

#         cb1::Float64
#         cb2::Float64
#         cb3::Float64
#         cb4::Float64
#         cb5::Float64

#         cc1::Float64
#         cc2::Float64
#         cc3::Float64
#         cc4::Float64
#         cc5::Float64

#         cd1::Float64
#         cd2::Float64
#         cd3::Float64
#         cd4::Float64
#         cd5::Float64

#         f1::Float64
#         f2::Float64
#         f3::Float64
#         f4::Float64
#         f5::Float64
#         f6::Float64
#         f7::Float64
#         f8::Float64
#         f9::Float64
#         f10::Float64
#         fr1::Float64
#         fr2::Float64

#         q1::Float64
#         q2::Float64
#         q3::Float64
#         q4::Float64
#         q5::Float64

#     end

#     mutable struct Decomposition_Trajectory


#         V1::Vector{Float64}
#         V2::Vector{Float64}
#         V3::Vector{Float64}
#         V4::Vector{Float64}
#         V5::Vector{Float64}

#         T1::Vector{Float64}
#         T2::Vector{Float64}
#         T3::Vector{Float64}
#         T4::Vector{Float64}
#         T5::Vector{Float64}

#         CA1::Vector{Float64}
#         CB1::Vector{Float64}
#         CC1::Vector{Float64}
#         CD1::Vector{Float64}

#         CA2::Vector{Float64}
#         CB2::Vector{Float64}
#         CC2::Vector{Float64}
#         CD2::Vector{Float64}

#         CA3::Vector{Float64}
#         CB3::Vector{Float64}
#         CC3::Vector{Float64}
#         CD3::Vector{Float64}

#         CA4::Vector{Float64}
#         CB4::Vector{Float64}
#         CC4::Vector{Float64}
#         CD4::Vector{Float64}

#         CA5::Vector{Float64}
#         CB5::Vector{Float64}
#         CC5::Vector{Float64}
#         CD5::Vector{Float64}

#         F1::Vector{Float64}
#         F2::Vector{Float64}
#         F3::Vector{Float64}
#         F4::Vector{Float64}
#         F5::Vector{Float64}
#         F6::Vector{Float64}
#         F7::Vector{Float64}
#         F8::Vector{Float64}
#         F9::Vector{Float64}
#         F10::Vector{Float64}
#         Fr1::Vector{Float64}
#         Fr2::Vector{Float64}
#         Q1::Vector{Float64}
#         Q2::Vector{Float64}
#         Q3::Vector{Float64}
#         Q4::Vector{Float64}
#         Q5::Vector{Float64}

#     end

#     mutable struct Solve_Strat_Comparison_Trajectory

#         V1::Vector{Float64}
#         V2::Vector{Float64}
#         V3::Vector{Float64}
#         V4::Vector{Float64}
#         V5::Vector{Float64}

#         T1::Vector{Float64}
#         T2::Vector{Float64}
#         T3::Vector{Float64}
#         T4::Vector{Float64}
#         T5::Vector{Float64}

#         CA1::Vector{Float64}
#         CB1::Vector{Float64}
#         CC1::Vector{Float64}
#         CD1::Vector{Float64}

#         CA2::Vector{Float64}
#         CB2::Vector{Float64}
#         CC2::Vector{Float64}
#         CD2::Vector{Float64}

#         CA3::Vector{Float64}
#         CB3::Vector{Float64}
#         CC3::Vector{Float64}
#         CD3::Vector{Float64}

#         CA4::Vector{Float64}
#         CB4::Vector{Float64}
#         CC4::Vector{Float64}
#         CD4::Vector{Float64}

#         CA5::Vector{Float64}
#         CB5::Vector{Float64}
#         CC5::Vector{Float64}
#         CD5::Vector{Float64}

#         F1c::Vector{Float64}
#         F2c::Vector{Float64}
#         F3c::Vector{Float64}
#         F4c::Vector{Float64}
#         F5c::Vector{Float64}
#         F6c::Vector{Float64}
#         F7c::Vector{Float64}
#         F8c::Vector{Float64}
#         F9c::Vector{Float64}
#         F10c::Vector{Float64}
#         Fr1c::Vector{Float64}
#         Fr2c::Vector{Float64}
#         Q1c::Vector{Float64}
#         Q2c::Vector{Float64}
#         Q3c::Vector{Float64}
#         Q4c::Vector{Float64}
#         Q5c::Vector{Float64}

#         F1d::Vector{Float64}
#         F2d::Vector{Float64}
#         F3d::Vector{Float64}
#         F4d::Vector{Float64}
#         F5d::Vector{Float64}
#         F6d::Vector{Float64}
#         F7d::Vector{Float64}
#         F8d::Vector{Float64}
#         F9d::Vector{Float64}
#         F10d::Vector{Float64}
#         Fr1d::Vector{Float64}
#         Fr2d::Vector{Float64}
#         Q1d::Vector{Float64}
#         Q2d::Vector{Float64}
#         Q3d::Vector{Float64}
#         Q4d::Vector{Float64}
#         Q5d::Vector{Float64}

#         PIc::Vector{Float64}
#         ISEc::Vector{Float64}
#         ISCc::Vector{Float64}

#         PId::Vector{Float64}
#         ISEd::Vector{Float64}
#         ISCd::Vector{Float64}
#     end

#     ex = Solve_Strat_Comparison_Trajectory(zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P))


#     data = CSV.read("C:\\Users\\escooper\\git\\bae\\final-formulation\\src-data.csv", DataFrame)
#     instance = data[mhmpc_src_instance, :]

#     # Declare setpoint, initial condition, parameters
#     function init_sp_and_params()

#         global V1_init = instance[1]
#         global V2_init = instance[2]
#         global V3_init = instance[3]
#         global V4_init = instance[4]
#         global V5_init = instance[5]

#         global T1_init = instance[6]
#         global T2_init = instance[7]
#         global T3_init = instance[8]
#         global T4_init = instance[9]
#         global T5_init = instance[10]

#         global CA1_init = instance[11]
#         global CB1_init = instance[16]
#         global CC1_init = instance[21]
#         global CD1_init = instance[26]

#         global CA2_init = instance[12]
#         global CB2_init = instance[17]
#         global CC2_init = instance[22]
#         global CD2_init = instance[27]

#         global CA3_init = instance[13]
#         global CB3_init = instance[18]
#         global CC3_init = instance[23]
#         global CD3_init = instance[28]

#         global CA4_init = instance[14]
#         global CB4_init = instance[19]
#         global CC4_init = instance[24]
#         global CD4_init = instance[29]

#         global CA5_init = instance[15]
#         global CB5_init = instance[20]
#         global CC5_init = instance[25]
#         global CD5_init = instance[30]

#         global V1_sp = instance[1]
#         global V2_sp = instance[2]
#         global V3_sp = instance[3]
#         global V4_sp = instance[4]
#         global V5_sp = instance[5]

#         global T1_sp = instance[31]
#         global T2_sp = instance[32]
#         global T3_sp = instance[33]
#         global T4_sp = instance[34]
#         global T5_sp = instance[35]

#         global CA1_sp = instance[36]
#         global CA2_sp = instance[37]
#         global CA3_sp = instance[38]
#         global CA4_sp = instance[39]
#         global CA5_sp = instance[40]

#         global CB1_sp = instance[41]
#         global CB2_sp = instance[42]
#         global CB3_sp = instance[43]
#         global CB4_sp = instance[44]
#         global CB5_sp = instance[45]

#         global CC1_sp = instance[46]
#         global CC2_sp = instance[47]
#         global CC3_sp = instance[48]
#         global CC4_sp = instance[49]
#         global CC5_sp = instance[50]

#         global CD1_sp = instance[51]
#         global CD2_sp = instance[52]
#         global CD3_sp = instance[53]
#         global CD4_sp = instance[54]
#         global CD5_sp = instance[55]


#         global F1_sp = instance[56]
#         global F2_sp = instance[57]
#         global F3_sp = instance[58]
#         global F4_sp = instance[59]
#         global F5_sp = instance[60]
#         global F6_sp = instance[61]
#         global F7_sp = instance[62]
#         global F8_sp = instance[63]
#         global F9_sp = instance[64]
#         global F10_sp = instance[65]
#         global Fr1_sp = instance[66]
#         global Fr2_sp = instance[67]

#         global Q1_sp = instance[68]
#         global Q2_sp = instance[69]
#         global Q3_sp = instance[70]
#         global Q4_sp = instance[71]
#         global Q5_sp = instance[72]

#         global H_vap_A = 3.073e4
#         global H_vap_B = 1.35e4
#         global H_vap_C = 4.226e4
#         global H_vap_D = 4.55e4

#         global H_ref_A = 7.44e4
#         global H_ref_B = 5.91e4
#         global H_ref_C = 2.01e4
#         global H_ref_D = -2.89e4

#         global Cp_A = 184.6
#         global Cp_B = 59.1
#         global Cp_C = 247
#         global Cp_D = 301.3

#         global CA0 = 1.126e4
#         global CB0 = 2.028e4
#         global CC0 = 8174
#         global CD0 = 6485

#         global Tref = 450
#         global TA0 = 473
#         global TB0 = 473
#         global TD0 = 473


#         global delH_r1 = -1.53e5
#         global delH_r2 = -1.118e5
#         global delH_r3 = 4.141e5

#         global R = 8.314
#     end


#     init_sp_and_params()

#     w = Weights(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

#     # Linear dependence of enthalpies
#     H_A(T) = H_ref_A + Cp_A * (T - Tref)
#     H_B(T) = H_ref_B + Cp_B * (T - Tref)
#     H_C(T) = H_ref_C + Cp_C * (T - Tref)
#     H_D(T) = H_ref_D + Cp_D * (T - Tref)

#     kEB2(T) = 0.152 * exp(-3933 / (R * T))
#     kEB3(T) = 0.490 * exp(-50870 / (R * T))

#     # Volatilities of the species in the seperator
#     alpha_A(T) = 0.0449 * T + 10
#     alpha_B(T) = 0.0260 * T + 10
#     alpha_C(T) = 0.0065 * T + 0.5
#     alpha_D(T) = 0.0058 * T + 0.25

#     # Debugging parameters for dual feasibility:
#     ub_f = 3.0
#     lb_f = 0.1

#     # Reaction rate expressions
#     r1(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * CA^(0.32) * CB^(1.5)
#     r2(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * CB^(2.5) * CC^(0.5)) / (1 + kEB2(T) * CD)
#     r3(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * CA^(1.0218) * CD) / (1 + kEB3(T) * CA)

#     # Reaction rate expressions
#     r1_(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * max(1e-6, CA)^(0.32) * max(1e-6, CB)^(1.5)
#     r2_(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * max(1e-6, CB)^(2.5) * max(1e-6, CC)^(0.5)) / (1 + kEB2(T) * CD)
#     r3_(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * max(1e-6, CA)^(1.0218) * CD) / (1 + kEB3(T) * CA)

#     # Molar flow in the overhead stream
#     MA(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_A(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
#     MB(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_B(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
#     MC(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_C(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))
#     MD(T, F7_, Ci3_, F9_, Ci5_, CA3_, CB3_, CC3_, CD3_, CA5_, CB5_, CC5_, CD5_) = 0.8 * ((alpha_D(T) * (F7_ * Ci3_ + F9_ * Ci5_) * ((F7_ * CA3_ + F9_ * CA5_) + (F7_ * CB3_ + F9_ * CB5_) + (F7_ * CC3_ + F9_ * CC5_) + (F7_ * CD3_ + F9_ * CD5_))) / (alpha_A(T) * (F7_ * CA3_ + F9_ * CA5_) + alpha_B(T) * (F7_ * CB3_ + F9_ * CB5_) + alpha_C(T) * (F7_ * CC3_ + F9_ * CC5_) + alpha_D(T) * (F7_ * CD3_ + F9_ * CD5_)))

#     # Concentration of species in the recycle stream
#     CAr(MA, MB, MC, MD) = MA / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
#     CBr(MA, MB, MC, MD) = MB / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
#     CCr(MA, MB, MC, MD) = MC / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
#     CDr(MA, MB, MC, MD) = MD / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))

#     fix = Decomposition_Trajectory(zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1), zeros(Nd + 1))
#     ig = Decomposition_Trajectory(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))

#     # Initialize fix structure for decomp
#     for set_traj = 1

#         # Set control variables to nominal steady state values
#         fix.F1 .= F1_sp
#         fix.F2 .= F2_sp
#         fix.F3 .= F3_sp
#         fix.F4 .= F4_sp
#         fix.F5 .= F5_sp
#         fix.F6 .= F6_sp
#         fix.F7 .= F7_sp
#         fix.F8 .= F8_sp
#         fix.F9 .= F9_sp
#         fix.F10 .= F10_sp
#         fix.Fr1 .= Fr1_sp
#         fix.Fr2 .= Fr2_sp

#         fix.Q1 .= Q1_sp
#         fix.Q2 .= Q2_sp
#         fix.Q3 .= Q3_sp
#         fix.Q4 .= Q4_sp
#         fix.Q5 .= Q5_sp

#         fix.V1[1] = V1_init
#         fix.V2[1] = V2_init
#         fix.V3[1] = V3_init
#         fix.V4[1] = V4_init
#         fix.V5[1] = V5_init
#         fix.T1[1] = T1_init
#         fix.T2[1] = T2_init
#         fix.T3[1] = T3_init
#         fix.T4[1] = T4_init
#         fix.T5[1] = T5_init

#         fix.CA1[1] = CA1_init
#         fix.CA2[1] = CA2_init
#         fix.CA3[1] = CA3_init
#         fix.CA4[1] = CA4_init
#         fix.CA5[1] = CA5_init
#         fix.CB1[1] = CB1_init
#         fix.CB2[1] = CB2_init
#         fix.CB3[1] = CB3_init
#         fix.CB4[1] = CB4_init
#         fix.CB5[1] = CB5_init
#         fix.CC1[1] = CC1_init
#         fix.CC2[1] = CC2_init
#         fix.CC3[1] = CC3_init
#         fix.CC4[1] = CC4_init
#         fix.CC5[1] = CC5_init
#         fix.CD1[1] = CD1_init
#         fix.CD2[1] = CD2_init
#         fix.CD3[1] = CD3_init
#         fix.CD4[1] = CD4_init
#         fix.CD5[1] = CD5_init

#         for i = 1:Nd
#             global k = i

#             F1 = fix.F1[k]
#             F2 = fix.F2[k]
#             F3 = fix.F3[k]
#             F4 = fix.F4[k]
#             F5 = fix.F5[k]
#             F6 = fix.F6[k]
#             F7 = fix.F7[k]
#             F8 = fix.F8[k]
#             F9 = fix.F9[k]
#             F10 = fix.F10[k]
#             Fr1 = fix.Fr1[k]
#             Fr2 = fix.Fr2[k]
#             Q1 = fix.Q1[k]
#             Q2 = fix.Q2[k]
#             Q3 = fix.Q3[k]
#             Q4 = fix.Q4[k]
#             Q5 = fix.Q5[k]

#             V1 = fix.V1[k]
#             V2 = fix.V2[k]
#             V3 = fix.V3[k]
#             V4 = fix.V4[k]
#             V5 = fix.V5[k]

#             T1 = fix.T1[k]
#             T2 = fix.T2[k]
#             T3 = fix.T3[k]
#             T4 = fix.T4[k]
#             T5 = fix.T5[k]

#             CA1 = fix.CA1[k]
#             CB1 = fix.CB1[k]
#             CC1 = fix.CC1[k]
#             CD1 = fix.CD1[k]

#             CA2 = fix.CA2[k]
#             CB2 = fix.CB2[k]
#             CC2 = fix.CC2[k]
#             CD2 = fix.CD2[k]

#             CA3 = fix.CA3[k]
#             CB3 = fix.CB3[k]
#             CC3 = fix.CC3[k]
#             CD3 = fix.CD3[k]

#             CA4 = fix.CA4[k]
#             CB4 = fix.CB4[k]
#             CC4 = fix.CC4[k]
#             CD4 = fix.CD4[k]

#             CA5 = fix.CA5[k]
#             CB5 = fix.CB5[k]
#             CC5 = fix.CC5[k]
#             CD5 = fix.CD5[k]

#             x0 = [fix.V1[k], fix.V2[k], fix.V3[k], fix.V4[k], fix.V5[k], fix.T1[k], fix.T2[k], fix.T3[k], fix.T4[k], fix.T5[k], fix.CA1[k], fix.CB1[k], fix.CC1[k], fix.CD1[k], fix.CA2[k], fix.CB2[k], fix.CC2[k], fix.CD2[k], fix.CA3[k], fix.CB3[k], fix.CC3[k], fix.CD3[k], fix.CA4[k], fix.CB4[k], fix.CC4[k], fix.CD4[k], fix.CA5[k], fix.CB5[k], fix.CC5[k], fix.CD5[k]]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1 + F2 + Fr2 - F3
#                 dV2 = F3 + F4 - F5
#                 dV3 = F5 + F6 - F7
#                 dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                 dV5 = F10 + Fr1 - F9

#                 dT1 = (
#                     ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                       (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                       (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                       (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                       (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2 + F4 * CB0 * H_B(TB0) +
#                      (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                      (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                      (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                      (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

#                 dT3 = (
#                     ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                       (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                       (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                       (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4
#                         + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5 +
#                       F10 * CD0 * H_D(TD0)
#                       + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                       + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                       + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                       + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CA1)
#                      /
#                      V1) - r1(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CB1) /
#                      V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CC1) /
#                      V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CD1) /
#                      V1) + r2(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                 dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                 dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                 dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                 dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                 dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                 dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                 dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                 dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dtd)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23())

#             # Obtain the next values for each element
#             fix.V1[k+1] = soln.u[end][1]
#             fix.V2[k+1] = soln.u[end][2]
#             fix.V3[k+1] = soln.u[end][3]
#             fix.V4[k+1] = soln.u[end][4]
#             fix.V5[k+1] = soln.u[end][5]

#             fix.T1[k+1] = soln.u[end][6]
#             fix.T2[k+1] = soln.u[end][7]
#             fix.T3[k+1] = soln.u[end][8]
#             fix.T4[k+1] = soln.u[end][9]
#             fix.T5[k+1] = soln.u[end][10]

#             fix.CA1[k+1] = soln.u[end][11]
#             fix.CB1[k+1] = soln.u[end][12]
#             fix.CC1[k+1] = soln.u[end][13]
#             fix.CD1[k+1] = soln.u[end][14]

#             fix.CA2[k+1] = soln.u[end][15]
#             fix.CB2[k+1] = soln.u[end][16]
#             fix.CC2[k+1] = soln.u[end][17]
#             fix.CD2[k+1] = soln.u[end][18]

#             fix.CA3[k+1] = soln.u[end][19]
#             fix.CB3[k+1] = soln.u[end][20]
#             fix.CC3[k+1] = soln.u[end][21]
#             fix.CD3[k+1] = soln.u[end][22]

#             fix.CA4[k+1] = soln.u[end][23]
#             fix.CB4[k+1] = soln.u[end][24]
#             fix.CC4[k+1] = soln.u[end][25]
#             fix.CD4[k+1] = soln.u[end][26]

#             fix.CA5[k+1] = soln.u[end][27]
#             fix.CB5[k+1] = soln.u[end][28]
#             fix.CC5[k+1] = soln.u[end][29]
#             fix.CD5[k+1] = soln.u[end][30]
#         end

#     end

#     # Initialize ig structure for centralized
#     for set_traj = 1

#         # Set control variables to nominal steady state values
#         ig.F1 .= F1_sp
#         ig.F2 .= F2_sp
#         ig.F3 .= F3_sp
#         ig.F4 .= F4_sp
#         ig.F5 .= F5_sp
#         ig.F6 .= F6_sp
#         ig.F7 .= F7_sp
#         ig.F8 .= F8_sp
#         ig.F9 .= F9_sp
#         ig.F10 .= F10_sp
#         ig.Fr1 .= Fr1_sp
#         ig.Fr2 .= Fr2_sp

#         ig.Q1 .= Q1_sp
#         ig.Q2 .= Q2_sp
#         ig.Q3 .= Q3_sp
#         ig.Q4 .= Q4_sp
#         ig.Q5 .= Q5_sp

#         ig.V1[1] = V1_init
#         ig.V2[1] = V2_init
#         ig.V3[1] = V3_init
#         ig.V4[1] = V4_init
#         ig.V5[1] = V5_init
#         ig.T1[1] = T1_init
#         ig.T2[1] = T2_init
#         ig.T3[1] = T3_init
#         ig.T4[1] = T4_init
#         ig.T5[1] = T5_init

#         ig.CA1[1] = CA1_init
#         ig.CA2[1] = CA2_init
#         ig.CA3[1] = CA3_init
#         ig.CA4[1] = CA4_init
#         ig.CA5[1] = CA5_init
#         ig.CB1[1] = CB1_init
#         ig.CB2[1] = CB2_init
#         ig.CB3[1] = CB3_init
#         ig.CB4[1] = CB4_init
#         ig.CB5[1] = CB5_init
#         ig.CC1[1] = CC1_init
#         ig.CC2[1] = CC2_init
#         ig.CC3[1] = CC3_init
#         ig.CC4[1] = CC4_init
#         ig.CC5[1] = CC5_init
#         ig.CD1[1] = CD1_init
#         ig.CD2[1] = CD2_init
#         ig.CD3[1] = CD3_init
#         ig.CD4[1] = CD4_init
#         ig.CD5[1] = CD5_init

#         for i = 1:N
#             global k = i

#             F1 = ig.F1[k]
#             F2 = ig.F2[k]
#             F3 = ig.F3[k]
#             F4 = ig.F4[k]
#             F5 = ig.F5[k]
#             F6 = ig.F6[k]
#             F7 = ig.F7[k]
#             F8 = ig.F8[k]
#             F9 = ig.F9[k]
#             F10 = ig.F10[k]
#             Fr1 = ig.Fr1[k]
#             Fr2 = ig.Fr2[k]
#             Q1 = ig.Q1[k]
#             Q2 = ig.Q2[k]
#             Q3 = ig.Q3[k]
#             Q4 = ig.Q4[k]
#             Q5 = ig.Q5[k]

#             V1 = ig.V1[k]
#             V2 = ig.V2[k]
#             V3 = ig.V3[k]
#             V4 = ig.V4[k]
#             V5 = ig.V5[k]

#             T1 = ig.T1[k]
#             T2 = ig.T2[k]
#             T3 = ig.T3[k]
#             T4 = ig.T4[k]
#             T5 = ig.T5[k]

#             CA1 = ig.CA1[k]
#             CB1 = ig.CB1[k]
#             CC1 = ig.CC1[k]
#             CD1 = ig.CD1[k]

#             CA2 = ig.CA2[k]
#             CB2 = ig.CB2[k]
#             CC2 = ig.CC2[k]
#             CD2 = ig.CD2[k]

#             CA3 = ig.CA3[k]
#             CB3 = ig.CB3[k]
#             CC3 = ig.CC3[k]
#             CD3 = ig.CD3[k]

#             CA4 = ig.CA4[k]
#             CB4 = ig.CB4[k]
#             CC4 = ig.CC4[k]
#             CD4 = ig.CD4[k]

#             CA5 = ig.CA5[k]
#             CB5 = ig.CB5[k]
#             CC5 = ig.CC5[k]
#             CD5 = ig.CD5[k]

#             x0 = [ig.V1[k], ig.V2[k], ig.V3[k], ig.V4[k], ig.V5[k], ig.T1[k], ig.T2[k], ig.T3[k], ig.T4[k], ig.T5[k], ig.CA1[k], ig.CB1[k], ig.CC1[k], ig.CD1[k], ig.CA2[k], ig.CB2[k], ig.CC2[k], ig.CD2[k], ig.CA3[k], ig.CB3[k], ig.CC3[k], ig.CD3[k], ig.CA4[k], ig.CB4[k], ig.CC4[k], ig.CD4[k], ig.CA5[k], ig.CB5[k], ig.CC5[k], ig.CD5[k]]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1 + F2 + Fr2 - F3
#                 dV2 = F3 + F4 - F5
#                 dV3 = F5 + F6 - F7
#                 dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                 dV5 = F10 + Fr1 - F9

#                 dT1 = (
#                     ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                       (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                       (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                       (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                       (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2 + F4 * CB0 * H_B(TB0) +
#                      (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                      (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                      (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                      (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

#                 dT3 = (
#                     ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                       (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                       (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                       (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4
#                         + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5 +
#                       F10 * CD0 * H_D(TD0)
#                       + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                       + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                       + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                       + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CA1)
#                      /
#                      V1) - r1(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CB1) /
#                      V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CC1) /
#                      V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3 * CD1) /
#                      V1) + r2(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                 dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                 dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                 dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                 dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                 dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                 dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                 dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                 dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dt)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23())

#             # Obtain the next values for each element
#             ig.V1[k+1] = soln.u[end][1]
#             ig.V2[k+1] = soln.u[end][2]
#             ig.V3[k+1] = soln.u[end][3]
#             ig.V4[k+1] = soln.u[end][4]
#             ig.V5[k+1] = soln.u[end][5]

#             ig.T1[k+1] = soln.u[end][6]
#             ig.T2[k+1] = soln.u[end][7]
#             ig.T3[k+1] = soln.u[end][8]
#             ig.T4[k+1] = soln.u[end][9]
#             ig.T5[k+1] = soln.u[end][10]

#             ig.CA1[k+1] = soln.u[end][11]
#             ig.CB1[k+1] = soln.u[end][12]
#             ig.CC1[k+1] = soln.u[end][13]
#             ig.CD1[k+1] = soln.u[end][14]

#             ig.CA2[k+1] = soln.u[end][15]
#             ig.CB2[k+1] = soln.u[end][16]
#             ig.CC2[k+1] = soln.u[end][17]
#             ig.CD2[k+1] = soln.u[end][18]

#             ig.CA3[k+1] = soln.u[end][19]
#             ig.CB3[k+1] = soln.u[end][20]
#             ig.CC3[k+1] = soln.u[end][21]
#             ig.CD3[k+1] = soln.u[end][22]

#             ig.CA4[k+1] = soln.u[end][23]
#             ig.CB4[k+1] = soln.u[end][24]
#             ig.CC4[k+1] = soln.u[end][25]
#             ig.CD4[k+1] = soln.u[end][26]

#             ig.CA5[k+1] = soln.u[end][27]
#             ig.CB5[k+1] = soln.u[end][28]
#             ig.CC5[k+1] = soln.u[end][29]
#             ig.CD5[k+1] = soln.u[end][30]
#         end

#     end

#     function cmpc()

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin
#             # State variables

#             # Volume, state, [=] m3
#             V1[k=0:N], (lower_bound=0.2 * V1_sp, upper_bound=1.8 * V1_sp)
#             V2[k=0:N], (lower_bound=0.2 * V2_sp, upper_bound=1.8 * V2_sp)
#             V3[k=0:N], (lower_bound=0.2 * V3_sp, upper_bound=1.8 * V3_sp)
#             V4[k=0:N], (lower_bound=0.2 * V4_sp, upper_bound=1.8 * V4_sp)
#             V5[k=0:N], (lower_bound=0.2 * V5_sp, upper_bound=1.8 * V5_sp)

#             # Temperature, state, [=] K
#             T1[k=0:N], (lower_bound=0.2 * T1_sp, upper_bound=1.8 * T1_sp)
#             T2[k=0:N], (lower_bound=0.2 * T2_sp, upper_bound=1.8 * T2_sp)
#             T3[k=0:N], (lower_bound=0.2 * T3_sp, upper_bound=1.8 * T3_sp)
#             T4[k=0:N], (lower_bound=0.2 * T4_sp, upper_bound=1.8 * T4_sp)
#             T5[k=0:N], (lower_bound=0.2 * T5_sp, upper_bound=1.8 * T5_sp)

#             # Concentration, state, [=] mol/m3
#             CA1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CA5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             CB1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CB5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             CC1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CC5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             CD1[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD2[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD3[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD4[k=0:N], (lower_bound=1e-4, upper_bound=20000)
#             CD5[k=0:N], (lower_bound=1e-4, upper_bound=20000)

#             F1[k=0:N], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp)
#             F2[k=0:N], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp)
#             F3[k=0:N], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp)
#             F4[k=0:N], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp)
#             F5[k=0:N], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp)
#             F6[k=0:N], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp)
#             F7[k=0:N], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp)
#             F8[k=0:N], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp)
#             F9[k=0:N], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp)
#             F10[k=0:N], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp)

#             Fr1[k=0:N], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp)
#             Fr2[k=0:N], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp)

#             Q1[k=0:N], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
#             Q2[k=0:N], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)
#             Q3[k=0:N], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)
#             Q4[k=0:N], (start=Q4_sp, upper_bound=1.8 * Q4_sp, lower_bound=1e-6)
#             Q5[k=0:N], (start=Q5_sp, upper_bound=1.8 * Q5_sp, lower_bound=1e-6)

#         end

#         for k = 0:N
#             set_start_value(V1[k], ig.V1[k+1])
#             set_start_value(V2[k], ig.V2[k+1])
#             set_start_value(V3[k], ig.V3[k+1])
#             set_start_value(V4[k], ig.V4[k+1])
#             set_start_value(V5[k], ig.V5[k+1])

#             set_start_value(T1[k], ig.T1[k+1])
#             set_start_value(T2[k], ig.T2[k+1])
#             set_start_value(T3[k], ig.T3[k+1])
#             set_start_value(T4[k], ig.T4[k+1])
#             set_start_value(T5[k], ig.T5[k+1])

#             set_start_value(CA1[k], ig.CA1[k+1])
#             set_start_value(CA2[k], ig.CA2[k+1])
#             set_start_value(CA3[k], ig.CA3[k+1])
#             set_start_value(CA4[k], ig.CA4[k+1])
#             set_start_value(CA5[k], ig.CA5[k+1])

#             set_start_value(CB1[k], ig.CB1[k+1])
#             set_start_value(CB2[k], ig.CB2[k+1])
#             set_start_value(CB3[k], ig.CB3[k+1])
#             set_start_value(CB4[k], ig.CB4[k+1])
#             set_start_value(CB5[k], ig.CB5[k+1])

#             set_start_value(CC1[k], ig.CC1[k+1])
#             set_start_value(CC2[k], ig.CC2[k+1])
#             set_start_value(CC3[k], ig.CC3[k+1])
#             set_start_value(CC4[k], ig.CC4[k+1])
#             set_start_value(CC5[k], ig.CC5[k+1])

#             set_start_value(CD1[k], ig.CD1[k+1])
#             set_start_value(CD2[k], ig.CD2[k+1])
#             set_start_value(CD3[k], ig.CD3[k+1])
#             set_start_value(CD4[k], ig.CD4[k+1])
#             set_start_value(CD5[k], ig.CD5[k+1])

#             set_start_value(F1[k], F1_sp)
#             set_start_value(F2[k], F2_sp)
#             set_start_value(F3[k], F3_sp)
#             set_start_value(F4[k], F4_sp)
#             set_start_value(F5[k], F5_sp)
#             set_start_value(F6[k], F6_sp)
#             set_start_value(F7[k], F7_sp)
#             set_start_value(F8[k], F8_sp)
#             set_start_value(F9[k], F9_sp)
#             set_start_value(F10[k], F10_sp)
#             set_start_value(Fr1[k], Fr1_sp)
#             set_start_value(Fr2[k], Fr2_sp)

#         end

#         for k = 0:N
#             JuMP.fix(Q1[k], Q1_sp; force=true)
#             JuMP.fix(Q2[k], Q2_sp; force=true)
#             JuMP.fix(Q3[k], Q3_sp; force=true)
#             JuMP.fix(Q4[k], Q4_sp; force=true)
#             JuMP.fix(Q5[k], Q5_sp; force=true)

#             JuMP.fix(F7[k], F7_sp; force=true)
#             # JuMP.fix(F8[k], F8_sp; force=true)
#             # JuMP.fix(F9[k], F9_sp; force=true)
#             # JuMP.fix(F10[k], F10_sp; force=true)
#             # JuMP.fix(Fr1[k], Fr1_sp; force=true)
#             JuMP.fix(Fr2[k], Fr2_sp; force=true)
#         end

#         @constraints mpc begin
#             # Initial condition
#             V1_inital, V1[0] == V1_init
#             V2_inital, V2[0] == V2_init
#             V3_inital, V3[0] == V3_init
#             V4_inital, V4[0] == V4_init
#             V5_inital, V5[0] == V5_init

#             T1_inital, T1[0] == T1_init
#             T2_inital, T2[0] == T2_init
#             T3_inital, T3[0] == T3_init
#             T4_inital, T4[0] == T4_init
#             T5_inital, T5[0] == T5_init

#             CA1_initial, CA1[0] == CA1_init
#             CA2_initial, CA2[0] == CA2_init
#             CA3_initial, CA3[0] == CA3_init
#             CA4_initial, CA4[0] == CA4_init
#             CA5_initial, CA5[0] == CA5_init

#             CB1_initial, CB1[0] == CB1_init
#             CB2_initial, CB2[0] == CB2_init
#             CB3_initial, CB3[0] == CB3_init
#             CB4_initial, CB4[0] == CB4_init
#             CB5_initial, CB5[0] == CB5_init

#             CC1_initial, CC1[0] == CC1_init
#             CC2_initial, CC2[0] == CC2_init
#             CC3_initial, CC3[0] == CC3_init
#             CC4_initial, CC4[0] == CC4_init
#             CC5_initial, CC5[0] == CC5_init

#             CD1_initial, CD1[0] == CD1_init
#             CD2_initial, CD2[0] == CD2_init
#             CD3_initial, CD3[0] == CD3_init
#             CD4_initial, CD4[0] == CD4_init
#             CD5_initial, CD5[0] == CD5_init

#             F1_hold[k in k_indices], F1[k] == F1[k+1]
#             F2_hold[k in k_indices], F2[k] == F2[k+1]
#             F3_hold[k in k_indices], F3[k] == F3[k+1]
#             F4_hold[k in k_indices], F4[k] == F4[k+1]
#             F5_hold[k in k_indices], F5[k] == F5[k+1]
#             F6_hold[k in k_indices], F6[k] == F6[k+1]
#             F7_hold[k in k_indices], F7[k] == F7[k+1]
#             F8_hold[k in k_indices], F8[k] == F8[k+1]
#             F9_hold[k in k_indices], F9[k] == F9[k+1]
#             F10_hold[k in k_indices], F10[k] == F10[k+1]
#             Fr1_hold[k in k_indices], Fr1[k] == Fr1[k+1]
#             Fr2_hold[k in k_indices], Fr2[k] == Fr2[k+1]

#             Q1_hold[k in k_indices], Q1[k] == Q1[k+1]
#             Q2_hold[k in k_indices], Q2[k] == Q2[k+1]
#             Q3_hold[k in k_indices], Q3[k] == Q3[k+1]
#             Q4_hold[k in k_indices], Q4[k] == Q4[k+1]
#             Q5_hold[k in k_indices], Q5[k] == Q5[k+1]


#         end

#         @NLconstraints mpc begin
#             # NLconstraints are the differential equations that describe the dynamics of the system
#             dV1_dt[k=0:N-1], V1[k] + (F1[k] + F2[k] + Fr2[k] - F3[k]) * dt == V1_sp

#             dT1_dt[k=0:N-1], T1[k] + (
#                 ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                   (Fr2[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F3[k] * CA1[k] * H_A(T1[k])) +
#                   (Fr2[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F3[k] * CB1[k] * H_B(T1[k])) +
#                   (Fr2[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F3[k] * CC1[k] * H_C(T1[k])) +
#                   (Fr2[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F3[k] * CD1[k] * H_D(T1[k])))
#                  /
#                  (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
#                 (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
#                 (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dt == T1[k+1]

#             dCA1_dt[k=0:N-1], CA1[k] + (
#                 ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CA1[k])
#                  /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dt == CA1[k+1]

#             dCB1_dt[k=0:N-1], CB1[k] + (
#                 ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CB1[k]) /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CB1[k+1]

#             dCC1_dt[k=0:N-1], CC1[k] + (
#                 ((Fr2[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CC1[k]) /
#                  V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CC1[k+1]

#             dCD1_dt[k=0:N-1], CD1[k] + (
#                 ((Fr2[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]))
#                   -
#                   F3[k] * CD1[k]) /
#                  V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dt == CD1[k+1]

#             dV2_dt[k=0:N-1], V2[k] + (F3[k] + F4[k] - F5[k]) * dt == V2_sp

#             dT2_dt[k=0:N-1], T2[k] + (
#                 (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                  (F3[k] * CA1[k] * H_A(T1[k]) - F5[k] * CA2[k] * H_A(T2[k])) +
#                  (F3[k] * CB1[k] * H_B(T1[k]) - F5[k] * CB2[k] * H_B(T2[k])) +
#                  (F3[k] * CC1[k] * H_C(T1[k]) - F5[k] * CC2[k] * H_C(T2[k])) +
#                  (F3[k] * CD1[k] * H_D(T1[k]) - F5[k] * CD2[k] * H_D(T2[k])))
#                 /
#                 (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
#                 +
#                 (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
#                 /
#                 (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dt == T2[k+1]


#             dCA2_dt[k=0:N-1], CA2[k] + (((F3[k] * CA1[k] - F5[k] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dt == CA2[k+1]
#             dCB2_dt[k=0:N-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - F5[k] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CB2[k+1]
#             dCC2_dt[k=0:N-1], CC2[k] + ((F3[k] * CC1[k] - F5[k] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CC2[k+1]
#             dCD2_dt[k=0:N-1], CD2[k] + ((F3[k] * CD1[k] - F5[k] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dt == CD2[k+1]

#             dV3_dt[k=0:N-1], V3[k] + (F5[k] + F6[k] - F7[k]) * dt == V3_sp


#             dT3_dt[k=0:N-1], T3[k] + (
#                 ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2[k] * H_A(T2[k]) - F7[k] * CA3[k] * H_A(T3[k])) +
#                   (F5[k] * CB2[k] * H_B(T2[k]) - F7[k] * CB3[k] * H_B(T3[k])) +
#                   (F5[k] * CC2[k] * H_C(T2[k]) - F7[k] * CC3[k] * H_C(T3[k])) +
#                   (F5[k] * CD2[k] * H_D(T2[k]) - F7[k] * CD3[k] * H_D(T3[k])))
#                  /
#                  (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
#                 +
#                 (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
#                 /
#                 (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dt == T3[k+1]

#             dCA3_dt[k=0:N-1], CA3[k] + (((F5[k] * CA2[k] - F7[k] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dt == CA3[k+1]
#             dCB3_dt[k=0:N-1], CB3[k] + (((F5[k] * CB2[k] + F6[k] * CB0 - F7[k] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CB3[k+1]
#             dCC3_dt[k=0:N-1], CC3[k] + ((F5[k] * CC2[k] - F7[k] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CC3[k+1]
#             dCD3_dt[k=0:N-1], CD3[k] + ((F5[k] * CD2[k] - F7[k] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dt == CD3[k+1]

#             dV4_dt[k=0:N-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dt == V4_sp

#             dT4_dt[k=0:N-1], T4[k] +
#                              ((Q4[k]
#                                + (F7[k] * CA3[k] * H_A(T3[k]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
#                                + (F7[k] * CB3[k] * H_B(T3[k]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
#                                + (F7[k] * CC3[k] * H_C(T3[k]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
#                                + (F7[k] * CD3[k] * H_D(T3[k]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
#                               /
#                               (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dt == T4[k+1]

#             dCA4_dt[k=0:N-1], CA4[k] + ((F7[k] * CA3[k] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dt == CA4[k+1]
#             dCB4_dt[k=0:N-1], CB4[k] + ((F7[k] * CB3[k] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dt == CB4[k+1]
#             dCC4_dt[k=0:N-1], CC4[k] + ((F7[k] * CC3[k] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dt == CC4[k+1]
#             dCD4_dt[k=0:N-1], CD4[k] + ((F7[k] * CD3[k] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dt == CD4[k+1]

#             dV5_dt[k=0:N-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dt == V5_sp

#             dT5_dt[k=0:N-1], T5[k] + (
#                 ((Q5[k] +
#                   F10[k] * CD0 * H_D(TD0)
#                   + (Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
#                   + (Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
#                   + (Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
#                   + (Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
#                  /
#                  (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
#                 +
#                 ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
#                  /
#                  (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dt == T5[k+1]

#             dCA5_dt[k=0:N-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dt == CA5[k+1]
#             dCB5_dt[k=0:N-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dt == CB5[k+1]
#             dCC5_dt[k=0:N-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dt == CC5[k+1]
#             dCD5_dt[k=0:N-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], CA3[k], F9[k], CA5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], CB3[k], F9[k], CB5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], CC3[k], F9[k], CC5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], CD3[k], F9[k], CD5[k], CA3[k], CB3[k], CC3[k], CD3[k], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dt == CD5[k+1]

#             # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + Fr2[k] - F3[k] == -(V1[k] - V1_sp) / 200
#             # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / 200
#             # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / 200
#             # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
#             # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200

#         end


#         @NLobjective(mpc, Min, sum(
#             w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 + w.v * (V3[k] - V3_sp)^2 + w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
#             w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
#             w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
#             w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 + w.cb3 * (CB3[k] - CB3_sp)^2 + w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
#             w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 + w.cc3 * (CC3[k] - CC3_sp)^2 + w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
#             w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2 + w.cd3 * (CD3[k] - CD3_sp)^2 + w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
#             for k = 0:N)
#                                +
#                                sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 + w.f5 * (F5[k] - F5_sp)^2 +
#                                    w.f6 * (F6[k] - F6_sp)^2 + w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 + w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 + w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
#                                    w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2 for k = 0:N)
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", c_opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", c_dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", c_constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", c_compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", c_cpu_max)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global cmpc_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

#         F1 = Vector(JuMP.value.(F1))
#         F2 = Vector(JuMP.value.(F2))
#         F3 = Vector(JuMP.value.(F3))
#         F4 = Vector(JuMP.value.(F4))
#         Q1 = Vector(JuMP.value.(Q1))
#         Q2 = Vector(JuMP.value.(Q2))

#         F5 = Vector(JuMP.value.(F5))
#         F6 = Vector(JuMP.value.(F6))
#         Q3 = Vector(JuMP.value.(Q3))

#         F7 = Vector(JuMP.value.(F7))
#         F8 = Vector(JuMP.value.(F8))
#         F9 = Vector(JuMP.value.(F9))
#         F10 = Vector(JuMP.value.(F10))
#         Fr1 = Vector(JuMP.value.(Fr1))
#         Fr2 = Vector(JuMP.value.(Fr2))
#         Q4 = Vector(JuMP.value.(Q4))
#         Q5 = Vector(JuMP.value.(Q5))

#         return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5

#     end

#     function dmpc_1()

#         # Control variables: F1, F2, F3, F4, Q1, Q2
#         # State variables: All state variables associated with CSTR-1 and CSTR-2

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin

#             # State variables

#             # Volume, state, [=] m3
#             V1[k=0:Nd], (lower_bound=0.2 * V1_sp, upper_bound=1.8 * V1_sp, start=V1_sp)
#             V2[k=0:Nd], (lower_bound=0.2 * V2_sp, upper_bound=1.8 * V2_sp, start=V2_sp)

#             # Temperature, state, [=] K
#             T1[k=0:Nd], (lower_bound=0.2 * T1_sp, upper_bound=1.8 * T1_sp, start=T1_sp)
#             T2[k=0:Nd], (lower_bound=0.2 * T2_sp, upper_bound=1.8 * T2_sp, start=T2_sp)

#             # Concentration, state, [=] mol/m3
#             CA1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA1_sp)
#             CA2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA2_sp)

#             CB1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB1_sp)
#             CB2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB2_sp)

#             CC1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC1_sp)
#             CC2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC2_sp)

#             CD1[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD1_sp)
#             CD2[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD2_sp)

#             F1[k=0:Nd], (lower_bound=lb_f * F1_sp, upper_bound=ub_f * F1_sp, start=F1_sp)
#             F2[k=0:Nd], (lower_bound=lb_f * F2_sp, upper_bound=ub_f * F2_sp, start=F2_sp)
#             F3[k=0:Nd], (lower_bound=lb_f * F3_sp, upper_bound=ub_f * F3_sp, start=F3_sp)
#             F4[k=0:Nd], (lower_bound=lb_f * F4_sp, upper_bound=ub_f * F4_sp, start=F4_sp)

#             Q1[k=0:Nd], (start=Q1_sp, lower_bound=1.8 * Q1_sp, upper_bound=1e-6)
#             Q2[k=0:Nd], (start=Q2_sp, lower_bound=1.8 * Q2_sp, upper_bound=1e-6)

#         end


#         # for k = 0:N
#         #     JuMP.fix(F1[k], F1_sp; force=true)
#         #     JuMP.fix(F2[k], F2_sp; force=true)
#         #     JuMP.fix(F3[k], F3_sp; force=true)
#         #     JuMP.fix(F4[k], F4_sp; force=true)
#         # end


#         for k = 0:Nd
#             JuMP.fix(Q1[k], Q1_sp; force=true)
#             JuMP.fix(Q2[k], Q2_sp; force=true)
#             # JuMP.fix(Q3[k], Q3_sp; force=true)
#             # JuMP.fix(Q4[k], Q4_sp; force=true)
#             # JuMP.fix(Q5[k], Q5_sp; force=true)
#         end

#         for k = 0:Nd
#             set_start_value(V1[k], fix.V1[k+1])
#             set_start_value(V2[k], fix.V2[k+1])

#             set_start_value(T1[k], fix.T1[k+1])
#             set_start_value(T2[k], fix.T2[k+1])

#             set_start_value(CA1[k], fix.CA1[k+1])
#             set_start_value(CA2[k], fix.CA2[k+1])

#             set_start_value(CB1[k], fix.CB1[k+1])
#             set_start_value(CB2[k], fix.CB2[k+1])

#             set_start_value(CC1[k], fix.CC1[k+1])
#             set_start_value(CC2[k], fix.CC2[k+1])

#             set_start_value(CD1[k], fix.CD1[k+1])
#             set_start_value(CD2[k], fix.CD2[k+1])

#             set_start_value(F1[k], fix.F1[k+1])
#             set_start_value(F2[k], fix.F2[k+1])
#             set_start_value(F3[k], fix.F3[k+1])
#             set_start_value(F4[k], fix.F4[k+1])

#             set_start_value(Q1[k], fix.Q1[k+1])
#             set_start_value(Q2[k], fix.Q2[k+1])
#         end

#         @constraints mpc begin

#             # Initial condition
#             V1_inital, V1[0] == V1_init
#             V2_inital, V2[0] == V2_init

#             T1_inital, T1[0] == T1_init
#             T2_inital, T2[0] == T2_init

#             CA1_initial, CA1[0] == CA1_init
#             CA2_initial, CA2[0] == CA2_init

#             CB1_initial, CB1[0] == CB1_init
#             CB2_initial, CB2[0] == CB2_init

#             CC1_initial, CC1[0] == CC1_init
#             CC2_initial, CC2[0] == CC2_init

#             CD1_initial, CD1[0] == CD1_init
#             CD2_initial, CD2[0] == CD2_init

#         end

#         @NLconstraints mpc begin
#             # NLconstraints are the differential equations that describe the dynamics of the system
#             dV1_dt[k=0:Nd-1], V1[k] + (F1[k] + F2[k] + fix.Fr2[k+1] - F3[k]) * dtd == V1_sp

#             dT1_dt[k=0:Nd-1], T1[k] + (
#                 ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                   (fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_A(fix.T4[k+1]) - F3[k] * CA1[k] * H_A(T1[k])) +
#                   (fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_B(fix.T4[k+1]) - F3[k] * CB1[k] * H_B(T1[k])) +
#                   (fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_C(fix.T4[k+1]) - F3[k] * CC1[k] * H_C(T1[k])) +
#                   (fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1])) * H_D(fix.T4[k+1]) - F3[k] * CD1[k] * H_D(T1[k])))
#                  /
#                  (CA1[k] * Cp_A * V1[k] + CB1[k] * Cp_B * V1[k] + CC1[k] * Cp_C * V1[k] + CD1[k] * Cp_D * V1[k])) +
#                 (-delH_r1 * r1(T1[k], CA1[k], CB1[k]) - delH_r2 * r2(T1[k], CB1[k], CC1[k], CD1[k])) /
#                 (CA1[k] * Cp_A + CB1[k] * Cp_B + CC1[k] * Cp_C + CD1[k] * Cp_D)) * dtd == T1[k+1]

#             dCA1_dt[k=0:Nd-1], CA1[k] + (
#                 ((F1[k] * CA0 + fix.Fr2[k+1] * CAr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CA1[k])
#                  /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k])) * dtd == CA1[k+1]

#             dCB1_dt[k=0:Nd-1], CB1[k] + (
#                 ((F2[k] * CB0 + fix.Fr2[k+1] * CBr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CB1[k]) /
#                  V1[k]) - r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CB1[k+1]

#             dCC1_dt[k=0:Nd-1], CC1[k] + (
#                 ((fix.Fr2[k+1] * CCr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CC1[k]) /
#                  V1[k]) + r1(T1[k], CA1[k], CB1[k]) - r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CC1[k+1]

#             dCD1_dt[k=0:Nd-1], CD1[k] + (
#                 ((fix.Fr2[k+1] * CDr(MA(fix.T4[k+1], fix.F7[k+1], fix.CA3[k+1], fix.F9[k+1], fix.CA5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MB(fix.T4[k+1], fix.F7[k+1], fix.CB3[k+1], fix.F9[k+1], fix.CB5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MC(fix.T4[k+1], fix.F7[k+1], fix.CC3[k+1], fix.F9[k+1], fix.CC5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]), MD(fix.T4[k+1], fix.F7[k+1], fix.CD3[k+1], fix.F9[k+1], fix.CD5[k+1], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], fix.CA5[k+1], fix.CB5[k+1], fix.CC5[k+1], fix.CD5[k+1]))
#                   -
#                   F3[k] * CD1[k]) /
#                  V1[k]) + r2(T1[k], CB1[k], CC1[k], CD1[k])) * dtd == CD1[k+1]

#             dV2_dt[k=0:Nd-1], V2[k] + (F3[k] + F4[k] - fix.F5[k+1]) * dtd == V2_sp

#             dT2_dt[k=0:Nd-1], T2[k] + (
#                 (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                  (F3[k] * CA1[k] * H_A(T1[k]) - fix.F5[k+1] * CA2[k] * H_A(T2[k])) +
#                  (F3[k] * CB1[k] * H_B(T1[k]) - fix.F5[k+1] * CB2[k] * H_B(T2[k])) +
#                  (F3[k] * CC1[k] * H_C(T1[k]) - fix.F5[k+1] * CC2[k] * H_C(T2[k])) +
#                  (F3[k] * CD1[k] * H_D(T1[k]) - fix.F5[k+1] * CD2[k] * H_D(T2[k])))
#                 /
#                 (CA2[k] * Cp_A * V2[k] + CB2[k] * Cp_B * V2[k] + CC2[k] * Cp_C * V2[k] + CD2[k] * Cp_D * V2[k])
#                 +
#                 (-delH_r1 * r1(T2[k], CA2[k], CB2[k]) - delH_r2 * r2(T2[k], CB2[k], CC2[k], CD2[k]))
#                 /
#                 (CA2[k] * Cp_A + CB2[k] * Cp_B + CC2[k] * Cp_C + CD2[k] * Cp_D)) * dtd == T2[k+1]


#             dCA2_dt[k=0:Nd-1], CA2[k] + (((F3[k] * CA1[k] - fix.F5[k+1] * CA2[k]) / V2[k]) - r1(T2[k], CA2[k], CB2[k])) * dtd == CA2[k+1]
#             dCB2_dt[k=0:Nd-1], CB2[k] + ((F3[k] * CB1[k] + F4[k] * CB0 - fix.F5[k+1] * CB2[k]) / V2[k] - r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CB2[k+1]
#             dCC2_dt[k=0:Nd-1], CC2[k] + ((F3[k] * CC1[k] - fix.F5[k+1] * CC2[k]) / V2[k] + r1(T2[k], CA2[k], CB2[k]) - r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CC2[k+1]
#             dCD2_dt[k=0:Nd-1], CD2[k] + ((F3[k] * CD1[k] - fix.F5[k+1] * CD2[k]) / V2[k] + r2(T2[k], CB2[k], CC2[k], CD2[k])) * dtd == CD2[k+1]

#             # volHoldUp11[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] <= (-(V1[k] - V1_sp) / 200) + s_path
#             # volHoldUp12[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] >= (-(V1[k] - V1_sp) / 200) - s_path
#             # volHoldUp21[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] <= (-(V2[k] - V2_sp) / 200) + s_path
#             # volHoldUp22[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] >= (-(V2[k] - V2_sp) / 200) - s_path

#             # volHoldUp1[k=0:Nd-1], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / 200
#             # volHoldUp2[k=0:Nd-1], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / 200

#             # volDec1[k=0:N-1], (V1[k+1] - V1_sp) <= 0.8 * (V1[k] - V1_sp)
#             # volDec2[k=0:N-1], (V2[k+1] - V2_sp) <= 0.8 * (V2[k] - V2_sp)

#             # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + fix.Fr2[k+1] - F3[k] == -(V1[k] - V1_sp) / 200
#             # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - fix.F5[k+1] == -(V2[k] - V2_sp) / 200

#         end



#         @NLobjective(mpc, Min, 1e-5 * sum(
#             w.v * (V1[k] - V1_sp)^2 + w.v * (V2[k] - V2_sp)^2 +
#             w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 +
#             w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 +
#             w.cb1 * (CB1[k] - CB1_sp)^2 + w.cb2 * (CB2[k] - CB2_sp)^2 +
#             w.cc1 * (CC1[k] - CC1_sp)^2 + w.cc2 * (CC2[k] - CC2_sp)^2 +
#             w.cd1 * (CD1[k] - CD1_sp)^2 + w.cd2 * (CD2[k] - CD2_sp)^2
#             for k = 0:Nd) +
#                                1e-5 * sum(
#             w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
#             w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 for k = 0:Nd
#         )
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
#         # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
#         # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
#         # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
#         # set_optimizer_attribute(mpc, "mu_init", 1e-3)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global dmpc1_solve_time = MOI.get(mpc, MOI.SolveTimeSec())

#         F1 = Vector(JuMP.value.(F1))
#         F2 = Vector(JuMP.value.(F2))
#         F3 = Vector(JuMP.value.(F3))
#         F4 = Vector(JuMP.value.(F4))
#         Q1 = Vector(JuMP.value.(Q1))
#         Q2 = Vector(JuMP.value.(Q2))

#         V1 = Vector(JuMP.value.(V1))
#         V2 = Vector(JuMP.value.(V2))

#         T1 = Vector(JuMP.value.(T1))
#         T2 = Vector(JuMP.value.(T2))

#         CA1 = Vector(JuMP.value.(CA1))
#         CA2 = Vector(JuMP.value.(CA2))

#         CB1 = Vector(JuMP.value.(CB1))
#         CB2 = Vector(JuMP.value.(CB2))

#         CC1 = Vector(JuMP.value.(CC1))
#         CC2 = Vector(JuMP.value.(CC2))

#         CD1 = Vector(JuMP.value.(CD1))
#         CD2 = Vector(JuMP.value.(CD2))

#         return F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2
#     end

#     function dmpc_2()

#         # Control variables: F5, F6, Q3
#         # State variables: All state variables associated with CSTR-3

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin
#             # State variables

#             # Volume, state, [=] m3    
#             V3[k=0:Nd], (lower_bound=0.2 * V3_sp, upper_bound=1.8 * V3_sp, start=V3_sp)

#             # Temperature, state, [=] K
#             T3[k=0:Nd], (lower_bound=0.2 * T3_sp, upper_bound=1.8 * T3_sp, start=T3_sp)

#             # Concentration, state, [=] mol/m3
#             CA3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA3_sp)

#             CB3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB3_sp)

#             CC3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC3_sp)

#             CD3[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD3_sp)

#             F5[k=0:Nd], (lower_bound=lb_f * F5_sp, upper_bound=ub_f * F5_sp, start=F5_sp)
#             F6[k=0:Nd], (lower_bound=lb_f * F6_sp, upper_bound=ub_f * F6_sp, start=F6_sp)

#             Q3[k=0:Nd], (start=Q3_sp, lower_bound=1.8 * Q3_sp, upper_bound=1e-6)


#         end

#         for k = 0:Nd
#             # JuMP.fix(F5[k], F5_sp; force=true)
#             # JuMP.fix(F6[k], F6_sp; force=true)
#         end

#         for k = 0:Nd
#             # JuMP.fix(Q1[k], Q1_sp; force=true)
#             # JuMP.fix(Q2[k], Q2_sp; force=true)
#             JuMP.fix(Q3[k], Q3_sp; force=true)
#             # JuMP.fix(Q4[k], Q4_sp; force=true)
#             # JuMP.fix(Q5[k], Q5_sp; force=true)
#         end

#         for k = 0:Nd
#             set_start_value(V3[k], fix.V3[k+1])

#             set_start_value(T3[k], fix.T3[k+1])

#             set_start_value(CA3[k], fix.CA3[k+1])

#             set_start_value(CB3[k], fix.CB3[k+1])

#             set_start_value(CC3[k], fix.CC3[k+1])

#             set_start_value(CD3[k], fix.CD3[k+1])

#             set_start_value(F5[k], fix.F5[k+1])
#             set_start_value(F6[k], fix.F6[k+1])
#         end

#         @constraints mpc begin
#             # Initial condition
#             V3_inital, V3[0] == V3_init

#             T3_inital, T3[0] == T3_init

#             CA3_initial, CA3[0] == CA3_init

#             CB3_initial, CB3[0] == CB3_init

#             CC3_initial, CC3[0] == CC3_init

#             CD3_initial, CD3[0] == CD3_init


#             # volDec3[k=0:N-1], (V3[k+1] - V3_sp) <= 0.8 * (V3[k] - V3_sp)

#         end

#         @NLconstraints mpc begin

#             dV3_dt[k=0:Nd-1], V3[k] + (F5[k] + F6[k] - fix.F7[k+1]) * dtd == V3_sp

#             dT3_dt[k=0:Nd-1], T3[k] + (
#                 ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * fix.CA2[k+1] * H_A(fix.T2[k+1]) - fix.F7[k+1] * CA3[k] * H_A(T3[k])) +
#                   (F5[k] * fix.CB2[k+1] * H_B(fix.T2[k+1]) - fix.F7[k+1] * CB3[k] * H_B(T3[k])) +
#                   (F5[k] * fix.CC2[k+1] * H_C(fix.T2[k+1]) - fix.F7[k+1] * CC3[k] * H_C(T3[k])) +
#                   (F5[k] * fix.CD2[k+1] * H_D(fix.T2[k+1]) - fix.F7[k+1] * CD3[k] * H_D(T3[k])))
#                  /
#                  (CA3[k] * Cp_A * V3[k] + CB3[k] * Cp_B * V3[k] + CC3[k] * Cp_C * V3[k] + CD3[k] * Cp_D * V3[k]))
#                 +
#                 (-delH_r1 * r1(T3[k], CA3[k], CB3[k]) - delH_r2 * r2(T3[k], CB3[k], CC3[k], CD3[k]))
#                 /
#                 (CA3[k] * Cp_A + CB3[k] * Cp_B + CC3[k] * Cp_C + CD3[k] * Cp_D)) * dtd == T3[k+1]

#             dCA3_dt[k=0:Nd-1], CA3[k] + (((F5[k] * fix.CA2[k+1] - fix.F7[k+1] * CA3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k])) * dtd == CA3[k+1]
#             dCB3_dt[k=0:Nd-1], CB3[k] + (((F5[k] * fix.CB2[k+1] + F6[k] * CB0 - fix.F7[k+1] * CB3[k]) / V3[k]) - r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CB3[k+1]
#             dCC3_dt[k=0:Nd-1], CC3[k] + ((F5[k] * fix.CC2[k+1] - fix.F7[k+1] * CC3[k]) / V3[k] + r1(T3[k], CA3[k], CB3[k]) - r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CC3[k+1]
#             dCD3_dt[k=0:Nd-1], CD3[k] + ((F5[k] * fix.CD2[k+1] - fix.F7[k+1] * CD3[k]) / V3[k] + r2(T3[k], CB3[k], CC3[k], CD3[k])) * dtd == CD3[k+1]
#             # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / 200




#             # volHoldUp3[k=0:Nd-1], F5[k] + F6[k] - fix.F7[k+1] == -(V3[k] - V3_sp) / 200
#             # volHoldUp31[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] <= (-(V3[k] - V3_sp) / 200) + s_path
#             # volHoldUp32[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - fix.F7[k+1] >= (-(V3[k] - V3_sp) / 200) - s_path
#         end


#         @NLobjective(
#             mpc,
#             Min,
#             1e-5 * sum(
#                 w.v * (V3[k] - V3_sp)^2 +
#                 w.t3 * (T3[k] - T3_sp)^2 +
#                 w.ca3 * (CA3[k] - CA3_sp)^2 +
#                 w.cb3 * (CB3[k] - CB3_sp)^2 +
#                 w.cc3 * (CC3[k] - CC3_sp)^2 +
#                 w.cd3 * (CD3[k] - CD3_sp)^2
#                 for k = 0:Nd) +
#             1e-5 * sum(
#                 w.f5 * (F5[k] - F5_sp)^2 +
#                 w.f6 * (F6[k] - F6_sp)^2 +
#                 w.q3 * (Q3[k] - Q3_sp)^2
#                 for k = 0:Nd
#             )
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
#         # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
#         # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
#         # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
#         # set_optimizer_attribute(mpc, "mu_init", 1e-3)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global dmpc2_solve_time = MOI.get(mpc, MOI.SolveTimeSec())


#         F5 = Vector(JuMP.value.(F5))
#         F6 = Vector(JuMP.value.(F6))
#         Q3 = Vector(JuMP.value.(Q3))

#         V3 = Vector(JuMP.value.(V3))

#         T3 = Vector(JuMP.value.(T3))

#         CA3 = Vector(JuMP.value.(CA3))

#         CB3 = Vector(JuMP.value.(CB3))

#         CC3 = Vector(JuMP.value.(CC3))

#         CD3 = Vector(JuMP.value.(CD3))

#         return F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3

#     end

#     function dmpc_3()

#         mpc = JuMP.Model(Ipopt.Optimizer)

#         # Register functions, supressing warnings
#         register(mpc, :H_A, 1, H_A; autodiff=true)
#         register(mpc, :H_B, 1, H_B; autodiff=true)
#         register(mpc, :H_C, 1, H_C; autodiff=true)
#         register(mpc, :H_D, 1, H_D; autodiff=true)
#         register(mpc, :r1, 3, r1; autodiff=true)
#         register(mpc, :r2, 4, r2; autodiff=true)
#         register(mpc, :r3, 3, r3; autodiff=true)
#         register(mpc, :CAr, 4, CAr; autodiff=true)
#         register(mpc, :CBr, 4, CBr; autodiff=true)
#         register(mpc, :CCr, 4, CCr; autodiff=true)
#         register(mpc, :CDr, 4, CDr; autodiff=true)
#         register(mpc, :MA, 13, MA; autodiff=true)
#         register(mpc, :MB, 13, MB; autodiff=true)
#         register(mpc, :MC, 13, MC; autodiff=true)
#         register(mpc, :MD, 13, MD; autodiff=true)

#         JuMP.@variables mpc begin
#             # Volume, state, [=] m3
#             # State variables
#             # Volume, state, [=] m3
#             V4[k=0:Nd], (lower_bound=0.2 * V4_sp, upper_bound=1.8 * V4_sp, start=V4_sp)
#             V5[k=0:Nd], (lower_bound=0.2 * V5_sp, upper_bound=1.8 * V5_sp, start=V5_sp)

#             # Temperature, state, [=] K
#             T4[k=0:Nd], (lower_bound=0.2 * T4_sp, upper_bound=1.8 * T4_sp, start=T4_sp)
#             T5[k=0:Nd], (lower_bound=0.2 * T5_sp, upper_bound=1.8 * T5_sp, start=T5_sp)

#             # Concentration, state, [=] mol/m3
#             CA4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA4_sp)
#             CA5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CA5_sp)

#             CB4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB4_sp)
#             CB5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CB5_sp)

#             CC4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC4_sp)
#             CC5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CC5_sp)

#             CD4[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD4_sp)
#             CD5[k=0:Nd], (lower_bound=1e-4, upper_bound=20000, start=CD5_sp)

#             # Control variables
#             # Flow, control [=] m3/s
#             F7[k=0:Nd], (lower_bound=lb_f * F7_sp, upper_bound=ub_f * F7_sp, start=F7_sp)
#             F8[k=0:Nd], (lower_bound=lb_f * F8_sp, upper_bound=ub_f * F8_sp, start=F8_sp)
#             F9[k=0:Nd], (lower_bound=lb_f * F9_sp, upper_bound=ub_f * F9_sp, start=F9_sp)
#             F10[k=0:Nd], (lower_bound=lb_f * F10_sp, upper_bound=ub_f * F10_sp, start=F10_sp)
#             Fr1[k=0:Nd], (lower_bound=lb_f * Fr1_sp, upper_bound=ub_f * Fr1_sp, start=Fr1_sp)
#             Fr2[k=0:Nd], (lower_bound=lb_f * Fr2_sp, upper_bound=ub_f * Fr2_sp, start=Fr2_sp)



#             Q4[k=0:Nd], (start = Q4_sp)
#             Q5[k=0:Nd], (start = Q5_sp)
#         end

#         for k = 0:Nd
#             JuMP.fix(F7[k], F7_sp; force=true)
#             # JuMP.fix(F8[k], F8_sp; force=true)
#             # JuMP.fix(F9[k], F9_sp; force=true)
#             # JuMP.fix(F10[k], F10_sp; force=true)
#             # JuMP.fix(Fr1[k], Fr1_sp; force=true)
#             JuMP.fix(Fr2[k], Fr2_sp; force=true)
#         end

#         for k = 0:Nd
#             # JuMP.fix(Q1[k], Q1_sp; force=true)
#             # JuMP.fix(Q2[k], Q2_sp; force=true)
#             # JuMP.fix(Q3[k], Q3_sp; force=true)
#             JuMP.fix(Q4[k], Q4_sp; force=true)
#             JuMP.fix(Q5[k], Q5_sp; force=true)
#         end

#         for k = 0:Nd
#             set_start_value(V4[k], fix.V4[k+1])
#             set_start_value(V5[k], fix.V5[k+1])

#             set_start_value(T4[k], fix.T4[k+1])
#             set_start_value(T5[k], fix.T5[k+1])

#             set_start_value(CA4[k], fix.CA4[k+1])
#             set_start_value(CA5[k], fix.CA5[k+1])

#             set_start_value(CB4[k], fix.CB4[k+1])
#             set_start_value(CB5[k], fix.CB5[k+1])

#             set_start_value(CC4[k], fix.CC4[k+1])
#             set_start_value(CC5[k], fix.CC5[k+1])

#             set_start_value(CD4[k], fix.CD4[k+1])
#             set_start_value(CD5[k], fix.CD5[k+1])

#             set_start_value(F7[k], fix.F7[k+1])
#             set_start_value(F8[k], fix.F8[k+1])
#             set_start_value(F9[k], fix.F9[k+1])
#             set_start_value(F10[k], fix.F10[k+1])
#             set_start_value(Fr1[k], fix.Fr1[k+1])
#             set_start_value(Fr2[k], fix.Fr2[k+1])

#         end

#         @constraints mpc begin

#             # Initial condition
#             V4_inital, V4[0] == V4_init
#             V5_inital, V5[0] == V5_init

#             T4_inital, T4[0] == T4_init
#             T5_inital, T5[0] == T5_init

#             CA4_initial, CA4[0] == CA4_init
#             CA5_initial, CA5[0] == CA5_init

#             CB4_initial, CB4[0] == CB4_init
#             CB5_initial, CB5[0] == CB5_init

#             CC4_initial, CC4[0] == CC4_init
#             CC5_initial, CC5[0] == CC5_init

#             CD4_initial, CD4[0] == CD4_init
#             CD5_initial, CD5[0] == CD5_init

#             # volDec4[k=0:N-1], (V4[k+1] - V4_sp) <= 0.8 * (V4[k] - V4_sp)
#             # volDec5[k=0:N-1], (V5[k+1] - V5_sp) <= 0.8 * (V5[k] - V5_sp)

#         end

#         @NLconstraints mpc begin
#             # NLconstraints are the differential equations that describe the dynamics of the system

#             dV4_dt[k=0:Nd-1], V4[k] + (F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]) * dtd == V4_sp
#             dT4_dt[k=0:Nd-1], T4[k] +
#                               ((Q4[k]
#                                 + (F7[k] * fix.CA3[k+1] * H_A(fix.T3[k+1]) + F9[k] * CA5[k] * H_A(T5[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_A(T4[k]) - F8[k] * CA4[k] * H_A(T4[k]) - MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_A)
#                                 + (F7[k] * fix.CB3[k+1] * H_B(fix.T3[k+1]) + F9[k] * CB5[k] * H_B(T5[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_B(T4[k]) - F8[k] * CB4[k] * H_B(T4[k]) - MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_B)
#                                 + (F7[k] * fix.CC3[k+1] * H_C(fix.T3[k+1]) + F9[k] * CC5[k] * H_C(T5[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_C(T4[k]) - F8[k] * CC4[k] * H_C(T4[k]) - MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_C)
#                                 + (F7[k] * fix.CD3[k+1] * H_D(fix.T3[k+1]) + F9[k] * CD5[k] * H_D(T5[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_D(T4[k]) - F8[k] * CD4[k] * H_D(T4[k]) - MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]) * H_vap_D))
#                                /
#                                (CA4[k] * Cp_A * V4[k] + CB4[k] * Cp_B * V4[k] + CC4[k] * Cp_C * V4[k] + CD4[k] * Cp_D * V4[k])) * dtd == T4[k+1]

#             dCA4_dt[k=0:Nd-1], CA4[k] + ((F7[k] * fix.CA3[k+1] + F9[k] * CA5[k] - (Fr1[k] + Fr2[k]) * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CA4[k]) / V4[k]) * dtd == CA4[k+1]
#             dCB4_dt[k=0:Nd-1], CB4[k] + ((F7[k] * fix.CB3[k+1] + F9[k] * CB5[k] - (Fr1[k] + Fr2[k]) * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CB4[k]) / V4[k]) * dtd == CB4[k+1]
#             dCC4_dt[k=0:Nd-1], CC4[k] + ((F7[k] * fix.CC3[k+1] + F9[k] * CC5[k] - (Fr1[k] + Fr2[k]) * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CC4[k]) / V4[k]) * dtd == CC4[k+1]
#             dCD4_dt[k=0:Nd-1], CD4[k] + ((F7[k] * fix.CD3[k+1] + F9[k] * CD5[k] - (Fr1[k] + Fr2[k]) * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F8[k] * CD4[k]) / V4[k]) * dtd == CD4[k+1]

#             dV5_dt[k=0:Nd-1], V5[k] + (F10[k] + Fr1[k] - F9[k]) * dtd == V5_sp

#             dT5_dt[k=0:Nd-1], T5[k] + (
#                 ((Q5[k] +
#                   F10[k] * CD0 * H_D(TD0)
#                   + (Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_A(T4[k]) - F9[k] * CA5[k] * H_A(T5[k]))
#                   + (Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_B(T4[k]) - F9[k] * CB5[k] * H_B(T5[k]))
#                   + (Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_C(T4[k]) - F9[k] * CC5[k] * H_C(T5[k]))
#                   + (Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) * H_D(T4[k]) - F9[k] * CD5[k] * H_D(T5[k])))
#                  /
#                  (CA5[k] * Cp_A * V5[k] + CB5[k] * Cp_B * V5[k] + CC5[k] * Cp_C * V5[k] + CD5[k] * Cp_D * V5[k]))
#                 +
#                 ((-delH_r2 * r2(T5[k], CB5[k], CC5[k], CD5[k]) - delH_r3 * r3(T5[k], CA5[k], CD5[k]))
#                  /
#                  (CA5[k] * Cp_A + CB5[k] * Cp_B + CC5[k] * Cp_C + CD5[k] * Cp_D))) * dtd == T5[k+1]

#             dCA5_dt[k=0:Nd-1], CA5[k] + ((Fr1[k] * CAr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CA5[k]) / V5[k] - r3(T5[k], CA5[k], CD5[k])) * dtd == CA5[k+1]
#             dCB5_dt[k=0:Nd-1], CB5[k] + ((Fr1[k] * CBr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CB5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k])) * dtd == CB5[k+1]
#             dCC5_dt[k=0:Nd-1], CC5[k] + ((Fr1[k] * CCr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) - F9[k] * CC5[k]) / V5[k] - r2(T5[k], CB5[k], CC5[k], CD5[k]) + 2 * r3(T5[k], CA5[k], CD5[k])) * dtd == CC5[k+1]
#             dCD5_dt[k=0:Nd-1], CD5[k] + ((Fr1[k] * CDr(MA(T4[k], F7[k], fix.CA3[k+1], F9[k], CA5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MB(T4[k], F7[k], fix.CB3[k+1], F9[k], CB5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MC(T4[k], F7[k], fix.CC3[k+1], F9[k], CC5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k]), MD(T4[k], F7[k], fix.CD3[k+1], F9[k], CD5[k], fix.CA3[k+1], fix.CB3[k+1], fix.CC3[k+1], fix.CD3[k+1], CA5[k], CB5[k], CC5[k], CD5[k])) + F10[k] * CD0 - F9[k] * CD5[k]) / V5[k] + r2(T5[k], CB5[k], CC5[k], CD5[k]) - r3(T5[k], CA5[k], CD5[k])) * dtd == CD5[k+1]
#             # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
#             # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200

#             # volHoldUp4[k=0:N-1], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / 200
#             # volHoldUp5[k=0:N-1], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / 200


#             # volHoldUp41[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] <= -(V4[k] - V4_sp) / 200 + s_path
#             # volHoldUp42[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] >= -(V4[k] - V4_sp) / 200 - s_path

#             # volHoldUp51[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] <= -(V5[k] - V5_sp) / 200 + s_path
#             # volHoldUp52[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] >= -(V5[k] - V5_sp) / 200 - s_path


#         end


#         @NLobjective(mpc, Min, sum(
#             1e-5 * w.v * (V4[k] - V4_sp)^2 + w.v * (V5[k] - V5_sp)^2 +
#             w.t4 * (T4[k] - T4_sp)^2 + w.t5 * (T5[k] - T5_sp)^2 +
#             w.ca4 * (CA4[k] - CA4_sp)^2 + w.ca5 * (CA5[k] - CA5_sp)^2 +
#             w.cb4 * (CB4[k] - CB4_sp)^2 + w.cb5 * (CB5[k] - CB5_sp)^2 +
#             w.cc4 * (CC4[k] - CC4_sp)^2 + w.cc5 * (CC5[k] - CC5_sp)^2 +
#             w.cd4 * (CD4[k] - CD4_sp)^2 + w.cd5 * (CD5[k] - CD5_sp)^2
#             for k = 0:Nd) +
#                                1e-5 * sum(
#             w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 +
#             w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 +
#             w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
#             w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2
#             for k = 0:Nd
#         )
#         )

#         set_optimizer_attribute(mpc, "max_iter", 100000)
#         set_optimizer_attribute(mpc, "tol", opt_tol) # Default is 1e-8
#         set_optimizer_attribute(mpc, "dual_inf_tol", dual_inf_tol) # Default is 1
#         set_optimizer_attribute(mpc, "constr_viol_tol", constr_viol_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "compl_inf_tol", compl_inf_tol) # Default is 1e-4
#         set_optimizer_attribute(mpc, "max_cpu_time", cpu_max)
#         # set_optimizer_attribute(mpc, "warm_start_init_point", "yes")
#         # set_optimizer_attribute(mpc, "bound_relax_factor", 1e-1)
#         # set_optimizer_attribute(mpc, "mu_strategy", "adaptive")
#         # set_optimizer_attribute(mpc, "mu_init", 1e-3)
#         # set_optimizer_attribute(mpc, "linear_solver", "ma57")

#         set_silent(mpc)

#         optimize!(mpc)

#         global dmpc3_solve_time = MOI.get(mpc, MOI.SolveTimeSec())
#         F7 = Vector(JuMP.value.(F7))
#         F8 = Vector(JuMP.value.(F8))
#         F9 = Vector(JuMP.value.(F9))
#         F10 = Vector(JuMP.value.(F10))
#         Fr1 = Vector(JuMP.value.(Fr1))
#         Fr2 = Vector(JuMP.value.(Fr2))

#         Q4 = Vector(JuMP.value.(Q4))
#         Q5 = Vector(JuMP.value.(Q5))

#         V4 = Vector(JuMP.value.(V4))
#         V5 = Vector(JuMP.value.(V5))

#         T4 = Vector(JuMP.value.(T4))
#         T5 = Vector(JuMP.value.(T5))

#         CA4 = Vector(JuMP.value.(CA4))
#         CA5 = Vector(JuMP.value.(CA5))

#         CB4 = Vector(JuMP.value.(CB4))
#         CB5 = Vector(JuMP.value.(CB5))

#         CC4 = Vector(JuMP.value.(CC4))
#         CC5 = Vector(JuMP.value.(CC5))

#         CD4 = Vector(JuMP.value.(CD4))
#         CD5 = Vector(JuMP.value.(CD5))

#         return F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5
#     end

#     function dmpc()

#         max_steps = 15
#         global dmpc_solve_time = 0
#         for steps = 1:max_steps

#             F1, F2, F3, F4, Q1, Q2, V1, V2, T1, T2, CA1, CA2, CB1, CB2, CC1, CC2, CD1, CD2 = dmpc_1()
#             F5, F6, Q3, V3, T3, CA3, CB3, CC3, CD3 = dmpc_2()
#             F7, F8, F9, F10, Fr1, Fr2, Q4, Q5, V4, V5, T4, T5, CA4, CA5, CB4, CB5, CC4, CC5, CD4, CD5 = dmpc_3()

#             fix.F1 = F1
#             fix.F2 = F2
#             fix.F3 = F3
#             fix.F4 = F4
#             fix.F5 = F5
#             fix.F6 = F6
#             fix.F7 = F7
#             fix.F8 = F8
#             fix.F9 = F9
#             fix.F10 = F10
#             fix.Fr1 = Fr1
#             fix.Fr2 = Fr2

#             fix.Q1 = Q1
#             fix.Q2 = Q2
#             fix.Q3 = Q3
#             fix.Q4 = Q4
#             fix.Q5 = Q5


#             global dmpc_solve_time = dmpc_solve_time + max(dmpc1_solve_time, dmpc2_solve_time, dmpc3_solve_time)

#             if dmpc_solve_time > c_cpu_max
#                 break
#             end


#             fix.T1 = T1
#             fix.T2 = T2
#             fix.T3 = T3
#             fix.T4 = T4
#             fix.T5 = T5

#             fix.V1 = V1
#             fix.V2 = V2
#             fix.V3 = V3
#             fix.V4 = V4
#             fix.V5 = V5

#             fix.CA1 = CA1
#             fix.CA2 = CA2
#             fix.CA3 = CA3
#             fix.CA4 = CA4
#             fix.CA5 = CA5
#             fix.CB1 = CB1
#             fix.CB2 = CB2
#             fix.CB3 = CB3
#             fix.CB4 = CB4
#             fix.CB5 = CB5
#             fix.CC1 = CC1
#             fix.CC2 = CC2
#             fix.CC3 = CC3
#             fix.CC4 = CC4
#             fix.CC5 = CC5
#             fix.CD1 = CD1
#             fix.CD2 = CD2
#             fix.CD3 = CD3
#             fix.CD4 = CD4
#             fix.CD5 = CD5
#         end

#         F1 = fix.F1
#         F2 = fix.F2
#         F3 = fix.F3
#         F4 = fix.F4
#         F5 = fix.F5
#         F6 = fix.F6
#         F7 = fix.F7
#         F8 = fix.F8
#         F9 = fix.F9
#         F10 = fix.F10
#         Fr1 = fix.Fr1
#         Fr2 = fix.Fr2
#         Q1 = fix.Q1
#         Q2 = fix.Q2
#         Q3 = fix.Q3
#         Q4 = fix.Q4
#         Q5 = fix.Q5

#         Q1 = ones(dtd) .* fix.Q1[1]
#         Q2 = ones(dtd) .* fix.Q2[1]
#         Q3 = ones(dtd) .* fix.Q3[1]
#         Q4 = ones(dtd) .* fix.Q4[1]
#         Q5 = ones(dtd) .* fix.Q5[1]

#         F1 = ones(dtd) .* fix.F1[1]
#         F2 = ones(dtd) .* fix.F2[1]
#         F3 = ones(dtd) .* fix.F3[1]
#         F4 = ones(dtd) .* fix.F4[1]
#         F5 = ones(dtd) .* fix.F5[1]
#         F6 = ones(dtd) .* fix.F6[1]
#         F7 = ones(dtd) .* fix.F7[1]
#         F8 = ones(dtd) .* fix.F8[1]
#         F9 = ones(dtd) .* fix.F9[1]
#         F10 = ones(dtd) .* fix.F10[1]
#         Fr1 = ones(dtd) .* fix.Fr1[1]
#         Fr2 = ones(dtd) .* fix.Fr2[1]

#         for i = 2:Nd
#             append!(F1, ones(dtd) .* fix.F1[i])
#             append!(F2, ones(dtd) .* fix.F2[i])
#             append!(F3, ones(dtd) .* fix.F3[i])
#             append!(F4, ones(dtd) .* fix.F4[i])
#             append!(F5, ones(dtd) .* fix.F5[i])
#             append!(F6, ones(dtd) .* fix.F6[i])
#             append!(F7, ones(dtd) .* fix.F7[i])
#             append!(F8, ones(dtd) .* fix.F8[i])
#             append!(F9, ones(dtd) .* fix.F9[i])
#             append!(F10, ones(dtd) .* fix.F10[i])
#             append!(Fr1, ones(dtd) .* fix.Fr1[i])
#             append!(Fr2, ones(dtd) .* fix.Fr2[i])

#             append!(Q1, ones(dtd) .* fix.Q1[i])
#             append!(Q2, ones(dtd) .* fix.Q2[i])
#             append!(Q3, ones(dtd) .* fix.Q3[i])
#             append!(Q4, ones(dtd) .* fix.Q4[i])
#             append!(Q5, ones(dtd) .* fix.Q5[i])
#         end


#         append!(F1, F1_sp)
#         append!(F2, F2_sp)
#         append!(F3, F3_sp)
#         append!(F4, F4_sp)
#         append!(F5, F5_sp)
#         append!(F6, F6_sp)
#         append!(F7, F7_sp)
#         append!(F8, F8_sp)
#         append!(F9, F9_sp)
#         append!(F10, F10_sp)
#         append!(Fr1, Fr1_sp)
#         append!(Fr2, Fr2_sp)

#         append!(Q1, Q1_sp)
#         append!(Q2, Q2_sp)
#         append!(Q3, Q3_sp)
#         append!(Q4, Q4_sp)
#         append!(Q5, Q5_sp)


#         return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5


#     end

#     function getTraj(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5)

#         V1_vec = zeros(N + 1)
#         V2_vec = zeros(N + 1)
#         V3_vec = zeros(N + 1)
#         V4_vec = zeros(N + 1)
#         V5_vec = zeros(N + 1)
#         T1_vec = zeros(N + 1)
#         T2_vec = zeros(N + 1)
#         T3_vec = zeros(N + 1)
#         T4_vec = zeros(N + 1)
#         T5_vec = zeros(N + 1)

#         CA1_vec = zeros(N + 1)
#         CA2_vec = zeros(N + 1)
#         CA3_vec = zeros(N + 1)
#         CA4_vec = zeros(N + 1)
#         CA5_vec = zeros(N + 1)
#         CB1_vec = zeros(N + 1)
#         CB2_vec = zeros(N + 1)
#         CB3_vec = zeros(N + 1)
#         CB4_vec = zeros(N + 1)
#         CB5_vec = zeros(N + 1)
#         CC1_vec = zeros(N + 1)
#         CC2_vec = zeros(N + 1)
#         CC3_vec = zeros(N + 1)
#         CC4_vec = zeros(N + 1)
#         CC5_vec = zeros(N + 1)
#         CD1_vec = zeros(N + 1)
#         CD2_vec = zeros(N + 1)
#         CD3_vec = zeros(N + 1)
#         CD4_vec = zeros(N + 1)
#         CD5_vec = zeros(N + 1)



#         V1_vec[1] = V1_init
#         V2_vec[1] = V2_init
#         V3_vec[1] = V3_init
#         V4_vec[1] = V4_init
#         V5_vec[1] = V5_init
#         T1_vec[1] = T1_init
#         T2_vec[1] = T2_init
#         T3_vec[1] = T3_init
#         T4_vec[1] = T4_init
#         T5_vec[1] = T5_init

#         CA1_vec[1] = CA1_init
#         CA2_vec[1] = CA2_init
#         CA3_vec[1] = CA3_init
#         CA4_vec[1] = CA4_init
#         CA5_vec[1] = CA5_init
#         CB1_vec[1] = CB1_init
#         CB2_vec[1] = CB2_init
#         CB3_vec[1] = CB3_init
#         CB4_vec[1] = CB4_init
#         CB5_vec[1] = CB5_init
#         CC1_vec[1] = CC1_init
#         CC2_vec[1] = CC2_init
#         CC3_vec[1] = CC3_init
#         CC4_vec[1] = CC4_init
#         CC5_vec[1] = CC5_init
#         CD1_vec[1] = CD1_init
#         CD2_vec[1] = CD2_init
#         CD3_vec[1] = CD3_init
#         CD4_vec[1] = CD4_init
#         CD5_vec[1] = CD5_init

#         for j = 1:N
#             global k = j

#             V1 = V1_vec[k]
#             V2 = V2_vec[k]
#             V3 = V3_vec[k]
#             V4 = V4_vec[k]
#             V5 = V5_vec[k]

#             T1 = T1_vec[k]
#             T2 = T2_vec[k]
#             T3 = T3_vec[k]
#             T4 = T4_vec[k]
#             T5 = T5_vec[k]

#             CA1 = CA1_vec[k]
#             CB1 = CB1_vec[k]
#             CC1 = CC1_vec[k]
#             CD1 = CD1_vec[k]

#             CA2 = CA2_vec[k]
#             CB2 = CB2_vec[k]
#             CC2 = CC2_vec[k]
#             CD2 = CD2_vec[k]

#             CA3 = CA3_vec[k]
#             CB3 = CB3_vec[k]
#             CC3 = CC3_vec[k]
#             CD3 = CD3_vec[k]

#             CA4 = CA4_vec[k]
#             CB4 = CB4_vec[k]
#             CC4 = CC4_vec[k]
#             CD4 = CD4_vec[k]

#             CA5 = CA5_vec[k]
#             CB5 = CB5_vec[k]
#             CC5 = CC5_vec[k]
#             CD5 = CD5_vec[k]

#             x0 = [V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1[k] + F2[k] + Fr2[k] - F3[k]
#                 dV2 = F3[k] + F4[k] - F5[k]
#                 dV3 = F5[k] + F6[k] - F7[k]
#                 dV4 = F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]
#                 dV5 = F10[k] + Fr1[k] - F9[k]

#                 dT1 = (
#                     ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                       (Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3[k] * CA1 * H_A(T1)) +
#                       (Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3[k] * CB1 * H_B(T1)) +
#                       (Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3[k] * CC1 * H_C(T1)) +
#                       (Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3[k] * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1_(T1, CA1, CB1) - delH_r2 * r2_(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                      (F3[k] * CA1 * H_A(T1) - F5[k] * CA2 * H_A(T2)) +
#                      (F3[k] * CB1 * H_B(T1) - F5[k] * CB2 * H_B(T2)) +
#                      (F3[k] * CC1 * H_C(T1) - F5[k] * CC2 * H_C(T2)) +
#                      (F3[k] * CD1 * H_D(T1) - F5[k] * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1_(T2, CA2, CB2) - delH_r2 * r2_(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


#                 dT3 = (
#                     ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2 * H_A(T2) - F7[k] * CA3 * H_A(T3)) +
#                       (F5[k] * CB2 * H_B(T2) - F7[k] * CB3 * H_B(T3)) +
#                       (F5[k] * CC2 * H_C(T2) - F7[k] * CC3 * H_C(T3)) +
#                       (F5[k] * CD2 * H_D(T2) - F7[k] * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1_(T3, CA3, CB3) - delH_r2 * r2_(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4[k]
#                         + (F7[k] * CA3 * H_A(T3) + F9[k] * CA5 * H_A(T5) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8[k] * CA4 * H_A(T4) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7[k] * CB3 * H_B(T3) + F9[k] * CB5 * H_B(T5) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8[k] * CB4 * H_B(T4) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7[k] * CC3 * H_C(T3) + F9[k] * CC5 * H_C(T5) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8[k] * CC4 * H_C(T4) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7[k] * CD3 * H_D(T3) + F9[k] * CD5 * H_D(T5) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8[k] * CD4 * H_D(T4) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5[k] +
#                       F10[k] * CD0 * H_D(TD0)
#                       + (Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9[k] * CA5 * H_A(T5))
#                       + (Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9[k] * CB5 * H_B(T5))
#                       + (Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9[k] * CC5 * H_C(T5))
#                       + (Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9[k] * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2_(T5, CB5, CC5, CD5) - delH_r3 * r3_(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CA1)
#                      /
#                      V1) - r1_(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CB1) /
#                      V1) - r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CC1) /
#                      V1) + r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CD1) /
#                      V1) + r2_(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3[k] * CA1 - F5[k] * CA2) / V2) - r1_(T2, CA2, CB2))
#                 dCB2 = ((F3[k] * CB1 + F4[k] * CB0 - F5[k] * CB2) / V2 - r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3[k] * CC1 - F5[k] * CC2) / V2 + r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3[k] * CD1 - F5[k] * CD2) / V2 + r2_(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5[k] * CA2 - F7[k] * CA3) / V3) - r1_(T3, CA3, CB3))
#                 dCB3 = (((F5[k] * CB2 + F6[k] * CB0 - F7[k] * CB3) / V3) - r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5[k] * CC2 - F7[k] * CC3) / V3 + r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5[k] * CD2 - F7[k] * CD3) / V3 + r2_(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7[k] * CA3 + F9[k] * CA5 - (Fr1[k] + Fr2[k]) * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CA4) / V4)
#                 dCB4 = ((F7[k] * CB3 + F9[k] * CB5 - (Fr1[k] + Fr2[k]) * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CB4) / V4)
#                 dCC4 = ((F7[k] * CC3 + F9[k] * CC5 - (Fr1[k] + Fr2[k]) * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CC4) / V4)
#                 dCD4 = ((F7[k] * CD3 + F9[k] * CD5 - (Fr1[k] + Fr2[k]) * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CD4) / V4)

#                 dCA5 = ((Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CA5) / V5 - r3_(T5, CA5, CD5))
#                 dCB5 = ((Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CB5) / V5 - r2_(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CC5) / V5 - r2_(T5, CB5, CC5, CD5) + 2 * r3_(T5, CA5, CD5))
#                 dCD5 = ((Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10[k] * CD0 - F9[k] * CD5) / V5 + r2_(T5, CB5, CC5, CD5) - r3_(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dt)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23())

#             # Obtain the next values for each element
#             V1_vec[k+1] = max(1e-6, soln.u[end][1])
#             V2_vec[k+1] = max(1e-6, soln.u[end][2])
#             V3_vec[k+1] = max(1e-6, soln.u[end][3])
#             V4_vec[k+1] = max(1e-6, soln.u[end][4])
#             V5_vec[k+1] = max(1e-6, soln.u[end][5])

#             T1_vec[k+1] = max(1e-6, soln.u[end][6])
#             T2_vec[k+1] = max(1e-6, soln.u[end][7])
#             T3_vec[k+1] = max(1e-6, soln.u[end][8])
#             T4_vec[k+1] = max(1e-6, soln.u[end][9])
#             T5_vec[k+1] = max(1e-6, soln.u[end][10])

#             CA1_vec[k+1] = max(1e-6, soln.u[end][11])
#             CB1_vec[k+1] = max(1e-6, soln.u[end][12])
#             CC1_vec[k+1] = max(1e-6, soln.u[end][13])
#             CD1_vec[k+1] = max(1e-6, soln.u[end][14])

#             CA2_vec[k+1] = max(1e-6, soln.u[end][15])
#             CB2_vec[k+1] = max(1e-6, soln.u[end][16])
#             CC2_vec[k+1] = max(1e-6, soln.u[end][17])
#             CD2_vec[k+1] = max(1e-6, soln.u[end][18])

#             CA3_vec[k+1] = max(1e-6, soln.u[end][19])
#             CB3_vec[k+1] = max(1e-6, soln.u[end][20])
#             CC3_vec[k+1] = max(1e-6, soln.u[end][21])
#             CD3_vec[k+1] = max(1e-6, soln.u[end][22])

#             CA4_vec[k+1] = max(1e-6, soln.u[end][23])
#             CB4_vec[k+1] = max(1e-6, soln.u[end][24])
#             CC4_vec[k+1] = max(1e-6, soln.u[end][25])
#             CD4_vec[k+1] = max(1e-6, soln.u[end][26])

#             CA5_vec[k+1] = max(1e-6, soln.u[end][27])
#             CB5_vec[k+1] = max(1e-6, soln.u[end][28])
#             CC5_vec[k+1] = max(1e-6, soln.u[end][29])
#             CD5_vec[k+1] = max(1e-6, soln.u[end][30])

#         end

#         V1 = V1_vec
#         V2 = V2_vec
#         V3 = V3_vec
#         V4 = V4_vec
#         V5 = V5_vec

#         T1 = T1_vec
#         T2 = T2_vec
#         T3 = T3_vec
#         T4 = T4_vec
#         T5 = T5_vec

#         CA1 = CA1_vec
#         CB1 = CB1_vec
#         CC1 = CC1_vec
#         CD1 = CD1_vec

#         CA2 = CA2_vec
#         CB2 = CB2_vec
#         CC2 = CC2_vec
#         CD2 = CD2_vec

#         CA3 = CA3_vec
#         CB3 = CB3_vec
#         CC3 = CC3_vec
#         CD3 = CD3_vec

#         CA4 = CA4_vec
#         CB4 = CB4_vec
#         CC4 = CC4_vec
#         CD4 = CD4_vec

#         CA5 = CA5_vec
#         CB5 = CB5_vec
#         CC5 = CC5_vec
#         CD5 = CD5_vec

#         return V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5
#     end

#     function getPI(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fr1, Fr2, Q1, Q2, Q3, Q4, Q5)

#         V1_vec = zeros(N + 1)
#         V2_vec = zeros(N + 1)
#         V3_vec = zeros(N + 1)
#         V4_vec = zeros(N + 1)
#         V5_vec = zeros(N + 1)
#         T1_vec = zeros(N + 1)
#         T2_vec = zeros(N + 1)
#         T3_vec = zeros(N + 1)
#         T4_vec = zeros(N + 1)
#         T5_vec = zeros(N + 1)

#         CA1_vec = zeros(N + 1)
#         CA2_vec = zeros(N + 1)
#         CA3_vec = zeros(N + 1)
#         CA4_vec = zeros(N + 1)
#         CA5_vec = zeros(N + 1)
#         CB1_vec = zeros(N + 1)
#         CB2_vec = zeros(N + 1)
#         CB3_vec = zeros(N + 1)
#         CB4_vec = zeros(N + 1)
#         CB5_vec = zeros(N + 1)
#         CC1_vec = zeros(N + 1)
#         CC2_vec = zeros(N + 1)
#         CC3_vec = zeros(N + 1)
#         CC4_vec = zeros(N + 1)
#         CC5_vec = zeros(N + 1)
#         CD1_vec = zeros(N + 1)
#         CD2_vec = zeros(N + 1)
#         CD3_vec = zeros(N + 1)
#         CD4_vec = zeros(N + 1)
#         CD5_vec = zeros(N + 1)

#         V1_vec[1] = V1_init
#         V2_vec[1] = V2_init
#         V3_vec[1] = V3_init
#         V4_vec[1] = V4_init
#         V5_vec[1] = V5_init
#         T1_vec[1] = T1_init
#         T2_vec[1] = T2_init
#         T3_vec[1] = T3_init
#         T4_vec[1] = T4_init
#         T5_vec[1] = T5_init

#         CA1_vec[1] = CA1_init
#         CA2_vec[1] = CA2_init
#         CA3_vec[1] = CA3_init
#         CA4_vec[1] = CA4_init
#         CA5_vec[1] = CA5_init
#         CB1_vec[1] = CB1_init
#         CB2_vec[1] = CB2_init
#         CB3_vec[1] = CB3_init
#         CB4_vec[1] = CB4_init
#         CB5_vec[1] = CB5_init
#         CC1_vec[1] = CC1_init
#         CC2_vec[1] = CC2_init
#         CC3_vec[1] = CC3_init
#         CC4_vec[1] = CC4_init
#         CC5_vec[1] = CC5_init
#         CD1_vec[1] = CD1_init
#         CD2_vec[1] = CD2_init
#         CD3_vec[1] = CD3_init
#         CD4_vec[1] = CD4_init
#         CD5_vec[1] = CD5_init

#         for j = 1:N
#             global k = j

#             V1 = V1_vec[k]
#             V2 = V2_vec[k]
#             V3 = V3_vec[k]
#             V4 = V4_vec[k]
#             V5 = V5_vec[k]

#             T1 = T1_vec[k]
#             T2 = T2_vec[k]
#             T3 = T3_vec[k]
#             T4 = T4_vec[k]
#             T5 = T5_vec[k]

#             CA1 = CA1_vec[k]
#             CB1 = CB1_vec[k]
#             CC1 = CC1_vec[k]
#             CD1 = CD1_vec[k]

#             CA2 = CA2_vec[k]
#             CB2 = CB2_vec[k]
#             CC2 = CC2_vec[k]
#             CD2 = CD2_vec[k]

#             CA3 = CA3_vec[k]
#             CB3 = CB3_vec[k]
#             CC3 = CC3_vec[k]
#             CD3 = CD3_vec[k]

#             CA4 = CA4_vec[k]
#             CB4 = CB4_vec[k]
#             CC4 = CC4_vec[k]
#             CD4 = CD4_vec[k]

#             CA5 = CA5_vec[k]
#             CB5 = CB5_vec[k]
#             CC5 = CC5_vec[k]
#             CD5 = CD5_vec[k]

#             x0 = [V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5]

#             # Define the ODE function
#             function f(y, p, t)
#                 # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                 dV1 = F1[k] + F2[k] + Fr2[k] - F3[k]
#                 dV2 = F3[k] + F4[k] - F5[k]
#                 dV3 = F5[k] + F6[k] - F7[k]
#                 dV4 = F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k]
#                 dV5 = F10[k] + Fr1[k] - F9[k]

#                 dT1 = (
#                     ((Q1[k] + F1[k] * CA0 * H_A(TA0) + F2[k] * CB0 * H_B(TB0) +
#                       (Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3[k] * CA1 * H_A(T1)) +
#                       (Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3[k] * CB1 * H_B(T1)) +
#                       (Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3[k] * CC1 * H_C(T1)) +
#                       (Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3[k] * CD1 * H_D(T1)))
#                      /
#                      (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                     (-delH_r1 * r1_(T1, CA1, CB1) - delH_r2 * r2_(T1, CB1, CC1, CD1)) /
#                     (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                 dT2 = (
#                     (Q2[k] + F4[k] * CB0 * H_B(TB0) +
#                      (F3[k] * CA1 * H_A(T1) - F5[k] * CA2 * H_A(T2)) +
#                      (F3[k] * CB1 * H_B(T1) - F5[k] * CB2 * H_B(T2)) +
#                      (F3[k] * CC1 * H_C(T1) - F5[k] * CC2 * H_C(T2)) +
#                      (F3[k] * CD1 * H_D(T1) - F5[k] * CD2 * H_D(T2)))
#                     /
#                     (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                     +
#                     (-delH_r1 * r1_(T2, CA2, CB2) - delH_r2 * r2_(T2, CB2, CC2, CD2))
#                     /
#                     (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


#                 dT3 = (
#                     ((Q3[k] + F6[k] * CB0 * H_B(TB0) + (F5[k] * CA2 * H_A(T2) - F7[k] * CA3 * H_A(T3)) +
#                       (F5[k] * CB2 * H_B(T2) - F7[k] * CB3 * H_B(T3)) +
#                       (F5[k] * CC2 * H_C(T2) - F7[k] * CC3 * H_C(T3)) +
#                       (F5[k] * CD2 * H_D(T2) - F7[k] * CD3 * H_D(T3)))
#                      /
#                      (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                     +
#                     (-delH_r1 * r1_(T3, CA3, CB3) - delH_r2 * r2_(T3, CB3, CC3, CD3))
#                     /
#                     (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                 dT4 = ((Q4[k]
#                         + (F7[k] * CA3 * H_A(T3) + F9[k] * CA5 * H_A(T5) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8[k] * CA4 * H_A(T4) - MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                         + (F7[k] * CB3 * H_B(T3) + F9[k] * CB5 * H_B(T5) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8[k] * CB4 * H_B(T4) - MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                         + (F7[k] * CC3 * H_C(T3) + F9[k] * CC5 * H_C(T5) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8[k] * CC4 * H_C(T4) - MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                         + (F7[k] * CD3 * H_D(T3) + F9[k] * CD5 * H_D(T5) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8[k] * CD4 * H_D(T4) - MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                        /
#                        (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                 dT5 = (
#                     ((Q5[k] +
#                       F10[k] * CD0 * H_D(TD0)
#                       + (Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9[k] * CA5 * H_A(T5))
#                       + (Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9[k] * CB5 * H_B(T5))
#                       + (Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9[k] * CC5 * H_C(T5))
#                       + (Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9[k] * CD5 * H_D(T5)))
#                      /
#                      (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                     +
#                     ((-delH_r2 * r2_(T5, CB5, CC5, CD5) - delH_r3 * r3_(T5, CA5, CD5))
#                      /
#                      (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                 dCA1 = (
#                     ((F1[k] * CA0 + Fr2[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CA1)
#                      /
#                      V1) - r1_(T1, CA1, CB1))

#                 dCB1 = (
#                     ((F2[k] * CB0 + Fr2[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CB1) /
#                      V1) - r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCC1 = (
#                     ((Fr2[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CC1) /
#                      V1) + r1_(T1, CA1, CB1) - r2_(T1, CB1, CC1, CD1))

#                 dCD1 = (
#                     ((Fr2[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                       -
#                       F3[k] * CD1) /
#                      V1) + r2_(T1, CB1, CC1, CD1))

#                 dCA2 = (((F3[k] * CA1 - F5[k] * CA2) / V2) - r1_(T2, CA2, CB2))
#                 dCB2 = ((F3[k] * CB1 + F4[k] * CB0 - F5[k] * CB2) / V2 - r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCC2 = ((F3[k] * CC1 - F5[k] * CC2) / V2 + r1_(T2, CA2, CB2) - r2_(T2, CB2, CC2, CD2))
#                 dCD2 = ((F3[k] * CD1 - F5[k] * CD2) / V2 + r2_(T2, CB2, CC2, CD2))

#                 dCA3 = (((F5[k] * CA2 - F7[k] * CA3) / V3) - r1_(T3, CA3, CB3))
#                 dCB3 = (((F5[k] * CB2 + F6[k] * CB0 - F7[k] * CB3) / V3) - r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCC3 = ((F5[k] * CC2 - F7[k] * CC3) / V3 + r1_(T3, CA3, CB3) - r2_(T3, CB3, CC3, CD3))
#                 dCD3 = ((F5[k] * CD2 - F7[k] * CD3) / V3 + r2_(T3, CB3, CC3, CD3))

#                 dCA4 = ((F7[k] * CA3 + F9[k] * CA5 - (Fr1[k] + Fr2[k]) * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CA4) / V4)
#                 dCB4 = ((F7[k] * CB3 + F9[k] * CB5 - (Fr1[k] + Fr2[k]) * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CB4) / V4)
#                 dCC4 = ((F7[k] * CC3 + F9[k] * CC5 - (Fr1[k] + Fr2[k]) * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CC4) / V4)
#                 dCD4 = ((F7[k] * CD3 + F9[k] * CD5 - (Fr1[k] + Fr2[k]) * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8[k] * CD4) / V4)

#                 dCA5 = ((Fr1[k] * CAr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CA5) / V5 - r3_(T5, CA5, CD5))
#                 dCB5 = ((Fr1[k] * CBr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CB5) / V5 - r2_(T5, CB5, CC5, CD5))
#                 dCC5 = ((Fr1[k] * CCr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9[k] * CC5) / V5 - r2_(T5, CB5, CC5, CD5) + 2 * r3_(T5, CA5, CD5))
#                 dCD5 = ((Fr1[k] * CDr(MA(T4, F7[k], CA3, F9[k], CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7[k], CB3, F9[k], CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7[k], CC3, F9[k], CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7[k], CD3, F9[k], CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10[k] * CD0 - F9[k] * CD5) / V5 + r2_(T5, CB5, CC5, CD5) - r3_(T5, CA5, CD5))

#                 return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#             end

#             # Solve ODE 
#             tspan = (0.0, dt)
#             prob = ODEProblem(f, x0, tspan)
#             soln = solve(prob, Rosenbrock23(), alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, save_everystep=false)

#             # Obtain the next values for each element
#             V1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][1]))
#             V2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][2]))
#             V3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][3]))
#             V4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][4]))
#             V5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][5]))

#             T1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][6]))
#             T2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][7]))
#             T3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][8]))
#             T4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][9]))
#             T5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][10]))

#             CA1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][11]))
#             CB1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][12]))
#             CC1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][13]))
#             CD1_vec[k+1] = min(1e7, max(1e-6, soln.u[end][14]))

#             CA2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][15]))
#             CB2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][16]))
#             CC2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][17]))
#             CD2_vec[k+1] = min(1e7, max(1e-6, soln.u[end][18]))

#             CA3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][19]))
#             CB3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][20]))
#             CC3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][21]))
#             CD3_vec[k+1] = min(1e7, max(1e-6, soln.u[end][22]))

#             CA4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][23]))
#             CB4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][24]))
#             CC4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][25]))
#             CD4_vec[k+1] = min(1e7, max(1e-6, soln.u[end][26]))

#             CA5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][27]))
#             CB5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][28]))
#             CC5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][29]))
#             CD5_vec[k+1] = min(1e7, max(1e-6, soln.u[end][30]))


#         end

#         V1 = V1_vec
#         V2 = V2_vec
#         V3 = V3_vec
#         V4 = V4_vec
#         V5 = V5_vec

#         T1 = T1_vec
#         T2 = T2_vec
#         T3 = T3_vec
#         T4 = T4_vec
#         T5 = T5_vec

#         CA1 = CA1_vec
#         CB1 = CB1_vec
#         CC1 = CC1_vec
#         CD1 = CD1_vec

#         CA2 = CA2_vec
#         CB2 = CB2_vec
#         CC2 = CC2_vec
#         CD2 = CD2_vec

#         CA3 = CA3_vec
#         CB3 = CB3_vec
#         CC3 = CC3_vec
#         CD3 = CD3_vec

#         CA4 = CA4_vec
#         CB4 = CB4_vec
#         CC4 = CC4_vec
#         CD4 = CD4_vec

#         CA5 = CA5_vec
#         CB5 = CB5_vec
#         CC5 = CC5_vec
#         CD5 = CD5_vec

#         ISE = sum(w.v * (V1_vec[k] - V1_sp)^2 + w.v * (V2_vec[k] - V2_sp)^2 + w.v * (V3_vec[k] - V3_sp)^2 + w.v * (V4_vec[k] - V4_sp)^2 + w.v * (V5_vec[k] - V5_sp)^2 +
#                   w.t1 * (T1_vec[k] - T1_sp)^2 + w.t2 * (T2_vec[k] - T2_sp)^2 + w.t3 * (T3_vec[k] - T3_sp)^2 + w.t4 * (T4_vec[k] - T4_sp)^2 + w.t5 * (T5_vec[k] - T5_sp)^2 +
#                   w.ca1 * (CA1_vec[k] - CA1_sp)^2 + w.ca2 * (CA2_vec[k] - CA2_sp)^2 + w.ca3 * (CA3_vec[k] - CA3_sp)^2 + w.ca4 * (CA4_vec[k] - CA4_sp)^2 + w.ca5 * (CA5_vec[k] - CA5_sp)^2 +
#                   w.cb1 * (CB1_vec[k] - CB1_sp)^2 + w.cb2 * (CB2_vec[k] - CB2_sp)^2 + w.cb3 * (CB3_vec[k] - CB3_sp)^2 + w.cb4 * (CB4_vec[k] - CB4_sp)^2 + w.cb5 * (CB5_vec[k] - CB5_sp)^2 +
#                   w.cc1 * (CC1_vec[k] - CC1_sp)^2 + w.cc2 * (CC2_vec[k] - CC2_sp)^2 + w.cc3 * (CC3_vec[k] - CC3_sp)^2 + w.cc4 * (CC4_vec[k] - CC4_sp)^2 + w.cc5 * (CC5_vec[k] - CC5_sp)^2 +
#                   w.cd1 * (CD1_vec[k] - CD1_sp)^2 + w.cd2 * (CD2_vec[k] - CD2_sp)^2 + w.cd3 * (CD3_vec[k] - CD3_sp)^2 + w.cd4 * (CD4_vec[k] - CD4_sp)^2 + w.cd5 * (CD5_vec[k] - CD5_sp)^2 for k = 1:N+1)



#         ISC = sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 + w.f5 * (F5[k] - F5_sp)^2 +
#                   w.f6 * (F6[k] - F6_sp)^2 + w.f7 * (F7[k] - F7_sp)^2 + w.f8 * (F8[k] - F8_sp)^2 + w.f9 * (F9[k] - F9_sp)^2 + w.f10 * (F10[k] - F10_sp)^2 + w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
#                   w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2 + w.q5 * (Q5[k] - Q5_sp)^2 for k = 1:N+1)

#         PI = ISE + ISC

#         if isnan(PI)
#             PI = 1e12
#         end

#         return ISE, ISC, PI
#     end

#     function takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

#         # We are using a 30 second sampling time for the system, just with a fine discretization. Therefore, the next timestep is at k = 61

#         global V1_init = V1[dtd+1]
#         global V2_init = V2[dtd+1]
#         global V3_init = V3[dtd+1]
#         global V4_init = V4[dtd+1]
#         global V5_init = V5[dtd+1]

#         global T1_init = T1[dtd+1]
#         global T2_init = T2[dtd+1]
#         global T3_init = T3[dtd+1]
#         global T4_init = T4[dtd+1]
#         global T5_init = T5[dtd+1]

#         global CA1_init = CA1[dtd+1]
#         global CA2_init = CA2[dtd+1]
#         global CA3_init = CA3[dtd+1]
#         global CA4_init = CA4[dtd+1]
#         global CA5_init = CA5[dtd+1]

#         global CB1_init = CB1[dtd+1]
#         global CB2_init = CB2[dtd+1]
#         global CB3_init = CB3[dtd+1]
#         global CB4_init = CB4[dtd+1]
#         global CB5_init = CB5[dtd+1]

#         global CC1_init = CC1[dtd+1]
#         global CC2_init = CC2[dtd+1]
#         global CC3_init = CC3[dtd+1]
#         global CC4_init = CC4[dtd+1]
#         global CC5_init = CC5[dtd+1]

#         global CD1_init = CD1[dtd+1]
#         global CD2_init = CD2[dtd+1]
#         global CD3_init = CD3[dtd+1]
#         global CD4_init = CD4[dtd+1]
#         global CD5_init = CD5[dtd+1]
#     end

#     predict = Decomposition_Trajectory(zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P), zeros(P))

#     centr_PI_vec = zeros(P)
#     centr_ISE_vec = zeros(P)
#     centr_ISC_vec = zeros(P)
#     centr_time = zeros(P)

#     distr_PI_vec = zeros(P)
#     distr_ISE_vec = zeros(P)
#     distr_ISC_vec = zeros(P)
#     distr_time = zeros(P)

#     function mhmpc_sim()

#         for mpc_step = 1:P

#             println("MPC Step: $(mpc_step)")
#             println()
#             predict.V1[mpc_step] = V1_init
#             predict.V2[mpc_step] = V2_init
#             predict.V3[mpc_step] = V3_init
#             predict.V4[mpc_step] = V4_init
#             predict.V5[mpc_step] = V5_init

#             predict.T1[mpc_step] = T1_init
#             predict.T2[mpc_step] = T2_init
#             predict.T3[mpc_step] = T3_init
#             predict.T4[mpc_step] = T4_init
#             predict.T5[mpc_step] = T5_init

#             predict.CA1[mpc_step] = CA1_init
#             predict.CA2[mpc_step] = CA2_init
#             predict.CA3[mpc_step] = CA3_init
#             predict.CA4[mpc_step] = CA4_init
#             predict.CA5[mpc_step] = CA5_init

#             predict.CB1[mpc_step] = CB1_init
#             predict.CB2[mpc_step] = CB2_init
#             predict.CB3[mpc_step] = CB3_init
#             predict.CB4[mpc_step] = CB4_init
#             predict.CB5[mpc_step] = CB5_init

#             predict.CC1[mpc_step] = CC1_init
#             predict.CC2[mpc_step] = CC2_init
#             predict.CC3[mpc_step] = CC3_init
#             predict.CC4[mpc_step] = CC4_init
#             predict.CC5[mpc_step] = CC5_init

#             predict.CD1[mpc_step] = CD1_init
#             predict.CD2[mpc_step] = CD2_init
#             predict.CD3[mpc_step] = CD3_init
#             predict.CD4[mpc_step] = CD4_init
#             predict.CD5[mpc_step] = CD5_init

#             ex.T1[mpc_step] = T1_init
#             ex.T2[mpc_step] = T2_init
#             ex.T3[mpc_step] = T3_init
#             ex.T4[mpc_step] = T4_init
#             ex.T5[mpc_step] = T5_init

#             ex.V1[mpc_step] = V1_init
#             ex.V2[mpc_step] = V2_init
#             ex.V3[mpc_step] = V3_init
#             ex.V4[mpc_step] = V4_init
#             ex.V5[mpc_step] = V5_init

#             ex.CA1[mpc_step] = CA1_init
#             ex.CA2[mpc_step] = CA2_init
#             ex.CA3[mpc_step] = CA3_init
#             ex.CA4[mpc_step] = CA4_init
#             ex.CA5[mpc_step] = CA5_init

#             ex.CB1[mpc_step] = CB1_init
#             ex.CB2[mpc_step] = CB2_init
#             ex.CB3[mpc_step] = CB3_init
#             ex.CB4[mpc_step] = CB4_init
#             ex.CB5[mpc_step] = CB5_init

#             ex.CC1[mpc_step] = CC1_init
#             ex.CC2[mpc_step] = CC2_init
#             ex.CC3[mpc_step] = CC3_init
#             ex.CC4[mpc_step] = CC4_init
#             ex.CC5[mpc_step] = CC5_init

#             ex.CD1[mpc_step] = CD1_init
#             ex.CD2[mpc_step] = CD2_init
#             ex.CD3[mpc_step] = CD3_init
#             ex.CD4[mpc_step] = CD4_init
#             ex.CD5[mpc_step] = CD5_init

#             # Initialize fix structure for decomp
#             for set_traj = 1
#                 # Set control variables to nominal steady state values
#                 fix.F1 .= F1_sp
#                 fix.F2 .= F2_sp
#                 fix.F3 .= F3_sp
#                 fix.F4 .= F4_sp
#                 fix.F5 .= F5_sp
#                 fix.F6 .= F6_sp
#                 fix.F7 .= F7_sp
#                 fix.F8 .= F8_sp
#                 fix.F9 .= F9_sp
#                 fix.F10 .= F10_sp
#                 fix.Fr1 .= Fr1_sp
#                 fix.Fr2 .= Fr2_sp

#                 fix.Q1 .= Q1_sp
#                 fix.Q2 .= Q2_sp
#                 fix.Q3 .= Q3_sp
#                 fix.Q4 .= Q4_sp
#                 fix.Q5 .= Q5_sp

#                 fix.V1[1] = V1_init
#                 fix.V2[1] = V2_init
#                 fix.V3[1] = V3_init
#                 fix.V4[1] = V4_init
#                 fix.V5[1] = V5_init

#                 fix.T1[1] = T1_init
#                 fix.T2[1] = T2_init
#                 fix.T3[1] = T3_init
#                 fix.T4[1] = T4_init
#                 fix.T5[1] = T5_init

#                 fix.CA1[1] = CA1_init
#                 fix.CA2[1] = CA2_init
#                 fix.CA3[1] = CA3_init
#                 fix.CA4[1] = CA4_init
#                 fix.CA5[1] = CA5_init
#                 fix.CB1[1] = CB1_init
#                 fix.CB2[1] = CB2_init
#                 fix.CB3[1] = CB3_init
#                 fix.CB4[1] = CB4_init
#                 fix.CB5[1] = CB5_init
#                 fix.CC1[1] = CC1_init
#                 fix.CC2[1] = CC2_init
#                 fix.CC3[1] = CC3_init
#                 fix.CC4[1] = CC4_init
#                 fix.CC5[1] = CC5_init
#                 fix.CD1[1] = CD1_init
#                 fix.CD2[1] = CD2_init
#                 fix.CD3[1] = CD3_init
#                 fix.CD4[1] = CD4_init
#                 fix.CD5[1] = CD5_init

#                 # Set state variable guesses to behavior associated with these
#                 for i = 1:Nd
#                     global k = i

#                     F1 = fix.F1[k]
#                     F2 = fix.F2[k]
#                     F3 = fix.F3[k]
#                     F4 = fix.F4[k]
#                     F5 = fix.F5[k]
#                     F6 = fix.F6[k]
#                     F7 = fix.F7[k]
#                     F8 = fix.F8[k]
#                     F9 = fix.F9[k]
#                     F10 = fix.F10[k]
#                     Fr1 = fix.Fr1[k]
#                     Fr2 = fix.Fr2[k]

#                     Q1 = fix.Q1[k]
#                     Q2 = fix.Q2[k]
#                     Q3 = fix.Q3[k]
#                     Q4 = fix.Q4[k]
#                     Q5 = fix.Q5[k]

#                     V1 = fix.V1[k]
#                     V2 = fix.V2[k]
#                     V3 = fix.V3[k]
#                     V4 = fix.V4[k]
#                     V5 = fix.V5[k]

#                     T1 = fix.T1[k]
#                     T2 = fix.T2[k]
#                     T3 = fix.T3[k]
#                     T4 = fix.T4[k]
#                     T5 = fix.T5[k]

#                     CA1 = fix.CA1[k]
#                     CB1 = fix.CB1[k]
#                     CC1 = fix.CC1[k]
#                     CD1 = fix.CD1[k]

#                     CA2 = fix.CA2[k]
#                     CB2 = fix.CB2[k]
#                     CC2 = fix.CC2[k]
#                     CD2 = fix.CD2[k]

#                     CA3 = fix.CA3[k]
#                     CB3 = fix.CB3[k]
#                     CC3 = fix.CC3[k]
#                     CD3 = fix.CD3[k]

#                     CA4 = fix.CA4[k]
#                     CB4 = fix.CB4[k]
#                     CC4 = fix.CC4[k]
#                     CD4 = fix.CD4[k]

#                     CA5 = fix.CA5[k]
#                     CB5 = fix.CB5[k]
#                     CC5 = fix.CC5[k]
#                     CD5 = fix.CD5[k]

#                     x0 = [fix.V1[k], fix.V2[k], fix.V3[k], fix.V4[k], fix.V5[k], fix.T1[k], fix.T2[k], fix.T3[k], fix.T4[k], fix.T5[k], fix.CA1[k], fix.CB1[k], fix.CC1[k], fix.CD1[k], fix.CA2[k], fix.CB2[k], fix.CC2[k], fix.CD2[k], fix.CA3[k], fix.CB3[k], fix.CC3[k], fix.CD3[k], fix.CA4[k], fix.CB4[k], fix.CC4[k], fix.CD4[k], fix.CA5[k], fix.CB5[k], fix.CC5[k], fix.CD5[k]]

#                     # Define the ODE function
#                     function f(y, p, t)
#                         # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                         V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y


#                         dV1 = F1 + F2 + Fr2 - F3
#                         dV2 = F3 + F4 - F5
#                         dV3 = F5 + F6 - F7
#                         dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                         dV5 = F10 + Fr1 - F9

#                         dT1 = (
#                             ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                               (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                               (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                               (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                               (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                              /
#                              (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                             (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                             (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                         dT2 = (
#                             (Q2 + F4 * CB0 * H_B(TB0) +
#                              (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                              (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                              (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                              (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                             /
#                             (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                             +
#                             (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                             /
#                             (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))


#                         dT3 = (
#                             ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                               (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                               (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                               (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                              /
#                              (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                             +
#                             (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                             /
#                             (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                         dT4 = ((Q4
#                                 + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                                 + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                                 + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                                 + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                                /
#                                (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                         dT5 = (
#                             ((Q5 +
#                               F10 * CD0 * H_D(TD0)
#                               + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                               + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                               + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                               + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                              /
#                              (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                             +
#                             ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                              /
#                              (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                         dCA1 = (
#                             ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CA1)
#                              /
#                              V1) - r1(T1, CA1, CB1))

#                         dCB1 = (
#                             ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CB1) /
#                              V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCC1 = (
#                             ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CC1) /
#                              V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCD1 = (
#                             ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CD1) /
#                              V1) + r2(T1, CB1, CC1, CD1))

#                         dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                         dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                         dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                         dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                         dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                         dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                         dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                         dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                         dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                         dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                         dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                         dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                         return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#                     end

#                     # Solve ODE 
#                     tspan = (0.0, dtd)
#                     prob = ODEProblem(f, x0, tspan)
#                     soln = solve(prob, Rosenbrock23())

#                     # Obtain the next values for each element
#                     fix.V1[k+1] = soln.u[end][1]
#                     fix.V2[k+1] = soln.u[end][2]
#                     fix.V3[k+1] = soln.u[end][3]
#                     fix.V4[k+1] = soln.u[end][4]
#                     fix.V5[k+1] = soln.u[end][5]

#                     fix.T1[k+1] = soln.u[end][6]
#                     fix.T2[k+1] = soln.u[end][7]
#                     fix.T3[k+1] = soln.u[end][8]
#                     fix.T4[k+1] = soln.u[end][9]
#                     fix.T5[k+1] = soln.u[end][10]

#                     fix.CA1[k+1] = soln.u[end][11]
#                     fix.CB1[k+1] = soln.u[end][12]
#                     fix.CC1[k+1] = soln.u[end][13]
#                     fix.CD1[k+1] = soln.u[end][14]

#                     fix.CA2[k+1] = soln.u[end][15]
#                     fix.CB2[k+1] = soln.u[end][16]
#                     fix.CC2[k+1] = soln.u[end][17]
#                     fix.CD2[k+1] = soln.u[end][18]

#                     fix.CA3[k+1] = soln.u[end][19]
#                     fix.CB3[k+1] = soln.u[end][20]
#                     fix.CC3[k+1] = soln.u[end][21]
#                     fix.CD3[k+1] = soln.u[end][22]

#                     fix.CA4[k+1] = soln.u[end][23]
#                     fix.CB4[k+1] = soln.u[end][24]
#                     fix.CC4[k+1] = soln.u[end][25]
#                     fix.CD4[k+1] = soln.u[end][26]

#                     fix.CA5[k+1] = soln.u[end][27]
#                     fix.CB5[k+1] = soln.u[end][28]
#                     fix.CC5[k+1] = soln.u[end][29]
#                     fix.CD5[k+1] = soln.u[end][30]

#                 end
#             end

#             # Initialize ig structure for centralized
#             for set_traj = 1

#                 # Set control variables to nominal steady state values
#                 ig.F1 .= F1_sp
#                 ig.F2 .= F2_sp
#                 ig.F3 .= F3_sp
#                 ig.F4 .= F4_sp
#                 ig.F5 .= F5_sp
#                 ig.F6 .= F6_sp
#                 ig.F7 .= F7_sp
#                 ig.F8 .= F8_sp
#                 ig.F9 .= F9_sp
#                 ig.F10 .= F10_sp
#                 ig.Fr1 .= Fr1_sp
#                 ig.Fr2 .= Fr2_sp

#                 ig.Q1 .= Q1_sp
#                 ig.Q2 .= Q2_sp
#                 ig.Q3 .= Q3_sp
#                 ig.Q4 .= Q4_sp
#                 ig.Q5 .= Q5_sp

#                 ig.V1[1] = V1_init
#                 ig.V2[1] = V2_init
#                 ig.V3[1] = V3_init
#                 ig.V4[1] = V4_init
#                 ig.V5[1] = V5_init
#                 ig.T1[1] = T1_init
#                 ig.T2[1] = T2_init
#                 ig.T3[1] = T3_init
#                 ig.T4[1] = T4_init
#                 ig.T5[1] = T5_init

#                 ig.CA1[1] = CA1_init
#                 ig.CA2[1] = CA2_init
#                 ig.CA3[1] = CA3_init
#                 ig.CA4[1] = CA4_init
#                 ig.CA5[1] = CA5_init
#                 ig.CB1[1] = CB1_init
#                 ig.CB2[1] = CB2_init
#                 ig.CB3[1] = CB3_init
#                 ig.CB4[1] = CB4_init
#                 ig.CB5[1] = CB5_init
#                 ig.CC1[1] = CC1_init
#                 ig.CC2[1] = CC2_init
#                 ig.CC3[1] = CC3_init
#                 ig.CC4[1] = CC4_init
#                 ig.CC5[1] = CC5_init
#                 ig.CD1[1] = CD1_init
#                 ig.CD2[1] = CD2_init
#                 ig.CD3[1] = CD3_init
#                 ig.CD4[1] = CD4_init
#                 ig.CD5[1] = CD5_init

#                 for i = 1:N
#                     global k = i

#                     F1 = ig.F1[k]
#                     F2 = ig.F2[k]
#                     F3 = ig.F3[k]
#                     F4 = ig.F4[k]
#                     F5 = ig.F5[k]
#                     F6 = ig.F6[k]
#                     F7 = ig.F7[k]
#                     F8 = ig.F8[k]
#                     F9 = ig.F9[k]
#                     F10 = ig.F10[k]
#                     Fr1 = ig.Fr1[k]
#                     Fr2 = ig.Fr2[k]
#                     Q1 = ig.Q1[k]
#                     Q2 = ig.Q2[k]
#                     Q3 = ig.Q3[k]
#                     Q4 = ig.Q4[k]
#                     Q5 = ig.Q5[k]

#                     V1 = ig.V1[k]
#                     V2 = ig.V2[k]
#                     V3 = ig.V3[k]
#                     V4 = ig.V4[k]
#                     V5 = ig.V5[k]

#                     T1 = ig.T1[k]
#                     T2 = ig.T2[k]
#                     T3 = ig.T3[k]
#                     T4 = ig.T4[k]
#                     T5 = ig.T5[k]

#                     CA1 = ig.CA1[k]
#                     CB1 = ig.CB1[k]
#                     CC1 = ig.CC1[k]
#                     CD1 = ig.CD1[k]

#                     CA2 = ig.CA2[k]
#                     CB2 = ig.CB2[k]
#                     CC2 = ig.CC2[k]
#                     CD2 = ig.CD2[k]

#                     CA3 = ig.CA3[k]
#                     CB3 = ig.CB3[k]
#                     CC3 = ig.CC3[k]
#                     CD3 = ig.CD3[k]

#                     CA4 = ig.CA4[k]
#                     CB4 = ig.CB4[k]
#                     CC4 = ig.CC4[k]
#                     CD4 = ig.CD4[k]

#                     CA5 = ig.CA5[k]
#                     CB5 = ig.CB5[k]
#                     CC5 = ig.CC5[k]
#                     CD5 = ig.CD5[k]

#                     x0 = [ig.V1[k], ig.V2[k], ig.V3[k], ig.V4[k], ig.V5[k], ig.T1[k], ig.T2[k], ig.T3[k], ig.T4[k], ig.T5[k], ig.CA1[k], ig.CB1[k], ig.CC1[k], ig.CD1[k], ig.CA2[k], ig.CB2[k], ig.CC2[k], ig.CD2[k], ig.CA3[k], ig.CB3[k], ig.CC3[k], ig.CD3[k], ig.CA4[k], ig.CB4[k], ig.CC4[k], ig.CD4[k], ig.CA5[k], ig.CB5[k], ig.CC5[k], ig.CD5[k]]

#                     # Define the ODE function
#                     function f(y, p, t)
#                         # V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y
#                         V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

#                         dV1 = F1 + F2 + Fr2 - F3
#                         dV2 = F3 + F4 - F5
#                         dV3 = F5 + F6 - F7
#                         dV4 = F7 + F9 - F8 - Fr1 - Fr2
#                         dV5 = F10 + Fr1 - F9

#                         dT1 = (
#                             ((Q1 + F1 * CA0 * H_A(TA0) + F2 * CB0 * H_B(TB0) +
#                               (Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F3 * CA1 * H_A(T1)) +
#                               (Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F3 * CB1 * H_B(T1)) +
#                               (Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F3 * CC1 * H_C(T1)) +
#                               (Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F3 * CD1 * H_D(T1)))
#                              /
#                              (CA1 * Cp_A * V1 + CB1 * Cp_B * V1 + CC1 * Cp_C * V1 + CD1 * Cp_D * V1)) +
#                             (-delH_r1 * r1(T1, CA1, CB1) - delH_r2 * r2(T1, CB1, CC1, CD1)) /
#                             (CA1 * Cp_A + CB1 * Cp_B + CC1 * Cp_C + CD1 * Cp_D))

#                         dT2 = (
#                             (Q2 + F4 * CB0 * H_B(TB0) +
#                              (F3 * CA1 * H_A(T1) - F5 * CA2 * H_A(T2)) +
#                              (F3 * CB1 * H_B(T1) - F5 * CB2 * H_B(T2)) +
#                              (F3 * CC1 * H_C(T1) - F5 * CC2 * H_C(T2)) +
#                              (F3 * CD1 * H_D(T1) - F5 * CD2 * H_D(T2)))
#                             /
#                             (CA2 * Cp_A * V2 + CB2 * Cp_B * V2 + CC2 * Cp_C * V2 + CD2 * Cp_D * V2)
#                             +
#                             (-delH_r1 * r1(T2, CA2, CB2) - delH_r2 * r2(T2, CB2, CC2, CD2))
#                             /
#                             (CA2 * Cp_A + CB2 * Cp_B + CC2 * Cp_C + CD2 * Cp_D))

#                         dT3 = (
#                             ((Q3 + F6 * CB0 * H_B(TB0) + (F5 * CA2 * H_A(T2) - F7 * CA3 * H_A(T3)) +
#                               (F5 * CB2 * H_B(T2) - F7 * CB3 * H_B(T3)) +
#                               (F5 * CC2 * H_C(T2) - F7 * CC3 * H_C(T3)) +
#                               (F5 * CD2 * H_D(T2) - F7 * CD3 * H_D(T3)))
#                              /
#                              (CA3 * Cp_A * V3 + CB3 * Cp_B * V3 + CC3 * Cp_C * V3 + CD3 * Cp_D * V3))
#                             +
#                             (-delH_r1 * r1(T3, CA3, CB3) - delH_r2 * r2(T3, CB3, CC3, CD3))
#                             /
#                             (CA3 * Cp_A + CB3 * Cp_B + CC3 * Cp_C + CD3 * Cp_D))

#                         dT4 = ((Q4
#                                 + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
#                                 + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
#                                 + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
#                                 + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
#                                /
#                                (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))
#                         dT5 = (
#                             ((Q5 +
#                               F10 * CD0 * H_D(TD0)
#                               + (Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_A(T4) - F9 * CA5 * H_A(T5))
#                               + (Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_B(T4) - F9 * CB5 * H_B(T5))
#                               + (Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_C(T4) - F9 * CC5 * H_C(T5))
#                               + (Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) * H_D(T4) - F9 * CD5 * H_D(T5)))
#                              /
#                              (CA5 * Cp_A * V5 + CB5 * Cp_B * V5 + CC5 * Cp_C * V5 + CD5 * Cp_D * V5))
#                             +
#                             ((-delH_r2 * r2(T5, CB5, CC5, CD5) - delH_r3 * r3(T5, CA5, CD5))
#                              /
#                              (CA5 * Cp_A + CB5 * Cp_B + CC5 * Cp_C + CD5 * Cp_D)))


#                         dCA1 = (
#                             ((F1 * CA0 + Fr2 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CA1)
#                              /
#                              V1) - r1(T1, CA1, CB1))

#                         dCB1 = (
#                             ((F2 * CB0 + Fr2 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CB1) /
#                              V1) - r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCC1 = (
#                             ((Fr2 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CC1) /
#                              V1) + r1(T1, CA1, CB1) - r2(T1, CB1, CC1, CD1))

#                         dCD1 = (
#                             ((Fr2 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5))
#                               -
#                               F3 * CD1) /
#                              V1) + r2(T1, CB1, CC1, CD1))

#                         dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
#                         dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCC2 = ((F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
#                         dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

#                         dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
#                         dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
#                         dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

#                         dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
#                         dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
#                         dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
#                         dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

#                         dCA5 = ((Fr1 * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CA5) / V5 - r3(T5, CA5, CD5))
#                         dCB5 = ((Fr1 * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CB5) / V5 - r2(T5, CB5, CC5, CD5))
#                         dCC5 = ((Fr1 * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F9 * CC5) / V5 - r2(T5, CB5, CC5, CD5) + 2 * r3(T5, CA5, CD5))
#                         dCD5 = ((Fr1 * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) + F10 * CD0 - F9 * CD5) / V5 + r2(T5, CB5, CC5, CD5) - r3(T5, CA5, CD5))

#                         return [dV1, dV2, dV3, dV4, dV5, dT1, dT2, dT3, dT4, dT5, dCA1, dCB1, dCC1, dCD1, dCA2, dCB2, dCC2, dCD2, dCA3, dCB3, dCC3, dCD3, dCA4, dCB4, dCC4, dCD4, dCA5, dCB5, dCC5, dCD5]
#                     end

#                     # Solve ODE 
#                     tspan = (0.0, dt)
#                     prob = ODEProblem(f, x0, tspan)
#                     soln = solve(prob, Rosenbrock23())

#                     # Obtain the next values for each element
#                     ig.V1[k+1] = soln.u[end][1]
#                     ig.V2[k+1] = soln.u[end][2]
#                     ig.V3[k+1] = soln.u[end][3]
#                     ig.V4[k+1] = soln.u[end][4]
#                     ig.V5[k+1] = soln.u[end][5]

#                     ig.T1[k+1] = soln.u[end][6]
#                     ig.T2[k+1] = soln.u[end][7]
#                     ig.T3[k+1] = soln.u[end][8]
#                     ig.T4[k+1] = soln.u[end][9]
#                     ig.T5[k+1] = soln.u[end][10]

#                     ig.CA1[k+1] = soln.u[end][11]
#                     ig.CB1[k+1] = soln.u[end][12]
#                     ig.CC1[k+1] = soln.u[end][13]
#                     ig.CD1[k+1] = soln.u[end][14]

#                     ig.CA2[k+1] = soln.u[end][15]
#                     ig.CB2[k+1] = soln.u[end][16]
#                     ig.CC2[k+1] = soln.u[end][17]
#                     ig.CD2[k+1] = soln.u[end][18]

#                     ig.CA3[k+1] = soln.u[end][19]
#                     ig.CB3[k+1] = soln.u[end][20]
#                     ig.CC3[k+1] = soln.u[end][21]
#                     ig.CD3[k+1] = soln.u[end][22]

#                     ig.CA4[k+1] = soln.u[end][23]
#                     ig.CB4[k+1] = soln.u[end][24]
#                     ig.CC4[k+1] = soln.u[end][25]
#                     ig.CD4[k+1] = soln.u[end][26]

#                     ig.CA5[k+1] = soln.u[end][27]
#                     ig.CB5[k+1] = soln.u[end][28]
#                     ig.CC5[k+1] = soln.u[end][29]
#                     ig.CD5[k+1] = soln.u[end][30]
#                 end

#             end


#             global cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5 = cmpc()
#             centr_ISE, centr_ISC, centr_PI = getPI(cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5)

#             ex.F1c[mpc_step] = cF1[1]
#             ex.F2c[mpc_step] = cF2[1]
#             ex.F3c[mpc_step] = cF3[1]
#             ex.F4c[mpc_step] = cF4[1]
#             ex.F5c[mpc_step] = cF5[1]
#             ex.F6c[mpc_step] = cF6[1]
#             ex.F7c[mpc_step] = cF7[1]
#             ex.F8c[mpc_step] = cF8[1]
#             ex.F9c[mpc_step] = cF9[1]
#             ex.F10c[mpc_step] = cF10[1]
#             ex.Fr1c[mpc_step] = cFr1[1]
#             ex.Fr2c[mpc_step] = cFr2[1]

#             ex.Q1c[mpc_step] = cQ1[1]
#             ex.Q2c[mpc_step] = cQ2[1]
#             ex.Q3c[mpc_step] = cQ3[1]
#             ex.Q4c[mpc_step] = cQ4[1]
#             ex.Q5c[mpc_step] = cQ5[1]


#             centr_PI_vec[mpc_step] = centr_PI
#             centr_ISE_vec[mpc_step] = centr_ISE
#             centr_ISC_vec[mpc_step] = centr_ISC
#             centr_time[mpc_step] = cmpc_solve_time

#             println("cPI = $(centr_PI)")
#             println("cISE = $(centr_ISE)")
#             println("cISC = $(centr_ISC)")
#             println("time = $(cmpc_solve_time)")
#             println()

#             global dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5 = dmpc()
#             distr_ISE, distr_ISC, distr_PI = getPI(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)

#             ex.F1d[mpc_step] = dF1[1]
#             ex.F2d[mpc_step] = dF2[1]
#             ex.F3d[mpc_step] = dF3[1]
#             ex.F4d[mpc_step] = dF4[1]
#             ex.F5d[mpc_step] = dF5[1]
#             ex.F6d[mpc_step] = dF6[1]
#             ex.F7d[mpc_step] = dF7[1]
#             ex.F8d[mpc_step] = dF8[1]
#             ex.F9d[mpc_step] = dF9[1]
#             ex.F10d[mpc_step] = dF10[1]
#             ex.Fr1d[mpc_step] = dFr1[1]
#             ex.Fr2d[mpc_step] = dFr2[1]

#             ex.Q1d[mpc_step] = dQ1[1]
#             ex.Q2d[mpc_step] = dQ2[1]
#             ex.Q3d[mpc_step] = dQ3[1]
#             ex.Q4d[mpc_step] = dQ4[1]
#             ex.Q5d[mpc_step] = dQ5[1]

#             distr_PI_vec[mpc_step] = distr_PI
#             distr_ISE_vec[mpc_step] = distr_ISE
#             distr_ISC_vec[mpc_step] = distr_ISC
#             distr_time[mpc_step] = dmpc_solve_time

#             println("dPI = $(distr_PI)")
#             println("dISE = $(distr_ISE)")
#             println("dISC = $(distr_ISC)")
#             println("time = $(dmpc_solve_time)")
#             println()

#             ex.PIc[mpc_step] = centr_PI
#             ex.PId[mpc_step] = distr_PI
#             ex.ISEc[mpc_step] = centr_ISE
#             ex.ISEd[mpc_step] = distr_ISE
#             ex.ISCc[mpc_step] = centr_ISC
#             ex.ISCd[mpc_step] = distr_ISC

#             if centr_PI < distr_PI
#                 println("Centralized chosen")
#                 println()
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(cF1, cF2, cF3, cF4, cF5, cF6, cF7, cF8, cF9, cF10, cFr1, cFr2, cQ1, cQ2, cQ3, cQ4, cQ5)
#                 takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

#                 predict.F1[mpc_step] = cF1[1]
#                 predict.F2[mpc_step] = cF2[1]
#                 predict.F3[mpc_step] = cF3[1]
#                 predict.F4[mpc_step] = cF4[1]
#                 predict.F5[mpc_step] = cF5[1]
#                 predict.F6[mpc_step] = cF6[1]
#                 predict.F7[mpc_step] = cF7[1]
#                 predict.F8[mpc_step] = cF8[1]
#                 predict.F9[mpc_step] = cF9[1]
#                 predict.F10[mpc_step] = cF10[1]
#                 predict.Fr1[mpc_step] = cFr1[1]
#                 predict.Fr2[mpc_step] = cFr2[1]

#                 predict.Q1[mpc_step] = cQ1[1]
#                 predict.Q2[mpc_step] = cQ2[1]
#                 predict.Q3[mpc_step] = cQ3[1]
#                 predict.Q4[mpc_step] = cQ4[1]
#                 predict.Q5[mpc_step] = cQ5[1]
#             else
#                 println("Distributed chosen")
#                 println()
#                 V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5 = getTraj(dF1, dF2, dF3, dF4, dF5, dF6, dF7, dF8, dF9, dF10, dFr1, dFr2, dQ1, dQ2, dQ3, dQ4, dQ5)
#                 takeMPCstep(V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CA2, CA3, CA4, CA5, CB1, CB2, CB3, CB4, CB5, CC1, CC2, CC3, CC4, CC5, CD1, CD2, CD3, CD4, CD5)

#                 predict.F1[mpc_step] = dF1[1]
#                 predict.F2[mpc_step] = dF2[1]
#                 predict.F3[mpc_step] = dF3[1]
#                 predict.F4[mpc_step] = dF4[1]
#                 predict.F5[mpc_step] = dF5[1]
#                 predict.F6[mpc_step] = dF6[1]
#                 predict.F7[mpc_step] = dF7[1]
#                 predict.F8[mpc_step] = dF8[1]
#                 predict.F9[mpc_step] = dF9[1]
#                 predict.F10[mpc_step] = dF10[1]
#                 predict.Fr1[mpc_step] = dFr1[1]
#                 predict.Fr2[mpc_step] = dFr2[1]

#                 predict.Q1[mpc_step] = dQ1[1]
#                 predict.Q2[mpc_step] = dQ2[1]
#                 predict.Q3[mpc_step] = dQ3[1]
#                 predict.Q4[mpc_step] = dQ4[1]
#                 predict.Q5[mpc_step] = dQ5[1]
#             end

#         end

#     end


#     mhmpc_sim()

#     decision = zeros(P)
#     for i = 1:P
#         if distr_PI_vec[i] <= centr_PI_vec[i]
#             decision[i] = 1
#         end
#     end

#     print(decision)
#     plot(decision)
#     plot(log10.(centr_PI_vec[1:P]), label="Log10 Centralized PI")
#     plot!(log10.(distr_PI_vec[1:P]), label="Distributed", ylabel="Log10 PI")

#     plot(predict.T3)
#     hline!([T3_sp])

#     PI_diff = centr_PI_vec - distr_PI_vec
#     plot(centr_time, label="Centralized time", ylabel="Seconds")
#     plot!(distr_time, label="Distributed time")

#     df = DataFrame(PIc=ex.PIc, ISEc=ex.ISEc, ISCc=ex.ISCc,
#         PId=ex.PId, ISEd=ex.ISEd, ISCd=ex.ISCd,
#         V1=ex.V1, V2=ex.V2, V3=ex.V3, V4=ex.V4, V5=ex.V5,
#         T1=ex.T1, T2=ex.T2, T3=ex.T3, T4=ex.T4, T5=ex.T5,
#         CA1=ex.CA1, CA2=ex.CA2, CA3=ex.CA3, CA4=ex.CA4, CA5=ex.CA5,
#         CB1=ex.CB1, CB2=ex.CB2, CB3=ex.CB3, CB4=ex.CB4, CB5=ex.CB5,
#         CC1=ex.CC1, CC2=ex.CC2, CC3=ex.CC3, CC4=ex.CC4, CC5=ex.CC5,
#         CD1=ex.CD1, CD2=ex.CD2, CD3=ex.CD3, CD4=ex.CD4, CD5=ex.CD5,
#         F1d=ex.F1d, F2d=ex.F2d, F3d=ex.F3d, F4d=ex.F4d, F5d=ex.F5d,
#         F6d=ex.F6d, F7d=ex.F7d, F8d=ex.F8d, F9d=ex.F9d, F10d=ex.F10d,
#         Fr1d=ex.Fr1d, Fr2d=ex.Fr2d,
#         Q1d=ex.Q1d, Q2d=ex.Q2d, Q3d=ex.Q3d, Q4d=ex.Q4d, Q5d=ex.Q5d,
#         F1c=ex.F1c, F2c=ex.F2c, F3c=ex.F3c, F4c=ex.F4c, F5c=ex.F5c,
#         F6c=ex.F6c, F7c=ex.F7c, F8c=ex.F8c, F9c=ex.F9c, F10c=ex.F10c,
#         Fr1c=ex.Fr1c, Fr2c=ex.Fr2c,
#         Q1c=ex.Q1c, Q2c=ex.Q2c, Q3c=ex.Q3c, Q4c=ex.Q4c, Q5c=ex.Q5c
#     )
#     CSV.write("mhmpc-src-results-$(mhmpc_src_instance)-t-12-sec.csv", df)

#     global mhmpc_src_instance = mhmpc_src_instance + 1

# end
