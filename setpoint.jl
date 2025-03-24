using Plots, OrdinaryDiffEq, LinearAlgebra, CSV, DataFrames, Random

Random.seed!(123)

global F1_sp = 7.1E-3 # Parameter
global F2_sp = 8.697E-4 # Parameter
global Fr1_sp = 6E-3 # Parameter
global Fr2_sp = 6E-3 # = Fr1_sp
global F3_sp = 0.0139697 # = F1_sp + F2_sp + Fr2_sp
global F4_sp = 8.697E-4 # = F2_sp
global F6_sp = 8.697E-4 # = F2_sp
global F5_sp = 0.01483939999 # = F4_sp + F3_sp
global F7_sp = 0.0157091 # = F5_sp + F6_sp
global F10_sp = 2.3E-3 # Parameter
global F9_sp = 0.0083 # = Fr1_sp + F10_sp
global F8_sp = 0.012009099999 # = F7_sp + F9_sp - (Fr1_sp + Fr2_sp)

global Q1_sp = -4.4E6 # Parameter
global Q2_sp = -4.6E6  # Parameter
global Q3_sp = -4.7E6  # Parameter
global Q4_sp = 9.2E6  # Parameter
global Q5_sp = 5.6E6  # Parameter

global V1_init = 1 # Parameter
global V2_init = 1 # Parameter
global V3_init = 1 # Parameter
global V4_init = 3 # Parameter
global V5_init = 1 # Parameter


mutable struct Fixed

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


N = 1000 # Control horizon
P = 3 * N # Prediction horizon
dt = 60 # Sampling time of the system
# function init_sp_and_params()


fix = Fixed(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))

global T1_init = 473.9877033595075
global T2_init = 473.0184186946286
global T3_init = 468.8795312555716
global T4_init = 465.2055166084489
global T5_init = 468.1548934480822

global V1_init = 1.0
global V2_init = 1.0
global V3_init = 1.0
global V4_init = 3.0
global V5_init = 1.0

global CA1_init = 9101.95771973209
global CA2_init = 7548.2898874749335
global CA3_init = 6163.450393990001
global CA4_init = 1759.1824969530212
global CA5_init = 5815.836344458993

global CB1_init = 22.402693151343186
global CB2_init = 23.77755990309227
global CB3_init = 25.23953314492567
global CB4_init = 14.035249229042014
global CB5_init = 4.593654083519995

global CC1_init = 1116.111360164459
global CC2_init = 1905.2810570325862
global CC3_init = 2613.7136352864595
global CC4_init = 5403.81633425641
global CC5_init = 3713.5923653603145

global CD1_init = 220.2724041134235
global CD2_init = 373.0064317311925
global CD3_init = 505.38715944049324
global CD4_init = 736.123215014287
global CD5_init = 204.30702354138066

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

r1(T, CA, CB) = 0.0840 * exp(-9502 / (R * T)) * CA^(0.32) * CB^(1.5)
r2(T, CB, CC, CD) = (0.0850 * exp(-20643 / (R * T)) * CB^(2.5) * CC^(0.5)) / (1 + kEB2(T) * CD)
r3(T, CA, CD) = (66.1 * exp(-61280 / (R * T)) * CA^(1.0218) * CD) / (1 + kEB3(T) * CA)

# Molar flow in the overhead stream
MA(T, F7, Ci3, F9, Ci5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) = 0.8 * ((alpha_A(T) * (F7 * Ci3 + F9 * Ci5) * ((F7 * CA3 + F9 * CA5) + (F7 * CB3 + F9 * CB5) + (F7 * CC3 + F9 * CC5) + (F7 * CD3 + F9 * CD5))) / (alpha_A(T) * (F7 * CA3 + F9 * CA5) + alpha_B(T) * (F7 * CB3 + F9 * CB5) + alpha_C(T) * (F7 * CC3 + F9 * CC5) + alpha_D(T) * (F7 * CD3 + F9 * CD5)))
MB(T, F7, Ci3, F9, Ci5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) = 0.8 * ((alpha_B(T) * (F7 * Ci3 + F9 * Ci5) * ((F7 * CA3 + F9 * CA5) + (F7 * CB3 + F9 * CB5) + (F7 * CC3 + F9 * CC5) + (F7 * CD3 + F9 * CD5))) / (alpha_A(T) * (F7 * CA3 + F9 * CA5) + alpha_B(T) * (F7 * CB3 + F9 * CB5) + alpha_C(T) * (F7 * CC3 + F9 * CC5) + alpha_D(T) * (F7 * CD3 + F9 * CD5)))
MC(T, F7, Ci3, F9, Ci5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) = 0.8 * ((alpha_C(T) * (F7 * Ci3 + F9 * Ci5) * ((F7 * CA3 + F9 * CA5) + (F7 * CB3 + F9 * CB5) + (F7 * CC3 + F9 * CC5) + (F7 * CD3 + F9 * CD5))) / (alpha_A(T) * (F7 * CA3 + F9 * CA5) + alpha_B(T) * (F7 * CB3 + F9 * CB5) + alpha_C(T) * (F7 * CC3 + F9 * CC5) + alpha_D(T) * (F7 * CD3 + F9 * CD5)))
MD(T, F7, Ci3, F9, Ci5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) = 0.8 * ((alpha_D(T) * (F7 * Ci3 + F9 * Ci5) * ((F7 * CA3 + F9 * CA5) + (F7 * CB3 + F9 * CB5) + (F7 * CC3 + F9 * CC5) + (F7 * CD3 + F9 * CD5))) / (alpha_A(T) * (F7 * CA3 + F9 * CA5) + alpha_B(T) * (F7 * CB3 + F9 * CB5) + alpha_C(T) * (F7 * CC3 + F9 * CC5) + alpha_D(T) * (F7 * CD3 + F9 * CD5)))

# Concentration of species in the recycle stream
CAr(MA, MB, MC, MD) = MA / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
CBr(MA, MB, MC, MD) = MB / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
CCr(MA, MB, MC, MD) = MC / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))
CDr(MA, MB, MC, MD) = MD / ((MA / CA0) + (MB / CB0) + (MC / CC0) + (MD / CD0))

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

F1 = ones(N + 1) .* F1_sp
F2 = ones(N + 1) .* F2_sp
F3 = ones(N + 1) .* F3_sp
F4 = ones(N + 1) .* F4_sp
F5 = ones(N + 1) .* F5_sp
F6 = ones(N + 1) .* F6_sp
F7 = ones(N + 1) .* F7_sp
F8 = ones(N + 1) .* F8_sp
F9 = ones(N + 1) .* F9_sp
F10 = ones(N + 1) .* F10_sp
Fr1 = ones(N + 1) .* Fr1_sp
Fr2 = ones(N + 1) .* Fr2_sp

Q1 = ones(N + 1) .* Q1_sp
Q2 = ones(N + 1) .* Q2_sp
Q3 = ones(N + 1) .* Q3_sp
Q4 = ones(N + 1) .* Q4_sp
Q5 = ones(N + 1) .* Q5_sp

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
        # V1[k], V2[k], V3[k], V4[k], V5[k], T1[k], T2[k], T3[k], T4[k], T5[k], CA1[k], CB1[k], CC1[k], CD1[k], CA2[k], CB2[k], CC2[k], CD2[k], CA3[k], CB3[k], CC3[k], CD3[k], CA4[k], CB4[k], CC4[k], CD4[k], CA5[k], CB5[k], CC5[k], CD5[k] = y
        V1, V2, V3, V4, V5, T1, T2, T3, T4, T5, CA1, CB1, CC1, CD1, CA2, CB2, CC2, CD2, CA3, CB3, CC3, CD3, CA4, CB4, CC4, CD4, CA5, CB5, CC5, CD5 = y

        dV1 = (F1 + F2 + Fr2 - F3)

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

        dV2 = (F3 + F4 - F5)

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


        dCA2 = (((F3 * CA1 - F5 * CA2) / V2) - r1(T2, CA2, CB2))
        dCB2 = ((F3 * CB1 + F4 * CB0 - F5 * CB2) / V2 - r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2))
        dCC2 = (F3 * CC1 - F5 * CC2) / V2 + r1(T2, CA2, CB2) - r2(T2, CB2, CC2, CD2)
        dCD2 = ((F3 * CD1 - F5 * CD2) / V2 + r2(T2, CB2, CC2, CD2))

        dV3 = (F5 + F6 - F7)


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

        dCA3 = (((F5 * CA2 - F7 * CA3) / V3) - r1(T3, CA3, CB3))
        dCB3 = (((F5 * CB2 + F6 * CB0 - F7 * CB3) / V3) - r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
        dCC3 = ((F5 * CC2 - F7 * CC3) / V3 + r1(T3, CA3, CB3) - r2(T3, CB3, CC3, CD3))
        dCD3 = ((F5 * CD2 - F7 * CD3) / V3 + r2(T3, CB3, CC3, CD3))

        dV4 = (F7 + F9 - F8 - Fr1 - Fr2)

        dT4 =
            ((Q4
              + (F7 * CA3 * H_A(T3) + F9 * CA5 * H_A(T5) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_A(T4) - F8 * CA4 * H_A(T4) - MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_A)
              + (F7 * CB3 * H_B(T3) + F9 * CB5 * H_B(T5) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_B(T4) - F8 * CB4 * H_B(T4) - MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_B)
              + (F7 * CC3 * H_C(T3) + F9 * CC5 * H_C(T5) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_C(T4) - F8 * CC4 * H_C(T4) - MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_C)
              + (F7 * CD3 * H_D(T3) + F9 * CD5 * H_D(T5) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_D(T4) - F8 * CD4 * H_D(T4) - MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5) * H_vap_D))
             /
             (CA4 * Cp_A * V4 + CB4 * Cp_B * V4 + CC4 * Cp_C * V4 + CD4 * Cp_D * V4))

        dCA4 = ((F7 * CA3 + F9 * CA5 - (Fr1 + Fr2) * CAr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CA4) / V4)
        dCB4 = ((F7 * CB3 + F9 * CB5 - (Fr1 + Fr2) * CBr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CB4) / V4)
        dCC4 = ((F7 * CC3 + F9 * CC5 - (Fr1 + Fr2) * CCr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CC4) / V4)
        dCD4 = ((F7 * CD3 + F9 * CD5 - (Fr1 + Fr2) * CDr(MA(T4, F7, CA3, F9, CA5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MB(T4, F7, CB3, F9, CB5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MC(T4, F7, CC3, F9, CC5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5), MD(T4, F7, CD3, F9, CD5, CA3, CB3, CC3, CD3, CA5, CB5, CC5, CD5)) - F8 * CD4) / V4)

        dV5 = (F10 + Fr1 - F9)

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


V1 = fix.V1
V2 = fix.V2
V3 = fix.V3
V4 = fix.V4
V5 = fix.V5

T1 = fix.T1
T2 = fix.T2
T3 = fix.T3
T4 = fix.T4
T5 = fix.T5

CA1 = fix.CA1
CA2 = fix.CA2
CA3 = fix.CA3
CA4 = fix.CA4
CA5 = fix.CA5

CB1 = fix.CB1
CB2 = fix.CB2
CB3 = fix.CB3
CB4 = fix.CB4
CB5 = fix.CB5

CC1 = fix.CC1
CC2 = fix.CC2
CC3 = fix.CC3
CC4 = fix.CC4
CC5 = fix.CC5

CD1 = fix.CD1
CD2 = fix.CD2
CD3 = fix.CD3
CD4 = fix.CD4
CD5 = fix.CD5

V1 = V1[end]
V2 = V2[end]
V3 = V3[end]
V4 = V4[end]
V5 = V5[end]

T1 = T1[end]
T2 = T2[end]
T3 = T3[end]
T4 = T4[end]
T5 = T5[end]

CA1 = CA1[end]
CA2 = CA2[end]
CA3 = CA3[end]
CA4 = CA4[end]
CA5 = CA5[end]

CB1 = CB1[end]
CB2 = CB2[end]
CB3 = CB3[end]
CB4 = CB4[end]
CB5 = CB5[end]

CC1 = CC1[end]
CC2 = CC2[end]
CC3 = CC3[end]
CC4 = CC4[end]
CC5 = CC5[end]

CD1 = CD1[end]
CD2 = CD2[end]
CD3 = CD3[end]
CD4 = CD4[end]
CD5 = CD5[end]

println("
global T1_init = $T1
global T2_init = $T2
global T3_init = $T3
global T4_init = $T4
global T5_init = $T5

global V1_init = $V1
global V2_init = $V2
global V3_init = $V3
global V4_init = $V4
global V5_init = $V5

global CA1_init = $CA1
global CA2_init = $CA2
global CA3_init = $CA3
global CA4_init = $CA4
global CA5_init = $CA5

global CB1_init = $CB1
global CB2_init = $CB2
global CB3_init = $CB3
global CB4_init = $CB4
global CB5_init = $CB5

global CC1_init = $CC1
global CC2_init = $CC2
global CC3_init = $CC3
global CC4_init = $CC4
global CC5_init = $CC5

global CD1_init = $CD1
global CD2_init = $CD2
global CD3_init = $CD3
global CD4_init = $CD4
global CD5_init = $CD5
")

plot(fix.T1)

# v = [V1 V2 V3 V4 V5 T1 T2 T3 T4 T5 CA1 CA2 CA3 CA4 CA5 CB1 CB2 CB3 CB4 CB5 CC1 CC2 CC3 CC4 CC5 CD1 CD2 CD3 CD4 CD5 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 Fr1 Fr2]
# all(x -> x > 0, v)
# if all(x -> x > 0, v) == true
#     if T1 > 400 && T2 > 400 && T3 > 400 && T4 > 400 && T5 > 400 && T1 < 540 && T2 < 540 && T3 < 540 && T4 < 540 && T5 < 540
#         println("all postive and temperatures good")
#         global df = DataFrame(V1=V1, V2=V2, V3=V3, V4=V4, V5=V5, T1=T1, T2=T2, T3=T3, T4=T4, T5=T5, CA1=CA1, CA2=CA2, CA3=CA3, CA4=CA4, CA5=CA5, CB1=CB1, CB2=CB2, CB3=CB3, CB4=CB4, CB5=CB5, CC1=CC1, CC2=CC2, CC3=CC3, CC4=CC4, CC5=CC5, CD1=CD1, CD2=CD2, CD3=CD3, CD4=CD4, CD5=CD5, F1=F1, F2=F2, F3=F3, F4=F4, F5=F5, F6=F6, F7=F7, F8=F8, F9=F9, F10=F10, Fr1=Fr1, Fr2=Fr2, Q1=Q1, Q2=Q2, Q3=Q3, Q4=Q4, Q5=Q5)
#         CSV.write("data-$(i).csv", df)
#         i = i + 1
#     end
# end

