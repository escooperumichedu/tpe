using JuMP, Ipopt, CSV, DataFrames, Plots

dt = .05  # hr (1min) - all dynamics are in terms of hours

CA1_init = 2.8
CA2_init = 2.75
CA3_init = 2.7
CA4_init = 2.8

T1_init = 340
T2_init = 350
T3_init = 345
T4_init = 360

V1_init = 1.5
V2_init = 3.5
V3_init = 3.5
V4_init = 4.0

V1_sp = 1.0
V2_sp = 3.0
V3_sp = 4.0
V4_sp = 6.0

delH1 = -5.0E4
delH2 = -5.2E4
delH3 = -5.0E4

k10 = 3.0E6
k20 = 3.0E5
k30 = 3.0E5

E1 = 5.0E4
E2 = 7.53E4
E3 = 7.53E4

F1_sp = 35
F2_sp = 45
F3_sp = 33
F4_sp = 45

F01_sp = 5
F02_sp = 10
F03_sp = 8
F04_sp = 12

Fr1_sp = 20
Fr2_sp = 10

T01 = 300
T02 = 300
T03 = 300
T04 = 300

CA01 = 4
CA02 = 2
CA03 = 3
CA04 = 3.5

rho = 1000
cp = 0.231
tau = 0.5

T01 = 300
T02 = 300

R = 8.314

epsilon = 0.8

T1_sp = 310.8555541924274
T2_sp = 310.85006015821057
T3_sp = 312.4861747507381
T4_sp = 311.18104842259197
CA1_sp = 3.0316455602508587
CA2_sp = 2.800160642961407
CA3_sp = 2.8440480178008145
CA4_sp = 3.014060835642008

Q1_sp = 1E4
Q2_sp = 2E4
Q3_sp = 2.5E4
Q4_sp = 1E4


function gW(sp)
    wt = -Float64(floor(log10(abs(sp^2))))
    return 1 * 10^(wt)
end


mutable struct Weights
    ca1
    ca2
    ca3
    ca4
    t1
    t2
    t3
    t4
    v1
    v2
    v3
    v4
    f1
    f2
    f3
    f4
    f01
    f02
    f03
    f04
    fr1
    fr2
    q1
    q2
    q3
    q4
end

w = Weights(gW(CA1_sp), gW(CA2_sp), 10 * gW(CA3_sp), 10 * gW(CA4_sp), 10 * gW(T1_sp), 10 * gW(T2_sp), 10 * gW(T3_sp), 10 * gW(T4_sp), gW(V1_sp), gW(V2_sp), gW(V3_sp), gW(V4_sp), gW(F1_sp), gW(F2_sp), gW(F3_sp), gW(F4_sp), gW(F01_sp), gW(F02_sp), gW(F03_sp), gW(F04_sp), gW(Fr1_sp), gW(Fr2_sp), gW(Q1_sp), gW(Q2_sp), gW(Q3_sp), gW(Q4_sp))


N = 60

mpc = JuMP.Model(Ipopt.Optimizer)


JuMP.@variables mpc begin

    T1[k=0:N], (lower_bound=200, upper_bound=400)
    T2[k=0:N], (lower_bound=200, upper_bound=400)
    T3[k=0:N], (lower_bound=200, upper_bound=400)
    T4[k=0:N], (lower_bound=200, upper_bound=400)

    V1[k=0:N], (lower_bound=0.3, upper_bound=10)
    V2[k=0:N], (lower_bound=0.3, upper_bound=10)
    V3[k=0:N], (lower_bound=0.3, upper_bound=10)
    V4[k=0:N], (lower_bound=0.3, upper_bound=10)

    CA1[k=0:N], (lower_bound=1e-5, upper_bound=10)
    CA2[k=0:N], (lower_bound=1e-5, upper_bound=10)
    CA3[k=0:N], (lower_bound=1e-5, upper_bound=10)
    CA4[k=0:N], (lower_bound=1e-5, upper_bound=10)

    Q1[k=0:N], (lower_bound=Q1_sp * 0.5, upper_bound=Q1_sp * 1.5)
    Q2[k=0:N], (lower_bound=Q2_sp * 0.5, upper_bound=Q2_sp * 1.5)
    Q3[k=0:N], (lower_bound=Q3_sp * 0.5, upper_bound=Q3_sp * 1.5)
    Q4[k=0:N], (lower_bound=Q4_sp * 0.5, upper_bound=Q4_sp * 1.5)

    F1[k=0:N], (lower_bound=1, upper_bound=60)
    F2[k=0:N], (lower_bound=1, upper_bound=60)
    F3[k=0:N], (lower_bound=1, upper_bound=60)
    F4[k=0:N], (lower_bound=1, upper_bound=60)

    F01[k=0:N], (lower_bound=1, upper_bound=60)
    F02[k=0:N], (lower_bound=1, upper_bound=60)
    F03[k=0:N], (lower_bound=1, upper_bound=60)
    F04[k=0:N], (lower_bound=1, upper_bound=60)

    Fr1[k=0:N], (lower_bound=1, upper_bound=60)
    Fr2[k=0:N], (lower_bound=1, upper_bound=60)


end



@constraints mpc begin

    T1_initial_condition, T1[0] == T1_init
    T2_initial_condition, T2[0] == T2_init
    T3_initial_condition, T3[0] == T3_init
    T4_initial_condition, T4[0] == T4_init
    CA1_initial_condition, CA1[0] == CA1_init
    CA2_initial_condition, CA2[0] == CA2_init
    CA3_initial_condition, CA3[0] == CA3_init
    CA4_initial_condition, CA4[0] == CA4_init

    V1_initial_condition, V1[0] == V1_init
    V2_initial_condition, V2[0] == V2_init
    V3_initial_condition, V3[0] == V3_init
    V4_initial_condition, V4[0] == V4_init

    recycle_limit_01[k=0:N], Fr1[k] <= epsilon * F2[k]
    recycle_limit_02[k=0:N], Fr2[k] <= epsilon * F4[k]

end

@NLconstraints mpc begin


    dV1_dt[k=0:N-1], V1[k+1] == V1[k] + (F01[k] + Fr2[k] + Fr1[k] - F1[k]) * dt
    dV2_dt[k=0:N-1], V2[k+1] == V2[k] + (F1[k] + F02[k] - F2[k]) * dt
    dV3_dt[k=0:N-1], V3[k+1] == V3[k] + ((F2[k] - Fr1[k]) + F03[k] - F3[k]) * dt
    dV4_dt[k=0:N-1], V4[k+1] == V4[k] + (F3[k] + F04[k] - F4[k]) * dt

    holdUp1[k=0:N], F01[k] + Fr2[k] + Fr1[k] - F1[k] == -(V1[k] - V1_sp) / tau
    holdUp2[k=0:N], F1[k] + F02[k] - F2[k] == -(V2[k] - V2_sp) / tau
    holdUp3[k=0:N], (F2[k] - Fr1[k]) + F03[k] - F3[k] == -(V3[k] - V3_sp) / tau
    holdUp4[k=0:N], F3[k] + F04[k] - F4[k] == -(V4[k] - V4_sp) / tau


    dCA1_dt[k=0:N-1], CA1[k+1] == CA1[k] + ((F01[k] / V1[k]) * (CA01 - CA1[k]) + (Fr1[k] / V1[k]) * (CA2[k] - CA1[k]) + (Fr2[k] / V1[k]) * (CA4[k] - CA1[k]) - (k10 * exp(-E1 / (R * T1[k])) * CA1[k] + k20 * exp(-E2 / (R * T1[k])) * CA1[k] + k30 * exp(-E3 / (R * T1[k])) * CA1[k])) * dt
    dT1_dt[k=0:N-1], T1[k+1] == T1[k] + ((F01[k] / V1[k]) * (T01 - T1[k]) + (Fr1[k] / V1[k]) * (T2[k] - T1[k]) + (Fr2[k] / V1[k]) * (T4[k] - T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T1[k])) * CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T1[k])) * CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T1[k])) * CA1[k]) + Q1[k] / (rho * cp * V1[k])) * dt

    dCA2_dt[k=0:N-1], CA2[k+1] == CA2[k] + ((F1[k] / V2[k]) * (CA1[k] - CA2[k]) + (F02[k] / V2[k]) * (CA02 - CA2[k]) - (k10 * exp(-E1 / (R * T2[k])) * CA2[k] + k20 * exp(-E2 / (R * T2[k])) * CA2[k] + k30 * exp(-E3 / (R * T2[k])) * CA2[k])) * dt
    dT2_dt[k=0:N-1], T2[k+1] == T2[k] + ((F1[k] / V2[k]) * (T1[k] - T2[k]) + (F02[k] / V2[k]) * (T02 - T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T2[k])) * CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T2[k])) * CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T2[k])) * CA2[k]) + Q2[k] / (rho * cp * V2[k])) * dt

    dCA3_dt[k=0:N-1], CA3[k+1] == CA3[k] + (((F2[k] - Fr1[k]) / V3[k]) * (CA2[k] - CA3[k]) + (F03[k] / V3[k]) * (CA03 - CA3[k]) - (k10 * exp(-E1 / (R * T3[k])) * CA3[k] + k20 * exp(-E2 / (R * T3[k])) * CA3[k] + k30 * exp(-E3 / (R * T3[k])) * CA3[k])) * dt
    dT3_dt[k=0:N-1], T3[k+1] == T3[k] + (((F2[k] - Fr1[k]) / V3[k]) * (T2[k] - T3[k]) + (F03[k] / V3[k]) * (T03 - T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T3[k])) * CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T3[k])) * CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T3[k])) * CA3[k]) + Q3[k] / (rho * cp * V3[k])) * dt

    dCA4_dt[k=0:N-1], CA4[k+1] == CA4[k] + ((F3[k] / V4[k]) * (CA3[k] - CA4[k]) + (F04[k] / V4[k]) * (CA04 - CA4[k]) - (k10 * exp(-E1 / (R * T4[k])) * CA4[k] + k20 * exp(-E2 / (R * T4[k])) * CA4[k] + k30 * exp(-E3 / (R * T4[k])) * CA4[k])) * dt
    dT4_dt[k=0:N-1], T4[k+1] == T4[k] + ((F3[k] / V4[k]) * (T3[k] - T4[k]) + (F04[k] / V4[k]) * (T04 - T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T4[k])) * CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T4[k])) * CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T4[k])) * CA4[k]) + Q4[k] / (rho * cp * V4[k])) * dt

end


@NLobjective(mpc, Min, sum(w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 +
                           w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 +
                           w.v1 * (V1[k] - V1_sp)^2 + w.v2 * (V2[k] - V2_sp)^2 + w.v3 * (V3[k] - V3_sp)^2 + w.v4 * (V4[k] - V4_sp)^2 +
                           w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
                           w.f01 * (F01[k] - F01_sp)^2 + w.f02 * (F02[k] - F02_sp)^2 + w.f03 * (F03[k] - F03_sp)^2 + w.f04 * (F04[k] - F04_sp)^2 +
                           w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
                           w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2
                           for k = 0:N))

optimize!(mpc)


# Extract control actions
F1 = (Vector(JuMP.value.(F1)))
F2 = (Vector(JuMP.value.(F2)))
F3 = (Vector(JuMP.value.(F3)))
F4 = (Vector(JuMP.value.(F4)))
F01 = (Vector(JuMP.value.(F01)))
F02 = (Vector(JuMP.value.(F02)))
F03 = (Vector(JuMP.value.(F03)))
F04 = (Vector(JuMP.value.(F04)))
Q1 = (Vector(JuMP.value.(Q1)))
Q2 = (Vector(JuMP.value.(Q2)))
Q3 = (Vector(JuMP.value.(Q3)))
Q4 = (Vector(JuMP.value.(Q4)))
Fr1 = (Vector(JuMP.value.(Fr1)))
Fr2 = (Vector(JuMP.value.(Fr2)))


# Initialize state vectors
CA1 = zeros(N)
CA2 = zeros(N)
CA3 = zeros(N)
CA4 = zeros(N)

T1 = zeros(N)
T2 = zeros(N)
T3 = zeros(N)
T4 = zeros(N)

V1 = zeros(N)
V2 = zeros(N)
V3 = zeros(N)
V4 = zeros(N)

CA1[1] = CA1_init
CA2[1] = CA2_init
CA3[1] = CA3_init
CA4[1] = CA4_init

T1[1] = T1_init
T2[1] = T2_init
T3[1] = T3_init
T4[1] = T4_init

V1[1] = V1_init
V2[1] = V2_init
V3[1] = V3_init
V4[1] = V4_init

for k = 1:length(T1)-1

    V1[k+1] = V1[k] + (F01[k] + Fr2[k] + Fr1[k] - F1[k]) * dt
    V2[k+1] = V2[k] + (F1[k] + F02[k] - F2[k]) * dt
    V3[k+1] = V3[k] + ((F2[k] - Fr1[k]) + F03[k] - F3[k]) * dt
    V4[k+1] = V4[k] + (F3[k] + F04[k] - F4[k]) * dt

    CA1[k+1] = CA1[k] + ((F01[k] / V1[k]) * (CA01 - CA1[k]) + (Fr1[k] / V1[k]) * (CA2[k] - CA1[k]) + (Fr2[k] / V1[k]) * (CA4[k] - CA1[k]) - (k10 * exp(-E1 / (R * T1[k])) * CA1[k] + k20 * exp(-E2 / (R * T1[k])) * CA1[k] + k30 * exp(-E3 / (R * T1[k])) * CA1[k])) * dt
    T1[k+1] = T1[k] + ((F01[k] / V1[k]) * (T01 - T1[k]) + (Fr1[k] / V1[k]) * (T2[k] - T1[k]) + (Fr2[k] / V1[k]) * (T4[k] - T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T1[k])) * CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T1[k])) * CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T1[k])) * CA1[k]) + Q1[k] / (rho * cp * V1[k])) * dt

    CA2[k+1] = CA2[k] + ((F1[k] / V2[k]) * (CA1[k] - CA2[k]) + (F02[k] / V2[k]) * (CA02 - CA2[k]) - (k10 * exp(-E1 / (R * T2[k])) * CA2[k] + k20 * exp(-E2 / (R * T2[k])) * CA2[k] + k30 * exp(-E3 / (R * T2[k])) * CA2[k])) * dt
    T2[k+1] = T2[k] + ((F1[k] / V2[k]) * (T1[k] - T2[k]) + (F02[k] / V2[k]) * (T02 - T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T2[k])) * CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T2[k])) * CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T2[k])) * CA2[k]) + Q2[k] / (rho * cp * V2[k])) * dt

    CA3[k+1] = CA3[k] + (((F2[k] - Fr1[k]) / V3[k]) * (CA2[k] - CA3[k]) + (F03[k] / V3[k]) * (CA03 - CA3[k]) - (k10 * exp(-E1 / (R * T3[k])) * CA3[k] + k20 * exp(-E2 / (R * T3[k])) * CA3[k] + k30 * exp(-E3 / (R * T3[k])) * CA3[k])) * dt
    T3[k+1] = T3[k] + (((F2[k] - Fr1[k]) / V3[k]) * (T2[k] - T3[k]) + (F03[k] / V3[k]) * (T03 - T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T3[k])) * CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T3[k])) * CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T3[k])) * CA3[k]) + Q3[k] / (rho * cp * V3[k])) * dt

    CA4[k+1] = CA4[k] + ((F3[k] / V4[k]) * (CA3[k] - CA4[k]) + (F04[k] / V4[k]) * (CA04 - CA4[k]) - (k10 * exp(-E1 / (R * T4[k])) * CA4[k] + k20 * exp(-E2 / (R * T4[k])) * CA4[k] + k30 * exp(-E3 / (R * T4[k])) * CA4[k])) * dt
    T4[k+1] = T4[k] + ((F3[k] / V4[k]) * (T3[k] - T4[k]) + (F04[k] / V4[k]) * (T04 - T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T4[k])) * CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T4[k])) * CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T4[k])) * CA4[k]) + Q4[k] / (rho * cp * V4[k])) * dt

end


ISE = sum(w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 +
          w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 +
          w.v1 * (V1[k] - V1_sp)^2 + w.v2 * (V2[k] - V2_sp)^2 + w.v3 * (V3[k] - V3_sp)^2 + w.v4 * (V4[k] - V4_sp)^2
          for k = 1:N)

ISC = sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
          w.f01 * (F01[k] - F01_sp)^2 + w.f02 * (F02[k] - F02_sp)^2 + w.f03 * (F03[k] - F03_sp)^2 + w.f04 * (F04[k] - F04_sp)^2 +
          w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
          w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2
          for k = 1:N-1)

PI = ISC + ISE

plot(T1)
hline!([T1_sp])