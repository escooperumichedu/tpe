using JuMP, Ipopt, CSV, DataFrames, Plots, LinearAlgebra

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

mutable struct ProcessSystemVariables
    CA1
    CA2
    CA3
    CA4
    T1
    T2
    T3
    T4
    V1
    V2
    V3
    V4
    F1
    F2
    F3
    F4
    F01
    F02
    F03
    F04
    Fr1
    Fr2
    Q1
    Q2
    Q3
    Q4
end

mutable struct ADMM_variables

    u1_1
    u1_2
    u1_3

    u2_1
    u2_2
    u2_3
    u2_4

    rho1_1
    rho1_2
    rho1_3

    rho2_1
    rho2_2
    rho2_3
    rho2_4

end

dt = 0.05  # hr (1min) - all dynamics are in terms of hours

CA1_init = 2.8
CA2_init = 2.75
CA3_init = 2.7
CA4_init = 2.8

T1_init = 340
T2_init = 350
T3_init = 345
T4_init = 360

V1_init = 9.0
V2_init = 1.5
V3_init = 3.5
V4_init = 10.0

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
tau = 0.35

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

N = 100

initial_rho = 0.5 # 0.5 is the best

# Structures for ADMM decomposition information
traj = ProcessSystemVariables(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))
prev_traj = ProcessSystemVariables(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))
copy = ProcessSystemVariables(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))
prev_copy = ProcessSystemVariables(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))

function gW(sp)
    wt = -Float64(floor(log10(abs(sp^2))))
    return 1 * 10^(wt)
end

# w = Weights(gW(CA1_sp), gW(CA2_sp), gW(CA3_sp), gW(CA4_sp), gW(T1_sp), gW(T2_sp), gW(T3_sp), gW(T4_sp), gW(V1_sp), gW(V2_sp), gW(V3_sp), gW(V4_sp), 1e0*gW(F1_sp), 1e0*gW(F2_sp), 1e0*gW(F3_sp), 1e0*gW(F4_sp), 1e0*gW(F01_sp), 1e0*gW(F02_sp), 1e0*gW(F03_sp), 1e0*gW(F04_sp), 1e0*gW(Fr1_sp), 1e0*gW(Fr2_sp), 1e0*gW(Q1_sp), 1e0*gW(Q2_sp), 1e0*gW(Q3_sp), 1e0*gW(Q4_sp))

w = let elements = [gW(CA1_sp), gW(CA2_sp), gW(CA3_sp), gW(CA4_sp),
        gW(T1_sp), gW(T2_sp), gW(T3_sp), gW(T4_sp),
        gW(V1_sp), gW(V2_sp), gW(V3_sp), gW(V4_sp),
        1e2 * gW(F1_sp), 1e2 * gW(F2_sp), 1e2 * gW(F3_sp), 1e2 * gW(F4_sp),
        1e2 * gW(F01_sp), 1e2 * gW(F02_sp), 1e2 * gW(F03_sp), 1e2 * gW(F04_sp),
        1e2 * gW(Fr1_sp), 1e2 * gW(Fr2_sp),
        1e1 * gW(Q1_sp), 1e1 * gW(Q2_sp), 1e1 * gW(Q3_sp), 1e1 * gW(Q4_sp)]
    max_val = maximum(elements)
    Weights((e / max_val for e in elements)...)
end

# Formulate a guess for trajectory and copy by dynamics at steady state
traj.F1 .= F1_sp
traj.F2 .= F2_sp
traj.F3 .= F3_sp
traj.F4 .= F4_sp

traj.F01 .= F01_sp
traj.F02 .= F02_sp
traj.F03 .= F03_sp
traj.F04 .= F04_sp

traj.Fr1 .= Fr1_sp
traj.Fr2 .= Fr2_sp

traj.Q1 .= Q1_sp
traj.Q2 .= Q2_sp
traj.Q3 .= Q3_sp
traj.Q4 .= Q4_sp

traj.CA1[1] = CA1_init
traj.CA2[1] = CA2_init
traj.CA3[1] = CA3_init
traj.CA4[1] = CA4_init

traj.T1[1] = T1_init
traj.T2[1] = T2_init
traj.T3[1] = T3_init
traj.T4[1] = T4_init

traj.V1[1] = V1_init
traj.V2[1] = V2_init
traj.V3[1] = V3_init
traj.V4[1] = V4_init

for k = 1:N

    traj.V1[k+1] = traj.V1[k] + (traj.F01[k] + traj.Fr2[k] + traj.Fr1[k] - traj.F1[k]) * dt
    traj.V2[k+1] = traj.V2[k] + (traj.F1[k] + traj.F02[k] - traj.F2[k]) * dt
    traj.V3[k+1] = traj.V3[k] + ((traj.F2[k] - traj.Fr1[k]) + traj.F03[k] - traj.F3[k]) * dt
    traj.V4[k+1] = traj.V4[k] + (traj.F3[k] + traj.F04[k] - traj.F4[k]) * dt

    traj.CA1[k+1] = traj.CA1[k] + ((traj.F01[k] / traj.V1[k]) * (CA01 - traj.CA1[k]) + (traj.Fr1[k] / traj.V1[k]) * (traj.CA2[k] - traj.CA1[k]) + (traj.Fr2[k] / traj.V1[k]) * (traj.CA4[k] - traj.CA1[k]) - (k10 * exp(-E1 / (R * traj.T1[k])) * traj.CA1[k] + k20 * exp(-E2 / (R * traj.T1[k])) * traj.CA1[k] + k30 * exp(-E3 / (R * traj.T1[k])) * traj.CA1[k])) * dt
    traj.T1[k+1] = traj.T1[k] + ((traj.F01[k] / traj.V1[k]) * (T01 - traj.T1[k]) + (traj.Fr1[k] / traj.V1[k]) * (traj.T2[k] - traj.T1[k]) + (traj.Fr2[k] / traj.V1[k]) * (traj.T4[k] - traj.T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * traj.T1[k])) * traj.CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * traj.T1[k])) * traj.CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * traj.T1[k])) * traj.CA1[k]) + traj.Q1[k] / (rho * cp * traj.V1[k])) * dt

    traj.CA2[k+1] = traj.CA2[k] + ((traj.F1[k] / traj.V2[k]) * (traj.CA1[k] - traj.CA2[k]) + (traj.F02[k] / traj.V2[k]) * (CA02 - traj.CA2[k]) - (k10 * exp(-E1 / (R * traj.T2[k])) * traj.CA2[k] + k20 * exp(-E2 / (R * traj.T2[k])) * traj.CA2[k] + k30 * exp(-E3 / (R * traj.T2[k])) * traj.CA2[k])) * dt
    traj.T2[k+1] = traj.T2[k] + ((traj.F1[k] / traj.V2[k]) * (traj.T1[k] - traj.T2[k]) + (traj.F02[k] / traj.V2[k]) * (T02 - traj.T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * traj.T2[k])) * traj.CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * traj.T2[k])) * traj.CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * traj.T2[k])) * traj.CA2[k]) + traj.Q2[k] / (rho * cp * traj.V2[k])) * dt

    traj.CA3[k+1] = traj.CA3[k] + (((traj.F2[k] - traj.Fr1[k]) / traj.V3[k]) * (traj.CA2[k] - traj.CA3[k]) + (traj.F03[k] / traj.V3[k]) * (CA03 - traj.CA3[k]) - (k10 * exp(-E1 / (R * traj.T3[k])) * traj.CA3[k] + k20 * exp(-E2 / (R * traj.T3[k])) * traj.CA3[k] + k30 * exp(-E3 / (R * traj.T3[k])) * traj.CA3[k])) * dt
    traj.T3[k+1] = traj.T3[k] + (((traj.F2[k] - traj.Fr1[k]) / traj.V3[k]) * (traj.T2[k] - traj.T3[k]) + (traj.F03[k] / traj.V3[k]) * (T03 - traj.T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * traj.T3[k])) * traj.CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * traj.T3[k])) * traj.CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * traj.T3[k])) * traj.CA3[k]) + traj.Q3[k] / (rho * cp * traj.V3[k])) * dt

    traj.CA4[k+1] = traj.CA4[k] + ((traj.F3[k] / traj.V4[k]) * (traj.CA3[k] - traj.CA4[k]) + (traj.F04[k] / traj.V4[k]) * (CA04 - traj.CA4[k]) - (k10 * exp(-E1 / (R * traj.T4[k])) * traj.CA4[k] + k20 * exp(-E2 / (R * traj.T4[k])) * traj.CA4[k] + k30 * exp(-E3 / (R * traj.T4[k])) * traj.CA4[k])) * dt
    traj.T4[k+1] = traj.T4[k] + ((traj.F3[k] / traj.V4[k]) * (traj.T3[k] - traj.T4[k]) + (traj.F04[k] / traj.V4[k]) * (T04 - traj.T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * traj.T4[k])) * traj.CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * traj.T4[k])) * traj.CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * traj.T4[k])) * traj.CA4[k]) + traj.Q4[k] / (rho * cp * traj.V4[k])) * dt

end

# Formulate a guess for trajectory and copy by dynamics at steady state
prev_traj.F1 .= F1_sp
prev_traj.F2 .= F2_sp
prev_traj.F3 .= F3_sp
prev_traj.F4 .= F4_sp

prev_traj.F01 .= F01_sp
prev_traj.F02 .= F02_sp
prev_traj.F03 .= F03_sp
prev_traj.F04 .= F04_sp

prev_traj.Fr1 .= Fr1_sp
prev_traj.Fr2 .= Fr2_sp

prev_traj.Q1 .= Q1_sp
prev_traj.Q2 .= Q2_sp
prev_traj.Q3 .= Q3_sp
prev_traj.Q4 .= Q4_sp

prev_traj.CA1[1] = CA1_init
prev_traj.CA2[1] = CA2_init
prev_traj.CA3[1] = CA3_init
prev_traj.CA4[1] = CA4_init

prev_traj.T1[1] = T1_init
prev_traj.T2[1] = T2_init
prev_traj.T3[1] = T3_init
prev_traj.T4[1] = T4_init

prev_traj.V1[1] = V1_init
prev_traj.V2[1] = V2_init
prev_traj.V3[1] = V3_init
prev_traj.V4[1] = V4_init

for k = 1:N

    prev_traj.V1[k+1] = prev_traj.V1[k] + (prev_traj.F01[k] + prev_traj.Fr2[k] + prev_traj.Fr1[k] - prev_traj.F1[k]) * dt
    prev_traj.V2[k+1] = prev_traj.V2[k] + (prev_traj.F1[k] + prev_traj.F02[k] - prev_traj.F2[k]) * dt
    prev_traj.V3[k+1] = prev_traj.V3[k] + ((prev_traj.F2[k] - prev_traj.Fr1[k]) + prev_traj.F03[k] - prev_traj.F3[k]) * dt
    prev_traj.V4[k+1] = prev_traj.V4[k] + (prev_traj.F3[k] + prev_traj.F04[k] - prev_traj.F4[k]) * dt

    prev_traj.CA1[k+1] = prev_traj.CA1[k] + ((prev_traj.F01[k] / prev_traj.V1[k]) * (CA01 - prev_traj.CA1[k]) + (prev_traj.Fr1[k] / prev_traj.V1[k]) * (prev_traj.CA2[k] - prev_traj.CA1[k]) + (prev_traj.Fr2[k] / prev_traj.V1[k]) * (prev_traj.CA4[k] - prev_traj.CA1[k]) - (k10 * exp(-E1 / (R * prev_traj.T1[k])) * prev_traj.CA1[k] + k20 * exp(-E2 / (R * prev_traj.T1[k])) * prev_traj.CA1[k] + k30 * exp(-E3 / (R * prev_traj.T1[k])) * prev_traj.CA1[k])) * dt
    prev_traj.T1[k+1] = prev_traj.T1[k] + ((prev_traj.F01[k] / prev_traj.V1[k]) * (T01 - prev_traj.T1[k]) + (prev_traj.Fr1[k] / prev_traj.V1[k]) * (prev_traj.T2[k] - prev_traj.T1[k]) + (prev_traj.Fr2[k] / prev_traj.V1[k]) * (prev_traj.T4[k] - prev_traj.T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_traj.T1[k])) * prev_traj.CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_traj.T1[k])) * prev_traj.CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_traj.T1[k])) * prev_traj.CA1[k]) + prev_traj.Q1[k] / (rho * cp * prev_traj.V1[k])) * dt

    prev_traj.CA2[k+1] = prev_traj.CA2[k] + ((prev_traj.F1[k] / prev_traj.V2[k]) * (prev_traj.CA1[k] - prev_traj.CA2[k]) + (prev_traj.F02[k] / prev_traj.V2[k]) * (CA02 - prev_traj.CA2[k]) - (k10 * exp(-E1 / (R * prev_traj.T2[k])) * prev_traj.CA2[k] + k20 * exp(-E2 / (R * prev_traj.T2[k])) * prev_traj.CA2[k] + k30 * exp(-E3 / (R * prev_traj.T2[k])) * prev_traj.CA2[k])) * dt
    prev_traj.T2[k+1] = prev_traj.T2[k] + ((prev_traj.F1[k] / prev_traj.V2[k]) * (prev_traj.T1[k] - prev_traj.T2[k]) + (prev_traj.F02[k] / prev_traj.V2[k]) * (T02 - prev_traj.T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_traj.T2[k])) * prev_traj.CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_traj.T2[k])) * prev_traj.CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_traj.T2[k])) * prev_traj.CA2[k]) + prev_traj.Q2[k] / (rho * cp * prev_traj.V2[k])) * dt

    prev_traj.CA3[k+1] = prev_traj.CA3[k] + (((prev_traj.F2[k] - prev_traj.Fr1[k]) / prev_traj.V3[k]) * (prev_traj.CA2[k] - prev_traj.CA3[k]) + (prev_traj.F03[k] / prev_traj.V3[k]) * (CA03 - prev_traj.CA3[k]) - (k10 * exp(-E1 / (R * prev_traj.T3[k])) * prev_traj.CA3[k] + k20 * exp(-E2 / (R * prev_traj.T3[k])) * prev_traj.CA3[k] + k30 * exp(-E3 / (R * prev_traj.T3[k])) * prev_traj.CA3[k])) * dt
    prev_traj.T3[k+1] = prev_traj.T3[k] + (((prev_traj.F2[k] - prev_traj.Fr1[k]) / prev_traj.V3[k]) * (prev_traj.T2[k] - prev_traj.T3[k]) + (prev_traj.F03[k] / prev_traj.V3[k]) * (T03 - prev_traj.T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_traj.T3[k])) * prev_traj.CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_traj.T3[k])) * prev_traj.CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_traj.T3[k])) * prev_traj.CA3[k]) + prev_traj.Q3[k] / (rho * cp * prev_traj.V3[k])) * dt

    prev_traj.CA4[k+1] = prev_traj.CA4[k] + ((prev_traj.F3[k] / prev_traj.V4[k]) * (prev_traj.CA3[k] - prev_traj.CA4[k]) + (prev_traj.F04[k] / prev_traj.V4[k]) * (CA04 - prev_traj.CA4[k]) - (k10 * exp(-E1 / (R * prev_traj.T4[k])) * prev_traj.CA4[k] + k20 * exp(-E2 / (R * prev_traj.T4[k])) * prev_traj.CA4[k] + k30 * exp(-E3 / (R * prev_traj.T4[k])) * prev_traj.CA4[k])) * dt
    prev_traj.T4[k+1] = prev_traj.T4[k] + ((prev_traj.F3[k] / prev_traj.V4[k]) * (prev_traj.T3[k] - prev_traj.T4[k]) + (prev_traj.F04[k] / prev_traj.V4[k]) * (T04 - prev_traj.T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_traj.T4[k])) * prev_traj.CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_traj.T4[k])) * prev_traj.CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_traj.T4[k])) * prev_traj.CA4[k]) + prev_traj.Q4[k] / (rho * cp * prev_traj.V4[k])) * dt

end

# Formulate a guess for copyectory and copy by dynamics at steady state
copy.F1 .= F1_sp
copy.F2 .= F2_sp
copy.F3 .= F3_sp
copy.F4 .= F4_sp

copy.F01 .= F01_sp
copy.F02 .= F02_sp
copy.F03 .= F03_sp
copy.F04 .= F04_sp

copy.Fr1 .= Fr1_sp
copy.Fr2 .= Fr2_sp

copy.Q1 .= Q1_sp
copy.Q2 .= Q2_sp
copy.Q3 .= Q3_sp
copy.Q4 .= Q4_sp

copy.CA1[1] = CA1_init
copy.CA2[1] = CA2_init
copy.CA3[1] = CA3_init
copy.CA4[1] = CA4_init

copy.T1[1] = T1_init
copy.T2[1] = T2_init
copy.T3[1] = T3_init
copy.T4[1] = T4_init

copy.V1[1] = V1_init
copy.V2[1] = V2_init
copy.V3[1] = V3_init
copy.V4[1] = V4_init

for k = 1:N

    copy.V1[k+1] = copy.V1[k] + (copy.F01[k] + copy.Fr2[k] + copy.Fr1[k] - copy.F1[k]) * dt
    copy.V2[k+1] = copy.V2[k] + (copy.F1[k] + copy.F02[k] - copy.F2[k]) * dt
    copy.V3[k+1] = copy.V3[k] + ((copy.F2[k] - copy.Fr1[k]) + copy.F03[k] - copy.F3[k]) * dt
    copy.V4[k+1] = copy.V4[k] + (copy.F3[k] + copy.F04[k] - copy.F4[k]) * dt

    copy.CA1[k+1] = copy.CA1[k] + ((copy.F01[k] / copy.V1[k]) * (CA01 - copy.CA1[k]) + (copy.Fr1[k] / copy.V1[k]) * (copy.CA2[k] - copy.CA1[k]) + (copy.Fr2[k] / copy.V1[k]) * (copy.CA4[k] - copy.CA1[k]) - (k10 * exp(-E1 / (R * copy.T1[k])) * copy.CA1[k] + k20 * exp(-E2 / (R * copy.T1[k])) * copy.CA1[k] + k30 * exp(-E3 / (R * copy.T1[k])) * copy.CA1[k])) * dt
    copy.T1[k+1] = copy.T1[k] + ((copy.F01[k] / copy.V1[k]) * (T01 - copy.T1[k]) + (copy.Fr1[k] / copy.V1[k]) * (copy.T2[k] - copy.T1[k]) + (copy.Fr2[k] / copy.V1[k]) * (copy.T4[k] - copy.T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * copy.T1[k])) * copy.CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * copy.T1[k])) * copy.CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * copy.T1[k])) * copy.CA1[k]) + copy.Q1[k] / (rho * cp * copy.V1[k])) * dt

    copy.CA2[k+1] = copy.CA2[k] + ((copy.F1[k] / copy.V2[k]) * (copy.CA1[k] - copy.CA2[k]) + (copy.F02[k] / copy.V2[k]) * (CA02 - copy.CA2[k]) - (k10 * exp(-E1 / (R * copy.T2[k])) * copy.CA2[k] + k20 * exp(-E2 / (R * copy.T2[k])) * copy.CA2[k] + k30 * exp(-E3 / (R * copy.T2[k])) * copy.CA2[k])) * dt
    copy.T2[k+1] = copy.T2[k] + ((copy.F1[k] / copy.V2[k]) * (copy.T1[k] - copy.T2[k]) + (copy.F02[k] / copy.V2[k]) * (T02 - copy.T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * copy.T2[k])) * copy.CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * copy.T2[k])) * copy.CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * copy.T2[k])) * copy.CA2[k]) + copy.Q2[k] / (rho * cp * copy.V2[k])) * dt

    copy.CA3[k+1] = copy.CA3[k] + (((copy.F2[k] - copy.Fr1[k]) / copy.V3[k]) * (copy.CA2[k] - copy.CA3[k]) + (copy.F03[k] / copy.V3[k]) * (CA03 - copy.CA3[k]) - (k10 * exp(-E1 / (R * copy.T3[k])) * copy.CA3[k] + k20 * exp(-E2 / (R * copy.T3[k])) * copy.CA3[k] + k30 * exp(-E3 / (R * copy.T3[k])) * copy.CA3[k])) * dt
    copy.T3[k+1] = copy.T3[k] + (((copy.F2[k] - copy.Fr1[k]) / copy.V3[k]) * (copy.T2[k] - copy.T3[k]) + (copy.F03[k] / copy.V3[k]) * (T03 - copy.T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * copy.T3[k])) * copy.CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * copy.T3[k])) * copy.CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * copy.T3[k])) * copy.CA3[k]) + copy.Q3[k] / (rho * cp * copy.V3[k])) * dt

    copy.CA4[k+1] = copy.CA4[k] + ((copy.F3[k] / copy.V4[k]) * (copy.CA3[k] - copy.CA4[k]) + (copy.F04[k] / copy.V4[k]) * (CA04 - copy.CA4[k]) - (k10 * exp(-E1 / (R * copy.T4[k])) * copy.CA4[k] + k20 * exp(-E2 / (R * copy.T4[k])) * copy.CA4[k] + k30 * exp(-E3 / (R * copy.T4[k])) * copy.CA4[k])) * dt
    copy.T4[k+1] = copy.T4[k] + ((copy.F3[k] / copy.V4[k]) * (copy.T3[k] - copy.T4[k]) + (copy.F04[k] / copy.V4[k]) * (T04 - copy.T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * copy.T4[k])) * copy.CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * copy.T4[k])) * copy.CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * copy.T4[k])) * copy.CA4[k]) + copy.Q4[k] / (rho * cp * copy.V4[k])) * dt

end

# Formulate a guess for copyectory and copy by dynamics at steady state
prev_copy.F1 .= F1_sp
prev_copy.F2 .= F2_sp
prev_copy.F3 .= F3_sp
prev_copy.F4 .= F4_sp

prev_copy.F01 .= F01_sp
prev_copy.F02 .= F02_sp
prev_copy.F03 .= F03_sp
prev_copy.F04 .= F04_sp

prev_copy.Fr1 .= Fr1_sp
prev_copy.Fr2 .= Fr2_sp

prev_copy.Q1 .= Q1_sp
prev_copy.Q2 .= Q2_sp
prev_copy.Q3 .= Q3_sp
prev_copy.Q4 .= Q4_sp

prev_copy.CA1[1] = CA1_init
prev_copy.CA2[1] = CA2_init
prev_copy.CA3[1] = CA3_init
prev_copy.CA4[1] = CA4_init

prev_copy.T1[1] = T1_init
prev_copy.T2[1] = T2_init
prev_copy.T3[1] = T3_init
prev_copy.T4[1] = T4_init

prev_copy.V1[1] = V1_init
prev_copy.V2[1] = V2_init
prev_copy.V3[1] = V3_init
prev_copy.V4[1] = V4_init

for k = 1:N

    prev_copy.V1[k+1] = prev_copy.V1[k] + (prev_copy.F01[k] + prev_copy.Fr2[k] + prev_copy.Fr1[k] - prev_copy.F1[k]) * dt
    prev_copy.V2[k+1] = prev_copy.V2[k] + (prev_copy.F1[k] + prev_copy.F02[k] - prev_copy.F2[k]) * dt
    prev_copy.V3[k+1] = prev_copy.V3[k] + ((prev_copy.F2[k] - prev_copy.Fr1[k]) + prev_copy.F03[k] - prev_copy.F3[k]) * dt
    prev_copy.V4[k+1] = prev_copy.V4[k] + (prev_copy.F3[k] + prev_copy.F04[k] - prev_copy.F4[k]) * dt

    prev_copy.CA1[k+1] = prev_copy.CA1[k] + ((prev_copy.F01[k] / prev_copy.V1[k]) * (CA01 - prev_copy.CA1[k]) + (prev_copy.Fr1[k] / prev_copy.V1[k]) * (prev_copy.CA2[k] - prev_copy.CA1[k]) + (prev_copy.Fr2[k] / prev_copy.V1[k]) * (prev_copy.CA4[k] - prev_copy.CA1[k]) - (k10 * exp(-E1 / (R * prev_copy.T1[k])) * prev_copy.CA1[k] + k20 * exp(-E2 / (R * prev_copy.T1[k])) * prev_copy.CA1[k] + k30 * exp(-E3 / (R * prev_copy.T1[k])) * prev_copy.CA1[k])) * dt
    prev_copy.T1[k+1] = prev_copy.T1[k] + ((prev_copy.F01[k] / prev_copy.V1[k]) * (T01 - prev_copy.T1[k]) + (prev_copy.Fr1[k] / prev_copy.V1[k]) * (prev_copy.T2[k] - prev_copy.T1[k]) + (prev_copy.Fr2[k] / prev_copy.V1[k]) * (prev_copy.T4[k] - prev_copy.T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_copy.T1[k])) * prev_copy.CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_copy.T1[k])) * prev_copy.CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_copy.T1[k])) * prev_copy.CA1[k]) + prev_copy.Q1[k] / (rho * cp * prev_copy.V1[k])) * dt

    prev_copy.CA2[k+1] = prev_copy.CA2[k] + ((prev_copy.F1[k] / prev_copy.V2[k]) * (prev_copy.CA1[k] - prev_copy.CA2[k]) + (prev_copy.F02[k] / prev_copy.V2[k]) * (CA02 - prev_copy.CA2[k]) - (k10 * exp(-E1 / (R * prev_copy.T2[k])) * prev_copy.CA2[k] + k20 * exp(-E2 / (R * prev_copy.T2[k])) * prev_copy.CA2[k] + k30 * exp(-E3 / (R * prev_copy.T2[k])) * prev_copy.CA2[k])) * dt
    prev_copy.T2[k+1] = prev_copy.T2[k] + ((prev_copy.F1[k] / prev_copy.V2[k]) * (prev_copy.T1[k] - prev_copy.T2[k]) + (prev_copy.F02[k] / prev_copy.V2[k]) * (T02 - prev_copy.T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_copy.T2[k])) * prev_copy.CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_copy.T2[k])) * prev_copy.CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_copy.T2[k])) * prev_copy.CA2[k]) + prev_copy.Q2[k] / (rho * cp * prev_copy.V2[k])) * dt

    prev_copy.CA3[k+1] = prev_copy.CA3[k] + (((prev_copy.F2[k] - prev_copy.Fr1[k]) / prev_copy.V3[k]) * (prev_copy.CA2[k] - prev_copy.CA3[k]) + (prev_copy.F03[k] / prev_copy.V3[k]) * (CA03 - prev_copy.CA3[k]) - (k10 * exp(-E1 / (R * prev_copy.T3[k])) * prev_copy.CA3[k] + k20 * exp(-E2 / (R * prev_copy.T3[k])) * prev_copy.CA3[k] + k30 * exp(-E3 / (R * prev_copy.T3[k])) * prev_copy.CA3[k])) * dt
    prev_copy.T3[k+1] = prev_copy.T3[k] + (((prev_copy.F2[k] - prev_copy.Fr1[k]) / prev_copy.V3[k]) * (prev_copy.T2[k] - prev_copy.T3[k]) + (prev_copy.F03[k] / prev_copy.V3[k]) * (T03 - prev_copy.T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_copy.T3[k])) * prev_copy.CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_copy.T3[k])) * prev_copy.CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_copy.T3[k])) * prev_copy.CA3[k]) + prev_copy.Q3[k] / (rho * cp * prev_copy.V3[k])) * dt

    prev_copy.CA4[k+1] = prev_copy.CA4[k] + ((prev_copy.F3[k] / prev_copy.V4[k]) * (prev_copy.CA3[k] - prev_copy.CA4[k]) + (prev_copy.F04[k] / prev_copy.V4[k]) * (CA04 - prev_copy.CA4[k]) - (k10 * exp(-E1 / (R * prev_copy.T4[k])) * prev_copy.CA4[k] + k20 * exp(-E2 / (R * prev_copy.T4[k])) * prev_copy.CA4[k] + k30 * exp(-E3 / (R * prev_copy.T4[k])) * prev_copy.CA4[k])) * dt
    prev_copy.T4[k+1] = prev_copy.T4[k] + ((prev_copy.F3[k] / prev_copy.V4[k]) * (prev_copy.T3[k] - prev_copy.T4[k]) + (prev_copy.F04[k] / prev_copy.V4[k]) * (T04 - prev_copy.T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * prev_copy.T4[k])) * prev_copy.CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * prev_copy.T4[k])) * prev_copy.CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * prev_copy.T4[k])) * prev_copy.CA4[k]) + prev_copy.Q4[k] / (rho * cp * prev_copy.V4[k])) * dt

end

admm = ADMM_variables(zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1), w.ca4 * initial_rho, w.t4 * initial_rho, w.fr2 * initial_rho, w.ca2 * initial_rho, w.t2 * initial_rho, w.f2 * initial_rho, w.fr1 * initial_rho)


function ADMM_1()

    admm_steps = 30

    r1_1 = zeros(admm_steps)
    r1_2 = zeros(admm_steps)
    r1_3 = zeros(admm_steps)

    r2_1 = zeros(admm_steps)
    r2_2 = zeros(admm_steps)
    r2_3 = zeros(admm_steps)
    r2_4 = zeros(admm_steps)

    s1_1 = zeros(admm_steps)
    s1_2 = zeros(admm_steps)
    s1_3 = zeros(admm_steps)

    s2_1 = zeros(admm_steps)
    s2_2 = zeros(admm_steps)
    s2_3 = zeros(admm_steps)
    s2_4 = zeros(admm_steps)

    primal_residual = zeros(admm_steps)
    dual_residual = zeros(admm_steps)

    function sp1()

        mpc = JuMP.Model(Ipopt.Optimizer)

        JuMP.@variables mpc begin

            T1[k=0:N], (lower_bound=200, upper_bound=400)
            T2[k=0:N], (lower_bound=200, upper_bound=400)

            T4[k=0:N], (lower_bound=200, upper_bound=400) # copy from sp2

            V1[k=0:N], (lower_bound=0.3, upper_bound=10)
            V2[k=0:N], (lower_bound=0.3, upper_bound=10)


            CA1[k=0:N], (lower_bound=1e-5, upper_bound=10)
            CA2[k=0:N], (lower_bound=1e-5, upper_bound=10)

            CA4[k=0:N], (lower_bound=1e-5, upper_bound=10) # copy from sp2

            Q1[k=0:N], (lower_bound=Q1_sp * 0.5, upper_bound=Q1_sp * 1.5)
            Q2[k=0:N], (lower_bound=Q2_sp * 0.5, upper_bound=Q2_sp * 1.5)

            F1[k=0:N], (lower_bound=1, upper_bound=60)
            F2[k=0:N], (lower_bound=1, upper_bound=60)

            F01[k=0:N], (lower_bound=1, upper_bound=60)
            F02[k=0:N], (lower_bound=1, upper_bound=60)

            Fr1[k=0:N], (lower_bound=1, upper_bound=60)
            Fr2[k=0:N], (lower_bound=1, upper_bound=60) # copy from sp2


        end

        @constraints mpc begin

            T1_initial_condition, T1[0] == T1_init
            T2_initial_condition, T2[0] == T2_init

            T4_initial_condition, T4[0] == T4_init

            CA1_initial_condition, CA1[0] == CA1_init
            CA2_initial_condition, CA2[0] == CA2_init

            CA4_initial_condition, CA4[0] == CA4_init

            V1_initial_condition, V1[0] == V1_init
            V2_initial_condition, V2[0] == V2_init

            recycle_limit_01[k=0:N], Fr1[k] <= epsilon * F2[k]

        end

        @NLconstraints mpc begin


            dV1_dt[k=0:N-1], V1[k+1] == V1[k] + (F01[k] + Fr2[k] + Fr1[k] - F1[k]) * dt
            dV2_dt[k=0:N-1], V2[k+1] == V2[k] + (F1[k] + F02[k] - F2[k]) * dt

            holdUp1[k=0:N], F01[k] + Fr2[k] + Fr1[k] - F1[k] == -(V1[k] - V1_sp) / tau
            holdUp2[k=0:N], F1[k] + F02[k] - F2[k] == -(V2[k] - V2_sp) / tau

            dCA1_dt[k=0:N-1], CA1[k+1] == CA1[k] + ((F01[k] / V1[k]) * (CA01 - CA1[k]) + (Fr1[k] / V1[k]) * (CA2[k] - CA1[k]) + (Fr2[k] / V1[k]) * (CA4[k] - CA1[k]) - (k10 * exp(-E1 / (R * T1[k])) * CA1[k] + k20 * exp(-E2 / (R * T1[k])) * CA1[k] + k30 * exp(-E3 / (R * T1[k])) * CA1[k])) * dt
            dT1_dt[k=0:N-1], T1[k+1] == T1[k] + ((F01[k] / V1[k]) * (T01 - T1[k]) + (Fr1[k] / V1[k]) * (T2[k] - T1[k]) + (Fr2[k] / V1[k]) * (T4[k] - T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T1[k])) * CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T1[k])) * CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T1[k])) * CA1[k]) + Q1[k] / (rho * cp * V1[k])) * dt

            dCA2_dt[k=0:N-1], CA2[k+1] == CA2[k] + ((F1[k] / V2[k]) * (CA1[k] - CA2[k]) + (F02[k] / V2[k]) * (CA02 - CA2[k]) - (k10 * exp(-E1 / (R * T2[k])) * CA2[k] + k20 * exp(-E2 / (R * T2[k])) * CA2[k] + k30 * exp(-E3 / (R * T2[k])) * CA2[k])) * dt
            dT2_dt[k=0:N-1], T2[k+1] == T2[k] + ((F1[k] / V2[k]) * (T1[k] - T2[k]) + (F02[k] / V2[k]) * (T02 - T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T2[k])) * CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T2[k])) * CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T2[k])) * CA2[k]) + Q2[k] / (rho * cp * V2[k])) * dt

        end


        @NLobjective(mpc, Min, sum(w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 +
                                   w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 +
                                   w.v1 * (V1[k] - V1_sp)^2 + w.v2 * (V2[k] - V2_sp)^2 +
                                   w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 +
                                   w.f01 * (F01[k] - F01_sp)^2 + w.f02 * (F02[k] - F02_sp)^2 +
                                   w.fr1 * (Fr1[k] - Fr1_sp)^2 +
                                   w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2
                                   for k = 0:N)
                               # Add in copy variable lagrangians and quadratic penalty GIVEN to other subproblems
                               +
                               sum(admm.u2_1[k+1] * CA2[k] + (admm.rho2_1 / 2) * (CA2[k] - prev_copy.CA2[k+1])^2 for k = 0:N)
                               +
                               sum(admm.u2_2[k+1] * T2[k] + (admm.rho2_2 / 2) * (T2[k] - prev_copy.T2[k+1])^2 for k = 0:N)
                               +
                               sum(admm.u2_3[k+1] * F2[k] + (admm.rho2_3 / 2) * (F2[k] - prev_copy.F2[k+1])^2 for k = 0:N)
                               +
                               sum(admm.u2_4[k+1] * Fr1[k] + (admm.rho2_4 / 2) * (Fr1[k] - prev_copy.Fr1[k+1])^2 for k = 0:N)

                               # Add in copy variable lagrangians and quadratic penalty FROM other subproblems
                               +
                               sum(-1 * admm.u1_1[k+1] * CA4[k] + (admm.rho1_1 / 2) * (prev_traj.CA4[k+1] - CA4[k])^2 for k = 0:N)
                               +
                               sum(-1 * admm.u1_2[k+1] * T4[k] + (admm.rho1_2 / 2) * (prev_traj.T4[k+1] - T4[k])^2 for k = 0:N)
                               +
                               sum(-1 * admm.u1_3[k+1] * Fr2[k] + (admm.rho1_3 / 2) * (prev_traj.Fr2[k+1] - Fr2[k])^2 for k = 0:N)
        )
        set_silent(mpc)
        optimize!(mpc)

        # Extract control actions
        F1 = (Vector(JuMP.value.(F1)))
        F2 = (Vector(JuMP.value.(F2)))

        F01 = (Vector(JuMP.value.(F01)))
        F02 = (Vector(JuMP.value.(F02)))

        Fr1 = (Vector(JuMP.value.(Fr1)))

        Q1 = (Vector(JuMP.value.(Q1)))
        Q2 = (Vector(JuMP.value.(Q2)))

        T1 = (Vector(JuMP.value.(T1)))
        T2 = (Vector(JuMP.value.(T2)))

        V1 = (Vector(JuMP.value.(V1)))
        V2 = (Vector(JuMP.value.(V2)))

        CA1 = (Vector(JuMP.value.(CA1)))
        CA2 = (Vector(JuMP.value.(CA2)))

        CA4 = (Vector(JuMP.value.(CA4)))
        T4 = (Vector(JuMP.value.(T4)))
        Fr2 = (Vector(JuMP.value.(Fr2)))

        return F1, F2, F01, F02, Fr1, Q1, Q2, T1, T2, V1, V2, CA1, CA2, CA4, T4, Fr2
    end

    function sp2()

        mpc = JuMP.Model(Ipopt.Optimizer)

        JuMP.@variables mpc begin


            T2[k=0:N], (lower_bound=200, upper_bound=400) # copy from sp1
            T3[k=0:N], (lower_bound=200, upper_bound=400)
            T4[k=0:N], (lower_bound=200, upper_bound=400)

            V3[k=0:N], (lower_bound=0.3, upper_bound=10)
            V4[k=0:N], (lower_bound=0.3, upper_bound=10)


            CA2[k=0:N], (lower_bound=1e-5, upper_bound=10) # copy from sp1
            CA3[k=0:N], (lower_bound=1e-5, upper_bound=10)
            CA4[k=0:N], (lower_bound=1e-5, upper_bound=10)

            Q3[k=0:N], (lower_bound=Q3_sp * 0.5, upper_bound=Q3_sp * 1.5)
            Q4[k=0:N], (lower_bound=Q4_sp * 0.5, upper_bound=Q4_sp * 1.5)

            F2[k=0:N], (lower_bound=1, upper_bound=60)  # copy from sp1
            F3[k=0:N], (lower_bound=1, upper_bound=60)
            F4[k=0:N], (lower_bound=1, upper_bound=60)


            F03[k=0:N], (lower_bound=1, upper_bound=60)
            F04[k=0:N], (lower_bound=1, upper_bound=60)

            Fr1[k=0:N], (lower_bound=1, upper_bound=60)  # copy from sp1
            Fr2[k=0:N], (lower_bound=1, upper_bound=60)


        end

        @constraints mpc begin


            T2_initial_condition, T2[0] == T2_init
            T3_initial_condition, T3[0] == T3_init
            T4_initial_condition, T4[0] == T4_init


            CA2_initial_condition, CA2[0] == CA2_init
            CA3_initial_condition, CA3[0] == CA3_init
            CA4_initial_condition, CA4[0] == CA4_init


            V3_initial_condition, V3[0] == V3_init
            V4_initial_condition, V4[0] == V4_init

            recycle_limit_02[k=0:N], Fr2[k] <= epsilon * F4[k]

        end

        @NLconstraints mpc begin

            dV3_dt[k=0:N-1], V3[k+1] == V3[k] + ((F2[k] - Fr1[k]) + F03[k] - F3[k]) * dt
            dV4_dt[k=0:N-1], V4[k+1] == V4[k] + (F3[k] + F04[k] - F4[k]) * dt

            holdUp3[k=0:N], (F2[k] - Fr1[k]) + F03[k] - F3[k] == -(V3[k] - V3_sp) / tau
            holdUp4[k=0:N], F3[k] + F04[k] - F4[k] == -(V4[k] - V4_sp) / tau

            dCA3_dt[k=0:N-1], CA3[k+1] == CA3[k] + (((F2[k] - Fr1[k]) / V3[k]) * (CA2[k] - CA3[k]) + (F03[k] / V3[k]) * (CA03 - CA3[k]) - (k10 * exp(-E1 / (R * T3[k])) * CA3[k] + k20 * exp(-E2 / (R * T3[k])) * CA3[k] + k30 * exp(-E3 / (R * T3[k])) * CA3[k])) * dt
            dT3_dt[k=0:N-1], T3[k+1] == T3[k] + (((F2[k] - Fr1[k]) / V3[k]) * (T2[k] - T3[k]) + (F03[k] / V3[k]) * (T03 - T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T3[k])) * CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T3[k])) * CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T3[k])) * CA3[k]) + Q3[k] / (rho * cp * V3[k])) * dt

            dCA4_dt[k=0:N-1], CA4[k+1] == CA4[k] + ((F3[k] / V4[k]) * (CA3[k] - CA4[k]) + (F04[k] / V4[k]) * (CA04 - CA4[k]) - (k10 * exp(-E1 / (R * T4[k])) * CA4[k] + k20 * exp(-E2 / (R * T4[k])) * CA4[k] + k30 * exp(-E3 / (R * T4[k])) * CA4[k])) * dt
            dT4_dt[k=0:N-1], T4[k+1] == T4[k] + ((F3[k] / V4[k]) * (T3[k] - T4[k]) + (F04[k] / V4[k]) * (T04 - T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T4[k])) * CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T4[k])) * CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T4[k])) * CA4[k]) + Q4[k] / (rho * cp * V4[k])) * dt

        end


        @NLobjective(
            mpc,
            Min,
            sum(w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 +
                w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 +
                w.v3 * (V3[k] - V3_sp)^2 + w.v4 * (V4[k] - V4_sp)^2 +
                w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
                w.f03 * (F03[k] - F03_sp)^2 + w.f04 * (F04[k] - F04_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
                w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2
                for k = 0:N)
            # Add in copy variable lagrangians and quadratic penalty GIVEN to other subproblems
            +
            sum(admm.u1_1[k+1] * CA4[k] + (admm.rho1_1 / 2) * (CA4[k] - prev_copy.CA4[k+1])^2 for k = 0:N)
            +
            sum(admm.u1_2[k+1] * T4[k] + (admm.rho1_2 / 2) * (T4[k] - prev_copy.T4[k+1])^2 for k = 0:N)
            +
            sum(admm.u1_3[k+1] * Fr2[k] + (admm.rho1_3 / 2) * (Fr2[k] - prev_copy.Fr2[k+1])^2 for k = 0:N)
            # Add in copy variable lagrangians and quadratic penalty FROM other subproblems
            +
            sum(-1 * admm.u2_1[k+1] * CA2[k] + (admm.rho2_1 / 2) * (prev_traj.CA2[k+1] - CA2[k])^2 for k = 0:N)
            +
            sum(-1 * admm.u2_2[k+1] * T2[k] + (admm.rho2_2 / 2) * (prev_traj.T2[k+1] - T2[k])^2 for k = 0:N)
            +
            sum(-1 * admm.u2_3[k+1] * F2[k] + (admm.rho2_3 / 2) * (prev_traj.F2[k+1] - F2[k])^2 for k = 0:N)
            +
            sum(-1 * admm.u2_4[k+1] * Fr1[k] + (admm.rho2_4 / 2) * (prev_traj.Fr1[k+1] - Fr1[k])^2 for k = 0:N)
        )

        set_silent(mpc)

        optimize!(mpc)

        # Extract control actions
        F3 = (Vector(JuMP.value.(F3)))
        F4 = (Vector(JuMP.value.(F4)))

        F03 = (Vector(JuMP.value.(F03)))
        F04 = (Vector(JuMP.value.(F04)))

        Fr2 = (Vector(JuMP.value.(Fr2)))

        Q3 = (Vector(JuMP.value.(Q3)))
        Q4 = (Vector(JuMP.value.(Q4)))

        T3 = (Vector(JuMP.value.(T3)))
        T4 = (Vector(JuMP.value.(T4)))

        V3 = (Vector(JuMP.value.(V3)))
        V4 = (Vector(JuMP.value.(V4)))

        CA3 = (Vector(JuMP.value.(CA3)))
        CA4 = (Vector(JuMP.value.(CA4)))

        CA2 = (Vector(JuMP.value.(CA2)))
        T2 = (Vector(JuMP.value.(T2)))
        F2 = (Vector(JuMP.value.(F2)))
        Fr1 = (Vector(JuMP.value.(Fr1)))

        return F3, F4, F03, F04, Fr2, Q3, Q4, T3, T4, V3, V4, CA3, CA4, CA2, T2, F2, Fr1
    end

    for k = 1:admm_steps
        # Solve subproblems
        traj.F1, traj.F2, traj.F01, traj.F02, traj.Fr1, traj.Q1, traj.Q2, traj.T1, traj.T2, traj.V1, traj.V2, traj.CA1, traj.CA2, copy.CA4, copy.T4, copy.Fr2 = sp1()
        traj.F3, traj.F4, traj.F03, traj.F04, traj.Fr2, traj.Q3, traj.Q4, traj.T3, traj.T4, traj.V3, traj.V4, traj.CA3, traj.CA4, copy.CA2, copy.T2, copy.F2, copy.Fr1 = sp2()

        # Update dual variables
        for j = 1:N+1

            admm.u1_1[j] = admm.u1_1[j] + admm.rho1_1 * (traj.CA4[j] - copy.CA4[j])
            admm.u1_2[j] = admm.u1_2[j] + admm.rho1_2 * (traj.T4[j] - copy.T4[j])
            admm.u1_3[j] = admm.u1_3[j] + admm.rho1_3 * (traj.Fr2[j] - copy.Fr2[j])

            admm.u2_1[j] = admm.u2_1[j] + admm.rho2_1 * (traj.CA2[j] - copy.CA2[j])
            admm.u2_2[j] = admm.u2_2[j] + admm.rho2_2 * (traj.T2[j] - copy.T2[j])
            admm.u2_3[j] = admm.u2_3[j] + admm.rho2_3 * (traj.F2[j] - copy.F2[j])
            admm.u2_4[j] = admm.u2_4[j] + admm.rho2_4 * (traj.Fr1[j] - copy.Fr1[j])

        end

        r1_1[k] = norm(traj.CA4 - prev_traj.CA4, 2)
        r1_2[k] = norm(traj.T4 - prev_traj.T4, 2)
        r1_3[k] = norm(traj.Fr2 - prev_traj.Fr2, 2)

        r2_1[k] = norm(traj.CA2 - prev_traj.CA2, 2)
        r2_2[k] = norm(traj.T2 - prev_traj.T2, 2)
        r2_3[k] = norm(traj.F2 - prev_traj.F2, 2)
        r2_4[k] = norm(traj.Fr1 - prev_traj.Fr1, 2)

        
        s1_1[k] = norm(copy.CA4 - prev_copy.CA4, 2)
        s1_2[k] = norm(copy.T4 - prev_copy.T4, 2)
        s1_3[k] = norm(copy.Fr2 - prev_copy.Fr2, 2)

        s2_1[k] = norm(copy.CA2 - prev_copy.CA2, 2)
        s2_2[k] = norm(copy.T2 - prev_copy.T2, 2)
        s2_3[k] = norm(copy.F2 - prev_copy.F2, 2)
        s2_4[k] = norm(copy.Fr1 - prev_copy.Fr1, 2)

        primal_residual[k] = r1_1[k] + r1_2[k] + r1_3[k] + r2_1[k] + r2_2[k] + r2_3[k] + r2_4[k]
        dual_residual[k] = s1_1[k] + s1_2[k] + s1_3[k] + s2_1[k] + s2_2[k] + s2_3[k] + s2_4[k]


        prev_traj.CA4 = traj.CA4 
        prev_traj.T4 = traj.T4 
        prev_traj.Fr2 = traj.Fr2

        prev_traj.CA2 = traj.CA2 
        prev_traj.T2 = traj.T2 
        prev_traj.F2 = traj.F2
        prev_traj.Fr1 = traj.Fr1

        prev_copy.CA4 = copy.CA4 
        prev_copy.T4 = copy.T4 
        prev_copy.Fr2 = copy.Fr2

        prev_copy.CA2 = copy.CA2 
        prev_copy.T2 = copy.T2 
        prev_copy.F2 = copy.F2
        prev_copy.Fr1 = copy.Fr1
    end

    F1 = traj.F1
    F2 = traj.F2
    F3 = traj.F3
    F4 = traj.F4
    F01 = traj.F01
    F02 = traj.F02
    F03 = traj.F03
    F04 = traj.F04
    Q1 = traj.Q1
    Q2 = traj.Q2
    Q3 = traj.Q3
    Q4 = traj.Q4
    Fr1 = traj.Fr1
    Fr2 = traj.Fr2

    return F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2, r1_1, r1_2, r1_3, r2_1, r2_2, r2_3, r2_4, s1_1, s1_2, s1_3, s2_1, s2_2, s2_3, s2_4, primal_residual, dual_residual
end

function CMPC()

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

    set_silent(mpc)
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

    return F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2

end

function getTraj(F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2)

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

    return CA1, CA2, CA3, CA4, T1, T2, T3, T4, V1, V2, V3, V4

end

function getPI(F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2, CA1, CA2, CA3, CA4, T1, T2, T3, T4, V1, V2, V3, V4)

    ISE = sum(w.ca1 * (CA1[k] - CA1_sp)^2 + w.ca2 * (CA2[k] - CA2_sp)^2 + w.ca3 * (CA3[k] - CA3_sp)^2 + w.ca4 * (CA4[k] - CA4_sp)^2 +
              w.t1 * (T1[k] - T1_sp)^2 + w.t2 * (T2[k] - T2_sp)^2 + w.t3 * (T3[k] - T3_sp)^2 + w.t4 * (T4[k] - T4_sp)^2 +
              w.v1 * (V1[k] - V1_sp)^2 + w.v2 * (V2[k] - V2_sp)^2 + w.v3 * (V3[k] - V3_sp)^2 + w.v4 * (V4[k] - V4_sp)^2
              for k = 1:N) * 1E3

    ISC = sum(w.f1 * (F1[k] - F1_sp)^2 + w.f2 * (F2[k] - F2_sp)^2 + w.f3 * (F3[k] - F3_sp)^2 + w.f4 * (F4[k] - F4_sp)^2 +
              w.f01 * (F01[k] - F01_sp)^2 + w.f02 * (F02[k] - F02_sp)^2 + w.f03 * (F03[k] - F03_sp)^2 + w.f04 * (F04[k] - F04_sp)^2 +
              w.fr1 * (Fr1[k] - Fr1_sp)^2 + w.fr2 * (Fr2[k] - Fr2_sp)^2 +
              w.q1 * (Q1[k] - Q1_sp)^2 + w.q2 * (Q2[k] - Q2_sp)^2 + w.q3 * (Q3[k] - Q3_sp)^2 + w.q4 * (Q4[k] - Q4_sp)^2
              for k = 1:N-1) * 1E3

    PI = ISE + ISC

    return ISE, ISC, PI
end

# F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2 = CMPC()
F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2, r1_1, r1_2, r1_3, r2_1, r2_2, r2_3, r2_4, s1_1, s1_2, s1_3, s2_1, s2_2, s2_3, s2_4, primal_residual, dual_residual = ADMM_1()

CA1, CA2, CA3, CA4, T1, T2, T3, T4, V1, V2, V3, V4 = getTraj(F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2)
ISE, ISC, PI = getPI(F1, F2, F3, F4, F01, F02, F03, F04, Q1, Q2, Q3, Q4, Fr1, Fr2, CA1, CA2, CA3, CA4, T1, T2, T3, T4, V1, V2, V3, V4)
println(PI)
p1 = plot(primal_residual, label = "Primal norm")
p2 = plot(dual_residual, label = "Dual norm")
plot(p1, p2)

