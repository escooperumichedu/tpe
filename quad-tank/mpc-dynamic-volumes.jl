using JuMP, Ipopt, CSV, DataFrames

CA1_init = 2.8
CA2_init = 2.75
CA3_init = 2.7
CA4_init = 2.8

T1_init = 340
T2_init = 350
T3_init = 345
T4_init = 360

V1_sp = 1.5
V2_sp = 3.5
V3_sp = 3.5
V4_sp = 4.0

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

T01 = 300
T02 = 300

R = 8.314

T1_sp = 312.821
T2_sp = 312.671
T3_sp = 315.135
T4_sp = 314.314
CA1_sp = 3.02979
CA2_sp = 2.79845
CA3_sp = 2.84194
CA4_sp = 3.01147

Q1_sp = 1E4
Q2_sp = 2E4
Q3_sp = 2.5E4
Q4_sp = 1E4

N = 200

mpc = JuMP.Model(Ipopt.Optimizer)


JuMP.@variables mpc begin

    T1[k=0:N], (lower_bound = 200, upper_bound = 400)
    T2[k=0:N], (lower_bound = 200, upper_bound = 400)
    T3[k=0:N], (lower_bound = 200, upper_bound = 400)
    T4[k=0:N], (lower_bound = 200, upper_bound = 400)

    V1[k=0:N], (lower_bound = 200, upper_bound = 400)
    V2[k=0:N], (lower_bound = 200, upper_bound = 400)
    V3[k=0:N], (lower_bound = 200, upper_bound = 400)
    V4[k=0:N], (lower_bound = 200, upper_bound = 400)

    CA1[k=0:N], (lower_bound = 1e-5, upper_bound = 10)
    CA2[k=0:N], (lower_bound = 1e-5, upper_bound = 10)
    CA3[k=0:N], (lower_bound = 1e-5, upper_bound = 10)
    CA4[k=0:N], (lower_bound = 1e-5, upper_bound = 10)
    
    Q1[k=0:N], (lower_bound = Q1_sp*0.8, upper_bound = Q1_sp*1.2)
    Q2[k=0:N], (lower_bound = Q2_sp*0.8, upper_bound = Q2_sp*1.2)
    Q3[k=0:N], (lower_bound = Q3_sp*0.8, upper_bound = Q3_sp*1.2)
    Q4[k=0:N], (lower_bound = Q4_sp*0.8, upper_bound = Q4_sp*1.2)

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

end

@NLconstraints mpc begin

    dCA1_dt[k=0:N-1], CA1[k+1] == CA1[k] + ((F01 / V1) * (CA01 - CA1[k]) + (Fr1 / V1) * (CA2[k] - CA1[k]) + (Fr2 / V1) * (CA4[k] - CA1[k]) - (k10 * exp(-E1 / (R * T1[k])) * CA1[k] + k20 * exp(-E2 / (R * T1[k])) * CA1[k] + k30 * exp(-E3 / (R * T1[k])) * CA1[k])) * dt
    dT1_dt[k=0:N-1], T1[k+1] == T1[k] + ((F01 / V1) * (T01 - T1[k]) + (Fr1 / V1) * (T2[k] - T1[k]) + (Fr2 / V1) * (T4[k] - T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T1[k])) * CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T1[k])) * CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T1[k])) * CA1[k]) + Q1[k]/(rho*cp*V1))*dt

    dCA2_dt[k=0:N-1], CA2[k+1] == CA2[k] + ((F1 / V2) * (CA1[k] - CA2[k]) + (F02 / V2) * (CA02 - CA2[k]) - (k10 * exp(-E1 / (R * T2[k])) * CA2[k] + k20 * exp(-E2 / (R * T2[k])) * CA2[k] + k30 * exp(-E3 / (R * T2[k])) * CA2[k])) * dt
    dT2_dt[k=0:N-1], T2[k+1] == T2[k] + ((F1 / V1) * (T1[k] - T2[k]) + (F02 / V2) * (T02 - T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T2[k])) * CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T2[k])) * CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T2[k])) * CA2[k]) + Q2[k]/(rho*cp*V2))*dt

    dCA3_dt[k=0:N-1], CA3[k+1] == CA3[k] + (((F2 - Fr1) / V3) * (CA2[k] - CA3[k]) + (F03 / V3) * (CA03 - CA3[k]) - (k10 * exp(-E1 / (R * T3[k])) * CA3[k] + k20 * exp(-E2 / (R * T3[k])) * CA3[k] + k30 * exp(-E3 / (R * T3[k])) * CA3[k])) * dt
    dT3_dt[k=0:N-1], T3[k+1] == T3[k] + (((F2 - Fr1) / V3) * (T2[k] - T3[k]) + (F03 / V3) * (T03 - T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T3[k])) * CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T3[k])) * CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T3[k])) * CA3[k]) + Q3[k]/(rho*cp*V2))*dt

    dCA4_dt[k=0:N-1], CA4[k+1] == CA4[k] + ((F3 / V4) * (CA3[k] - CA4[k]) + (F04 / V4) * (CA04 - CA4[k]) - (k10 * exp(-E1 / (R * T4[k])) * CA4[k] + k20 * exp(-E2 / (R * T4[k])) * CA4[k] + k30 * exp(-E3 / (R * T4[k])) * CA4[k])) * dt
    dT4_dt[k=0:N-1], T4[k+1] == T4[k] + ((F3 / V4) * (T3[k] - T4[k]) + (F04 / V4) * (T04 - T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T4[k])) * CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T4[k])) * CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T4[k])) * CA4[k]) + Q4[k]/(rho*cp*V2))*dt

end


@NLobjective(mpc, Min, sum((CA1[k]-CA1_sp)^2 + (CA2[k]-CA2_sp)^2 + (CA3[k]-CA3_sp)^2 + (CA4[k]-CA4_sp)^2 +
(T1[k]-T1_sp)^2 + (T2[k]-T2_sp)^2 + (T3[k]-T3_sp)^2 + (T4[k]-T4_sp)^2 +
(Q1[k]-Q1_sp)^2 + (Q2[k]-Q2_sp)^2 + (Q3[k]-Q3_sp)^2 + (Q4[k]-Q4_sp)^2   
for k = 0:N))

optimize!(mpc)



