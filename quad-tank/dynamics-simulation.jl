using Plots
dt = 0.0166667 # hr (1min) - all dynamics are in terms of hours

CA1_init = 2.8
CA2_init = 2.75
CA3_init = 2.7
CA4_init = 2.8

T1_init = 340
T2_init = 350
T3_init = 345
T4_init = 360

V1 = 1.0
V2 = 3.0
V3 = 4.0
V4 = 6.0

delH1 = -5.0E4
delH2 = -5.2E4
delH3 = -5.0E4

k10 = 3.0E6
k20 = 3.0E5
k30 = 3.0E5

E1 = 5.0E4
E2 = 7.53E4
E3 = 7.53E4


F1 = 35
F2 = 45
F3 = 33

F01 = 5
F02 = 10
F03 = 8
F04 = 12

Fr1 = 20
Fr2 = 10

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

Q1 = 1E4
Q2 = 2E4
Q3 = 2.5E4
Q4 = 1E4

CA1 = zeros(200)
CA2 = zeros(200)
CA3 = zeros(200)
CA4 = zeros(200)

T1 = zeros(200)
T2 = zeros(200)
T3 = zeros(200)
T4 = zeros(200)

CA1[1] = CA1_init
CA2[1] = CA2_init
CA3[1] = CA3_init
CA4[1] = CA4_init

T1[1] = T1_init
T2[1] = T2_init
T3[1] = T3_init
T4[1] = T4_init

for k = 1:(length(CA1)-1)
    CA1[k+1] = CA1[k] + ((F01 / V1) * (CA01 - CA1[k]) + (Fr1 / V1) * (CA2[k] - CA1[k]) + (Fr2 / V1) * (CA4[k] - CA1[k]) - (k10 * exp(-E1 / (R * T1[k])) * CA1[k] + k20 * exp(-E2 / (R * T1[k])) * CA1[k] + k30 * exp(-E3 / (R * T1[k])) * CA1[k])) * dt
    T1[k+1] = T1[k] + ((F01 / V1) * (T01 - T1[k]) + (Fr1 / V1) * (T2[k] - T1[k]) + (Fr2 / V1) * (T4[k] - T1[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T1[k])) * CA1[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T1[k])) * CA1[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T1[k])) * CA1[k]) + Q1/(rho*cp*V1))*dt

    CA2[k+1] = CA2[k] + ((F1 / V2) * (CA1[k] - CA2[k]) + (F02 / V2) * (CA02 - CA2[k]) - (k10 * exp(-E1 / (R * T2[k])) * CA2[k] + k20 * exp(-E2 / (R * T2[k])) * CA2[k] + k30 * exp(-E3 / (R * T2[k])) * CA2[k])) * dt
    T2[k+1] = T2[k] + ((F1 / V2) * (T1[k] - T2[k]) + (F02 / V2) * (T02 - T2[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T2[k])) * CA2[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T2[k])) * CA2[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T2[k])) * CA2[k]) + Q2/(rho*cp*V2))*dt

    CA3[k+1] = CA3[k] + (((F2 - Fr1) / V3) * (CA2[k] - CA3[k]) + (F03 / V3) * (CA03 - CA3[k]) - (k10 * exp(-E1 / (R * T3[k])) * CA3[k] + k20 * exp(-E2 / (R * T3[k])) * CA3[k] + k30 * exp(-E3 / (R * T3[k])) * CA3[k])) * dt
    T3[k+1] = T3[k] + (((F2 - Fr1) / V3) * (T2[k] - T3[k]) + (F03 / V3) * (T03 - T3[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T3[k])) * CA3[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T3[k])) * CA3[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T3[k])) * CA3[k]) + Q3/(rho*cp*V3))*dt

    CA4[k+1] = CA4[k] + ((F3 / V4) * (CA3[k] - CA4[k]) + (F04 / V4) * (CA04 - CA4[k]) - (k10 * exp(-E1 / (R * T4[k])) * CA4[k] + k20 * exp(-E2 / (R * T4[k])) * CA4[k] + k30 * exp(-E3 / (R * T4[k])) * CA4[k])) * dt
    T4[k+1] = T4[k] + ((F3 / V4) * (T3[k] - T4[k]) + (F04 / V4) * (T04 - T4[k]) - ((delH1 / (rho * cp)) * k10 * exp(-E1 / (R * T4[k])) * CA4[k] + (delH2 / (rho * cp)) * k20 * exp(-E2 / (R * T4[k])) * CA4[k] + (delH3 / (rho * cp)) * k30 * exp(-E3 / (R * T4[k])) * CA4[k]) + Q4/(rho*cp*V4))*dt
end

CA4[end]