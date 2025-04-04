ISE = sum((V1_vec[k] - V1_sp)^2 + (V2_vec[k] - V2_sp)^2 + (V3_vec[k] - V3_sp)^2 + (V4_vec[k] - V4_sp)^2 + (V5_vec[k] - V5_sp)^2 +
(T1_vec[k] - T1_sp)^2 + (T2_vec[k] - T2_sp)^2 + (T3_vec[k] - T3_sp)^2 + (T4_vec[k] - T4_sp)^2 + (T5_vec[k] - T5_sp)^2 +
(CA1_vec[k] - CA1_sp)^2 + (CA2_vec[k] - CA2_sp)^2 + (CA3_vec[k] - CA3_sp)^2 + (CA4_vec[k] - CA4_sp)^2 + (CA5_vec[k] - CA5_sp)^2 +
(CB1_vec[k] - CB1_sp)^2 + (CB2_vec[k] - CB2_sp)^2 + (CB3_vec[k] - CB3_sp)^2 + (CB4_vec[k] - CB4_sp)^2 + (CB5_vec[k] - CB5_sp)^2 +
(CC1_vec[k] - CC1_sp)^2 + (CC2_vec[k] - CC2_sp)^2 + (CC3_vec[k] - CC3_sp)^2 + (CC4_vec[k] - CC4_sp)^2 + (CC5_vec[k] - CC5_sp)^2 +
(CD1_vec[k] - CD1_sp)^2 + (CD2_vec[k] - CD2_sp)^2 + (CD3_vec[k] - CD3_sp)^2 + (CD4_vec[k] - CD4_sp)^2 + (CD5_vec[k] - CD5_sp)^2 for k = 1:N+1)



ISC = sum((F1[k] - F1_sp)^2 + (F2[k] - F2_sp)^2 + (F3[k] - F3_sp)^2 + (F4[k] - F4_sp)^2 + (F5[k] - F5_sp)^2 +
(F6[k] - F6_sp)^2 + (F7[k] - F7_sp)^2 + (F8[k] - F8_sp)^2 + (F9[k] - F9_sp)^2 +  (F10[k] - F10_sp)^2 +  (Fr1[k] - Fr1_sp)^2 +  (Fr2[k] - Fr2_sp)^2 +
(Q1[k] - Q1_sp)^2 + (Q2[k] - Q2_sp)^2 + (Q3[k] - Q3_sp)^2 + (Q4[k] - Q4_sp)^2 + (Q5[k] - Q5_sp)^2 for k = 1:N+1)
