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

        # volHoldUp1[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F1[k] + F2[k] + Fr2[k] - F3[k] == -(V1[k] - V1_sp) / tau
        # volHoldUp2[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / tau
        # volHoldUp3[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / tau
        # volHoldUp4[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        # volHoldUp5[k=[0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140]], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau

        volHoldUp1[k in 0:dtd:(N-dtd)], F1[k] + F2[k] + Fr2[k] - F3[k] == -(V1[k] - V1_sp) / tau
        volHoldUp2[k in 0:dtd:(N-dtd)], F3[k] + F4[k] - F5[k] == -(V2[k] - V2_sp) / tau
        volHoldUp3[k in 0:dtd:(N-dtd)], F5[k] + F6[k] - F7[k] == -(V3[k] - V3_sp) / tau
        volHoldUp4[k in 0:dtd:(N-dtd)], F7[k] + F9[k] - F8[k] - Fr1[k] - Fr2[k] == -(V4[k] - V4_sp) / tau
        volHoldUp5[k in 0:dtd:(N-dtd)], F10[k] + Fr1[k] - F9[k] == -(V5[k] - V5_sp) / tau

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