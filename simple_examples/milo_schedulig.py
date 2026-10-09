import gurobipy as gp
from gurobipy import GRB

def solve_full_milo_model():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  
    env.start()
    m = gp.Model("MILO_Adaptive_Robust", env=env)

    d0 = [0.0580, 0.1945, 0.5866]
    d_bar = [0.95, 0.75, 0.48]
    GAMMA = 2.5
    
    # הפתרון למלכודת: הפרדת קבועי Big-M
    M_z = 10.0   # קבוע לפסילת מסלולים לא הגיוניים
    M_w = 100.0  # קבוע לבחירת המקסימום (חייב להיות גדול מ-Mz + Makespan מקסימלי)

    initials = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    t = {}
    
    t['root'] = m.addVar(lb=-GRB.INFINITY, name="t_root")
    
    for (ta, tb, tc) in initials:
        n_node = f"N_{ta}_{tb}"
        t[n_node] = m.addVar(lb=-GRB.INFINITY, name=f"t_{n_node}")
        
        m.addConstr(t['root'] <= t[n_node], name=f"min_root_to_{n_node}")
        w_n = []
        
        for first_finisher, other_running in [(ta, tb), (tb, ta)]:
            s_node = f"S_initial_{ta}_{tb}_finished_{first_finisher}"
            t[s_node] = m.addVar(lb=-GRB.INFINITY, name=f"t_{s_node}")
            
            w = m.addVar(vtype=GRB.BINARY, name=f"w_{n_node}_{s_node}")
            # שימוש ב-Mw לאילוצי בחירת היריב
            m.addConstr(t[n_node] <= t[s_node] + M_w * w)
            w_n.append(w)
            
            m_node = f"M_running_{other_running}_and_{tc}"
            t[m_node] = m.addVar(lb=-GRB.INFINITY, name=f"t_{m_node}")
            m.addConstr(t[s_node] <= t[m_node])
            
            w_m = []
            
            for second_finisher, last_finisher in [(tc, other_running), (other_running, tc)]:
                leaf = f"Leaf_{ta}_{tb}_finish1_{first_finisher}_finish2_{second_finisher}"
                t[leaf] = m.addVar(lb=-GRB.INFINITY, name=f"t_{leaf}")
                
                w_leaf = m.addVar(vtype=GRB.BINARY, name=f"w_{m_node}_{leaf}")
                # שימוש ב-Mw לאילוצי בחירת היריב
                m.addConstr(t[m_node] <= t[leaf] + M_w * w_leaf)
                w_m.append(w_leaf)
                
                d = m.addVars(3, name=f"d_{leaf}")
                u = m.addVars(3, lb=0.0, ub=1.0, name=f"u_{leaf}")
                z = m.addVar(vtype=GRB.BINARY, name=f"z_{leaf}")
                
                m.addConstr(gp.quicksum(u[i] for i in range(3)) <= GAMMA)
                for i in range(3):
                    m.addConstr(d[i] == d0[i] + d_bar[i] * u[i])
                    
                # שימוש ב-Mz לאילוצי פסילת עלה חריג
                m.addConstr(d[first_finisher] - d[other_running] <= z * M_z)
                
                if second_finisher == tc:
                    m.addConstr(d[first_finisher] + d[tc] - d[other_running] <= z * M_z)
                    m.addConstr(t[leaf] == d[other_running] - z * M_z)
                else:
                    m.addConstr(d[other_running] - (d[first_finisher] + d[tc]) <= z * M_z)
                    m.addConstr(t[leaf] == d[first_finisher] + d[tc] - z * M_z)
            
            m.addConstr(gp.quicksum(w_m) <= 1)
        m.addConstr(gp.quicksum(w_n) <= 1)

    m.update()  # build the variable-name index so getVarByName works
    for (ta, tb, tc) in initials:
        for first_finisher, other_running in [(ta, tb), (tb, ta)]:
            leaf1 = f"Leaf_{ta}_{tb}_finish1_{first_finisher}_finish2_{tc}"
            leaf2 = f"Leaf_{ta}_{tb}_finish1_{first_finisher}_finish2_{other_running}"
            d1 = m.getVarByName(f"d_{leaf1}[{first_finisher}]")
            d2 = m.getVarByName(f"d_{leaf2}[{first_finisher}]")
            m.addConstr(d1 == d2, name=f"NonAnticipativity_{leaf1}_{leaf2}")
            
    m.setObjective(t['root'], GRB.MAXIMIZE)
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        print(f"\n[MILO Formulation] Optimal AR Makespan = {m.ObjVal:.4f}")
    else:
        print("\nMILO Model could not be solved to optimality.")

if __name__ == "__main__":
    solve_full_milo_model()