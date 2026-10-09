import gurobipy as gp
from gurobipy import GRB

# נתוני הבעיה (תרחיש ייעודי ליצירת פער)
# משימות: 0, 1 (דמה שרצה 0 זמן), 2, 3
d0 = [10.0, 0.0, 10.0, 10.0]  # זמנים נומינליים
d_bar = [30.0, 0.0, 30.0, 30.0] # סטיות מקסימליות
GAMMA = 1.0                   # תקציב אי-ודאות

def solve_minimax_approach():
    """
    גישת ה-Minimax: פתרון כל עלה בנפרד (נותן ליריב כוח ראיית-הנולד)
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    # --- פתרון עלה A (המתזמן בחר במשימה 2) ---
    # Makespan = d_2. אילוץ: d_0 <= d_2
    model_A = gp.Model("Leaf_A", env=env)
    u_A = model_A.addVars(4, lb=0.0, ub=1.0)
    d_A = model_A.addVars(4)
    model_A.addConstr(gp.quicksum(u_A[i] for i in range(4)) <= GAMMA)
    for i in range(4): model_A.addConstr(d_A[i] == d0[i] + d_bar[i] * u_A[i])
    model_A.addConstr(d_A[0] <= d_A[2])
    model_A.setObjective(d_A[2], GRB.MAXIMIZE)
    model_A.optimize()
    val_A = model_A.ObjVal
    print(f"[Minimax] Branch A (Choose Task 2) -> Adversary Max Makespan: {val_A:.2f}")
    print(f"          Adversary sets: Task 0 time = {d_A[0].X:.2f}, Task 2 time = {d_A[2].X:.2f}")

    # --- פתרון עלה B (המתזמן בחר במשימה 3) ---
    # Makespan = d_0. אילוץ: d_3 <= d_0
    model_B = gp.Model("Leaf_B", env=env)
    u_B = model_B.addVars(4, lb=0.0, ub=1.0)
    d_B = model_B.addVars(4)
    model_B.addConstr(gp.quicksum(u_B[i] for i in range(4)) <= GAMMA)
    for i in range(4): model_B.addConstr(d_B[i] == d0[i] + d_bar[i] * u_B[i])
    model_B.addConstr(d_B[3] <= d_B[0])
    model_B.setObjective(d_B[0], GRB.MAXIMIZE)
    model_B.optimize()
    val_B = model_B.ObjVal
    print(f"[Minimax] Branch B (Choose Task 3) -> Adversary Max Makespan: {val_B:.2f}")
    print(f"          Adversary sets: Task 0 time = {d_B[0].X:.2f}, Task 3 time = {d_B[3].X:.2f}")

    # המתזמן לוקח את המינימום בין שני הענפים הגרועים
    minimax_result = min(val_A, val_B)
    print(f"\n[Minimax] Final Scheduler Decision Makespan: min({val_A:.2f}, {val_B:.2f}) = {minimax_result:.2f}\n")
    return minimax_result

def solve_milo_approach():
    """
    גישת ה-MILO: פתרון סימולטני שאוכף את עיקרון אי-הציפייה
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("MILO", env=env)
    
    t_root = model.addVar(lb=-GRB.INFINITY, name="t_root")
    
    # משתנים לעלה A
    u_A = model.addVars(4, lb=0.0, ub=1.0)
    d_A = model.addVars(4)
    model.addConstr(gp.quicksum(u_A[i] for i in range(4)) <= GAMMA)
    for i in range(4): model.addConstr(d_A[i] == d0[i] + d_bar[i] * u_A[i])
    model.addConstr(d_A[0] <= d_A[2])
    
    # משתנים לעלה B
    u_B = model.addVars(4, lb=0.0, ub=1.0)
    d_B = model.addVars(4)
    model.addConstr(gp.quicksum(u_B[i] for i in range(4)) <= GAMMA)
    for i in range(4): model.addConstr(d_B[i] == d0[i] + d_bar[i] * u_B[i])
    model.addConstr(d_B[3] <= d_B[0])
    
    # חיבור העלים לצומת היריב (היריב ממקסם את המינימום שהמתזמן יבחר)
    model.addConstr(t_root <= d_A[2])
    model.addConstr(t_root <= d_B[0])
    
    # האילוץ הקריטי: עיקרון אי-הציפייה (Non-anticipativity)
    # משימה 0 הסתיימה לפני הפיצול, ולכן זמן הביצוע שלה חייב להיות זהה בשני העולמות
    model.addConstr(d_A[0] == d_B[0], name="NonAnticipativity")
    
    model.setObjective(t_root, GRB.MAXIMIZE)
    model.optimize()
    
    val_milo = model.ObjVal
    print(f"[MILO]    Adversary is forced to set ONE consistent time for Task 0.")
    print(f"[MILO]    Task 0 time fixed at: {d_A[0].X:.2f}")
    print(f"[MILO]    Makespan achieved in Branch A (d_2): {d_A[2].X:.2f}")
    print(f"[MILO]    Makespan achieved in Branch B (d_0): {d_B[0].X:.2f}")
    print(f"\n[MILO]    Final Optimal AR Makespan: {val_milo:.2f}\n")
    return val_milo

if __name__ == "__main__":
    print("="*60)
    print("Comparing Minimax (Leaf-by-Leaf) vs MILO (Non-anticipativity)")
    print("="*60 + "\n")
    
    res_minimax = solve_minimax_approach()
    
    print("-" * 60 + "\n")
    
    res_milo = solve_milo_approach()
    
    print("="*60)
    print(f"Gap Identified: Minimax evaluates {res_minimax:.2f} but True MILO is {res_milo:.2f}")
    print("="*60)