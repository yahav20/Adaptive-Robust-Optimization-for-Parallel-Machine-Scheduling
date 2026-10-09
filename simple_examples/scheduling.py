import gurobipy as gp
from gurobipy import GRB
import itertools

d0 = [0.0580, 0.1945, 0.5866]      
d_bar = [0.95, 0.75, 0.48]         
GAMMA = 2.5                        

def sub_expr(expr1, expr2):
    """פונקציית עזר לחיסור ביטויים ליניאריים לצורך בניית מטריצת האילוצים"""
    return [expr1[i] - expr2[i] for i in range(len(expr1))]

def solve_leaf_lp(E_matrix, e_vector):
    """פתרון בעיית האופטימיזציה הרציפה עבור עלה ספציפי"""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("Leaf_LP", env=env)
    
    u = model.addVars(3, lb=0.0, ub=1.0, name="u")
    d = model.addVars(3, name="d")
    
    model.addConstr(gp.quicksum(u[i] for i in range(3)) <= GAMMA)
    
    for i in range(3):
        model.addConstr(d[i] == d0[i] + d_bar[i] * u[i])
        
    for row in E_matrix:
        model.addConstr(gp.quicksum(row[i] * d[i] for i in range(3)) <= 0)
        
    obj = gp.quicksum(e_vector[i] * d[i] for i in range(3))
    model.setObjective(obj, GRB.MAXIMIZE)
    
    model.optimize()
    return model.ObjVal if model.Status == GRB.OPTIMAL else float('-inf')

def adversary_turn(running1, running2, pending, C_expr, E_matrix):
    """
    תור היריב: מחליט מי משתי המשימות שכרגע רצות על המכונות תסתיים ראשונה.
    המטרה שלו היא למקסם את ה-makespan.
    """
    # אפשרות 1: משימה running1 מסתיימת קודם (C1 <= C2 -> C1 - C2 <= 0)
    E1 = E_matrix.copy()
    E1.append(sub_expr(C_expr[running1], C_expr[running2]))
    val1 = scheduler_turn(finished=running1, still_running=running2, pending=pending, 
                          C_expr=C_expr, E_matrix=E1)
                          
    # אפשרות 2: משימה running2 מסתיימת קודם (C2 <= C1 -> C2 - C1 <= 0)
    E2 = E_matrix.copy()
    E2.append(sub_expr(C_expr[running2], C_expr[running1]))
    val2 = scheduler_turn(finished=running2, still_running=running1, pending=pending, 
                          C_expr=C_expr, E_matrix=E2)
                          
    # היריב בוחר את התרחיש הגרוע ביותר (מקסום)
    return max(val1, val2)

def scheduler_turn(finished, still_running, pending, C_expr, E_matrix):
    """
    תור המתזמן: משבץ משימה חדשה למכונה שהתפנתה, או מסיים אם אין משימות.
    """
    if pending is None:
        # הגענו לעלה - כל המשימות שובצו ונותר רק לחכות שהאחרונה תסיים.
        # המשימה שתסיים אחרונה היא זו שמוגדרת כ-still_running.
        makespan = solve_leaf_lp(E_matrix, C_expr[still_running])
        return makespan
        
    # אם יש משימה ממתינה, משבצים אותה.
    # זמן הסיום שלה שווה לזמן הסיום של המשימה הקודמת על אותה מכונה + זמן הביצוע שלה.
    new_C_expr = C_expr.copy()
    new_task_expr = C_expr[finished].copy()
    new_task_expr[pending] += 1  
    new_C_expr[pending] = new_task_expr
    
    # לאחר השיבוץ, תור היריב להחליט מי תסיים קודם מבין שתי הרצות כעת.
    return adversary_turn(still_running, pending, None, new_C_expr, E_matrix)

def evaluate_tree():
    tasks = [0, 1, 2]
    best_makespan = float('inf')
    best_initial_pair = None
    
    print("--- Minimax Robust Scheduling (Recursive Tree Baseline) ---")
    
    # תור המתזמן הראשוני (Root): בוחר אילו שתי משימות יתחילו לרוץ
    for pair in itertools.combinations(tasks, 2):
        task_A, task_B = pair
        pending = [t for t in tasks if t not in pair][0]
        
        # יצירת ביטוי ליניארי שמייצג את זמן הסיום (C) של כל משימה (מיוצג ע"י מקדמים)
        C_expr = {
            task_A: [1 if i == task_A else 0 for i in range(3)],
            task_B: [1 if i == task_B else 0 for i in range(3)]
        }
        
        makespan_for_pair = adversary_turn(task_A, task_B, pending, C_expr, [])
        
        print(f"Initial choice Tasks {task_A+1},{task_B+1} -> Worst-case makespan: {makespan_for_pair:.4f}")
        
        # המתזמן מחפש למזער את ה-makespan
        if makespan_for_pair < best_makespan:
            best_makespan = makespan_for_pair
            best_initial_pair = pair
            
    print("\n" + "="*40)
    print(f"Optimal AR Policy: Start with Tasks {best_initial_pair[0]+1} and {best_initial_pair[1]+1}")
    print(f"Guaranteed Worst-Case Makespan: {best_makespan:.4f}")

if __name__ == "__main__":
    evaluate_tree()