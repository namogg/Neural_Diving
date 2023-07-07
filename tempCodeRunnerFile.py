def print_model_vars_cons(model):
    # Print variables
    print("Objective:")
    obj_expr = model.getObjective()
    print(f"Objective Expression: {obj_expr}")
    print("Variables:")
    for var in model.getVars():
        print(f"Variable: {var.name}, Type: {var.vtype()}")

    # Print constraints
    print("Constraints:")
    for cons in model.getConss():
        print(f"Constraint: {cons.name}, LHS: {model.getLhs(cons)}, RHS: {model.getRhs(cons)}")

    return