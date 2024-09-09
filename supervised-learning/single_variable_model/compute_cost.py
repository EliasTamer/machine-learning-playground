
# this calculate j(w,b), which is the cost function value of f(w,b)
# this evaluates the error rate of my model, idealy this needs to be close or equals to 0.
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost