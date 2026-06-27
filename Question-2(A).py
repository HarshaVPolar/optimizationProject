import numpy as np

np.random.seed(4)
n = 10

# coefficients for EV charging quartic cost
Q = np.random.randn(n, n)
Q = Q.T @ Q + 0.2 * np.eye(n)

alpha = 0.04 * (1 + np.random.rand(n))
beta  = 0.0015 * (1 + np.random.rand(n))

# constraints
E_total = 60.0
P_max = 540.0
xmin = 2.0
xmax = 12.0

# initial guess
x = np.random.uniform(low=xmin, high=xmax, size=n)

# single penalty
rho = 1e5

def f(x):
    return x @ (Q @ x) + np.sum(alpha * x**3) + np.sum(beta * x**4)

def grad_f(x):
    return 2*(Q @ x) + 3*alpha*x**2 + 4*beta*x**3

def P(x):
    # penalized function
    h = np.sum(x) - E_total
    g = np.sum(x**2) - P_max
    lb = np.maximum(0, xmin - x)
    ub = np.maximum(0, x - xmax)
    return f(x) + (rho/2)*h**2 + (rho/2)*max(0, g)**2 + (rho/2)*np.sum(lb**2) + (rho/2)*np.sum(ub**2)

def grad_P(x):
    g = grad_f(x)
    h = np.sum(x) - E_total
    g += rho * h * np.ones_like(x)

    g1 = np.sum(x**2) - P_max
    if g1 > 0:
        g += rho * g1 * (2*x)

    for i in range(n):
        if x[i] < xmin:
            g[i] -= rho * (xmin - x[i])
        if x[i] > xmax:
            g[i] += rho * (x[i] - xmax)

    return g

def backtracking(x, g, Pval):
    alpha = 1.0
    tau = 0.5
    c = 1e-4
    g_sq = np.dot(g, g)
    while True:
        x_new = x - alpha * g
        if P(x_new) <= Pval - c * alpha * g_sq:
            return x_new
        alpha *= tau
        if alpha < 1e-12:
            return x

# run 100 iterations
print("Iter |      f(x)       |       P(x)       | ||grad_P||")
print("---------------------------------------------------------")

max_iters = 100
for it in range(1, max_iters+1):
    gP = grad_P(x)
    fval = f(x)       # original cost at current x
    Pval = P(x)       # penalized value
    grad_norm_P = np.linalg.norm(gP)

    print(f"{it:4d} | {fval:14.6f} | {Pval:14.6f} | {grad_norm_P:12.6f}")

    x = backtracking(x, gP, Pval)

# FINAL x* after penalty method
fx_after_penalty = f(x)
print("\nFINAL x* =\n", x)
print("\nFinal f(x*) after penalty method =", fx_after_penalty)
print("sum x_i =", np.sum(x))
print("sum x_i^2 =", np.sum(x**2))

