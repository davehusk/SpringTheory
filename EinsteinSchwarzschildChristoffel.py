
import sympy as sp

# Define the symbols for the Einstein field equations
R, g, T = sp.symbols('R g T', cls=sp.Function)
G, c, pi = sp.symbols('G c pi')
mu, nu = sp.symbols('mu nu')

# Define the Einstein field equations
einstein_eq = sp.Eq(R(mu, nu) - 1/2 * R(g(mu, nu)), 8 * pi * G / c**4 * T(mu, nu))

# Display the Einstein field equations
print("Einstein field equations:", einstein_eq)

# Define the Schwarzschild metric components
t, r, theta, phi, M = sp.symbols('t r theta phi M')
G, c = sp.symbols('G c')
g_tt = -(1 - 2*G*M/(c**2*r))
g_rr = (1 - 2*G*M/(c**2*r))**(-1)
g_thth = r**2
g_phiphi = r**2 * sp.sin(theta)**2

# Display the metric components
print("Schwarzschild metric components:")
print("g_tt:", g_tt)                # (2*G*M/(c**2*r) - 1)
print("g_rr:", g_rr)                # (1/(-2*G*M/(c**2*r) + 1))
print("g_thth:", g_thth)            # (r**2)
print("g_phiphi:", g_phiphi)        # (r**2*sin(theta)**2)

# Define the Schwarzschild metric as a matrix
g = sp.Matrix([
    [g_tt, 0, 0, 0],
    [0, g_rr, 0, 0],
    [0, 0, g_thth, 0],
    [0, 0, 0, g_phiphi]
])

# Define the inverse metric
g_inv = g.inv()

# Define the Christoffel symbols
def christoffel(g, g_inv):
    symbols = sp.symbols('t r theta phi')
    Gamma = sp.MutableDenseNDimArray.zeros(4, 4, 4)
    for k in range(4):
        for i in range(4):
            for j in range(4):
                Gamma[k, i, j] = 0.5 * sum([g_inv[k, l] * (sp.diff(g[l, i], symbols[j]) +
                                                          sp.diff(g[l, j], symbols[i]) -
                                                          sp.diff(g[i, j], symbols[l])) for l in range(4)])
    return Gamma

# Redefine the symbols correctly
symbols = sp.symbols('t r theta phi')

# Recompute the Christoffel symbols
Gamma = christoffel(g, g_inv)

# Define the Ricci tensor computation correctly
def ricci_tensor(Gamma, symbols):
    R = sp.MutableDenseNDimArray.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            R[i, j] = sum([sp.diff(Gamma[k, i, j], symbols[k]) -
                           sp.diff(Gamma[k, i, k], symbols[j]) +
                           sum([Gamma[l, i, j] * Gamma[k, l, k] -
                                Gamma[l, i, k] * Gamma[k, l, j] for l in range(4)]) for k in range(4)])
    return R

R = ricci_tensor(Gamma, symbols)

# Display the Ricci tensor components
print("Ricci tensor components:")
print("R_tt:", R[0, 0])             # R_tt: -1.0*G**2*M**2/(c**4*r**4) - 1.0*G**2*M**2*(-2*G*M + c**2*r)**2/(c**8*r**6*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M/(c**2*r**3) - 1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4)
print("R_rr:", R[1, 1])             # R_rr: -1.0*G**2*M**2/(r**2*(-2*G*M + c**2*r)**2) - 1.0*G**2*M**2/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M*c**2/(r*(-2*G*M + c**2*r)**2) + 1.0*G*M/(r**2*(-2*G*M + c**2*r)) - 2.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2)
print("R_thth:", R[2, 2])           # R_thth: -1.0*G*M/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2)
print("R_phiphi:", R[3, 3])         # R_phiphi: -1.0*G*M*sin(theta)**2/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2*sin(theta)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2)
print("\n")
print("Display the full Ricci tensor matrix:")
print(R)      

# [[-1.0*G**2*M**2/(c**4*r**4) - 1.0*G**2*M**2*(-2*G*M + c**2*r)**2/(c**8*r**6*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M/(c**2*r**3) - 1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4), 0, 0, 0], [0, -1.0*G**2*M**2/(r**2*(-2*G*M + c**2*r)**2) - 1.0*G**2*M**2/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M*c**2/(r*(-2*G*M + c**2*r)**2) + 1.0*G*M/(r**2*(-2*G*M + c**2*r)) - 2.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2), 0, 0], [0, 0, -1.0*G*M/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2), 0], [0, 0, 0, -1.0*G*M*sin(theta)**2/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2*sin(theta)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2)]] 
