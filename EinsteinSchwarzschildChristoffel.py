import sympy as sp

# Define the symbols for the Einstein field equations
R, g, T = sp.symbols('R g T', cls=sp.Function)
G, c, pi = sp.symbols('G c pi')
mu, nu = sp.symbols('mu nu')

# Define the Einstein field equations
einstein_eq = sp.Eq(R(mu, nu) - 1/2 * R(g(mu, nu)), 8 * pi * G / c**4 * T(mu, nu))

# Display the Einstein field equations
print("Einstein field equations:", einstein_eq)
# Eq(-0.5*R(g(mu, nu)) + R(mu, nu), 8*G*pi*T(mu, nu)/c**4)
sp.pprint(einstein_eq)
"""
                            8⋅G⋅π⋅T(μ, ν)
-0.5⋅R(g(μ, ν)) + R(μ, ν) = ─────────────
                                   4
                                  c
"""

# Define the Schwarzschild metric components
t, r, theta, phi, M = sp.symbols('t r theta phi M')
G, c = sp.symbols('G c')
g_tt = -(1 - 2*G*M/(c**2*r))
g_rr = (1 - 2*G*M/(c**2*r))**(-1)
g_thth = r**2
g_phiphi = r**2 * sp.sin(theta)**2

# Display the metric components
print("Schwarzschild metric components:")

sp.pprint("g_tt:", g_tt)                
# (2*G*M/(c**2*r) - 1)
# 2⋅G⋅M
# ───── - 1
#   2
#  c ⋅r

sp.pprint(g_tt)
# 2⋅G⋅M
# ───── - 1
#   2
#  c ⋅r

sp.pprint(g_rr)                
# (1/(-2*G*M/(c**2*r) + 1))
#     1
#───────────
#  2⋅G⋅M
#- ───── + 1
#    2
#   c ⋅r

sp.pprint(g_thth)            
# (r**2)
#  2
# r

sp.pprint(g_phiphi)        
# (r**2*sin(theta)**2)
#  2    2
# r ⋅sin (θ)


# Define the Schwarzschild metric as a matrix
g = sp.Matrix([
    [g_tt, 0, 0, 0],
    [0, g_rr, 0, 0],
    [0, 0, g_thth, 0],
    [0, 0, 0, g_phiphi]
])

"""
⎡2⋅G⋅M                                 ⎤
⎢───── - 1       0       0       0     ⎥
⎢  2                                   ⎥
⎢ c ⋅r                                 ⎥
⎢                                      ⎥
⎢                1                     ⎥
⎢    0      ───────────  0       0     ⎥
⎢             2⋅G⋅M                    ⎥
⎢           - ───── + 1                ⎥
⎢               2                      ⎥
⎢              c ⋅r                    ⎥
⎢                                      ⎥
⎢                         2            ⎥
⎢    0           0       r       0     ⎥
⎢                                      ⎥
⎢                             2    2   ⎥
⎣    0           0       0   r ⋅sin (θ)⎦
"""

# Define the inverse metric
g_inv = g.inv()

"""
⎡      2                                     ⎤
⎢    -c ⋅r                                   ⎥
⎢─────────────        0        0       0     ⎥
⎢          2                                 ⎥
⎢-2⋅G⋅M + c ⋅r                               ⎥
⎢                                            ⎥
⎢                         2                  ⎥
⎢               -2⋅G⋅M + c ⋅r                ⎥
⎢      0        ─────────────  0       0     ⎥
⎢                     2                      ⎥
⎢                    c ⋅r                    ⎥
⎢                                            ⎥
⎢                              1             ⎥
⎢      0              0        ──      0     ⎥
⎢                               2            ⎥
⎢                              r             ⎥
⎢                                            ⎥
⎢                                      1     ⎥
⎢      0              0        0   ──────────⎥
⎢                                   2    2   ⎥
⎣                                  r ⋅sin (θ)⎦
"""


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
# [[[0, 1.0*G*M/(r*(-2*G*M + c**2*r)), 0, 0], [1.0*G*M/(r*(-2*G*M + c**2*r)), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**3), 0, 0, 0], [0, -1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**3*(-2*G*M/(c**2*r) + 1)**2), 0, 0], [0, 0, -1.0*(-2*G*M + c**2*r)/c**2, 0], [0, 0, 0, -1.0*(-2*G*M + c**2*r)*sin(theta)**2/c**2]], [[0, 0, 0, 0], [0, 0, 1.0/r, 0], [0, 1.0/r, 0, 0], [0, 0, 0, -1.0*sin(theta)*cos(theta)]], [[0, 0, 0, 0], [0, 0, 0, 1.0/r], [0, 0, 0, 1.0*cos(theta)/sin(theta)], [0, 1.0/r, 1.0*cos(theta)/sin(theta), 0]]]

# Display the Christoffel symbols
print("Christoffel symbols:")
print("Gamma_ttt:", Gamma[0, 0, 0])     # Gamma_ttt: 0
print("Gamma_trr:", Gamma[0, 1, 1])     # Gamma_trr: 0
print("Gamma_tth:", Gamma[0, 2, 2])     # Gamma_tth: 0
print("Gamma_tph:", Gamma[0, 3, 3])     # Gamma_tph: 0
print("\n")
print("Display the full Christoffel symbols matrix:")
print(Gamma)
# [[[0, 1.0*G*M/(r*(-2*G*M + c**2*r)), 0, 0], [1.0*G*M/(r*(-2*G*M + c**2*r)), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**3), 0, 0, 0], [0, -1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**3*(-2*G*M/(c**2*r) + 1)**2), 0, 0], [0, 0, -1.0*(-2*G*M + c**2*r)/c**2, 0], [0, 0, 0, -1.0*(-2*G*M + c**2*r)*sin(theta)**2/c**2]], [[0, 0, 0, 0], [0, 0, 1.0/r, 0], [0, 1.0/r, 0, 0], [0, 0, 0, -1.0*sin(theta)*cos(theta)]], [[0, 0, 0, 0], [0, 0, 0, 1.0/r], [0, 0, 0, 1.0*cos(theta)/sin(theta)], [0, 1.0/r, 1.0*cos(theta)/sin(theta), 0]]]
sp.pprint(Gamma)
"""
⎡                                              ⎡        ⎛          2  ⎞       
⎢                                              ⎢1.0⋅G⋅M⋅⎝-2⋅G⋅M + c ⋅r⎠
⎢                                              ⎢───────────────────────
⎢                                              ⎢          4  3
⎢                                              ⎢         c ⋅r
⎢                                              ⎢
⎢                                              ⎢
⎢⎡                        1.0⋅G⋅M           ⎤  ⎢                         -1.0⋅
⎢⎢        0          ─────────────────  0  0⎥  ⎢           0             ─────
⎢⎢                     ⎛          2  ⎞      ⎥  ⎢
⎢⎢                   r⋅⎝-2⋅G⋅M + c ⋅r⎠      ⎥  ⎢                             4
⎢⎢                                          ⎥  ⎢                            c
⎢⎢     1.0⋅G⋅M                              ⎥  ⎢
⎢⎢─────────────────          0          0  0⎥  ⎢
⎢⎢  ⎛          2  ⎞                         ⎥  ⎢
⎢⎢r⋅⎝-2⋅G⋅M + c ⋅r⎠                         ⎥  ⎢
⎢⎢                                          ⎥  ⎢
⎢⎢        0                  0          0  0⎥  ⎢           0
⎢⎢                                          ⎥  ⎢
⎢⎣        0                  0          0  0⎦  ⎢
⎢                                              ⎢
⎢                                              ⎢
⎢                                              ⎢
⎢                                              ⎢           0
⎢                                              ⎢
⎣                                              ⎣

                                                                          ⎤
                                                                          ⎥
       0                        0                          0              ⎥
                                                                          ⎥
                                                                          ⎥
                                                                          ⎥
    ⎛          2  ⎞                                                       ⎥
G⋅M⋅⎝-2⋅G⋅M + c ⋅r⎠                                                       ⎥
────────────────────            0                          0              ⎥  ⎡
                 2                                                        ⎥  ⎢
  3 ⎛  2⋅G⋅M    ⎞                                                         ⎥  ⎢
⋅r ⋅⎜- ───── + 1⎟                                                         ⎥  ⎢
    ⎜    2      ⎟                                                         ⎥  ⎢
    ⎝   c ⋅r    ⎠                                                         ⎥  ⎢
                                                                          ⎥  ⎢
                           ⎛          2  ⎞                                ⎥  ⎢
                      -1.0⋅⎝-2⋅G⋅M + c ⋅r⎠                                ⎥  ⎢
       0              ─────────────────────                0              ⎥  ⎢
                                 2                                        ⎥  ⎣
                                c                                         ⎥
                                                                          ⎥
                                                  ⎛          2  ⎞    2    ⎥
                                             -1.0⋅⎝-2⋅G⋅M + c ⋅r⎠⋅sin (θ) ⎥
       0                        0            ─────────────────────────────⎥
                                                            2             ⎥
                                                           c              ⎦

                                                                  ⎤
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                  ⎡0   0       0           0     ⎤⎥
0   0    0           0         ⎤  ⎢                              ⎥⎥
                               ⎥  ⎢                       1.0    ⎥⎥
        1.0                    ⎥  ⎢0   0       0          ───    ⎥⎥
0   0   ───          0         ⎥  ⎢                        r     ⎥⎥
         r                     ⎥  ⎢                              ⎥⎥
                               ⎥  ⎢                    1.0⋅cos(θ)⎥⎥
0  ───   0           0         ⎥  ⎢                      sin(θ)  ⎥⎥
    r                          ⎥  ⎢                              ⎥⎥
                               ⎥  ⎢   1.0  1.0⋅cos(θ)            ⎥⎥
0   0    0   -1.0⋅sin(θ)⋅cos(θ)⎦  ⎢0  ───  ──────────      0     ⎥⎥
                                  ⎣    r     sin(θ)              ⎦⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎥
                                                                  ⎦
                                                                  
"""

# Define the Ricci tensor computation correctly
def ricci_tensor(Gamma, symbols):
    R = sp.MutableDenseNDimArray.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            R[i, j] = sum([sp.diff(Gamma[k, i, j], symbols[k]) - sp.diff(Gamma[k, i, k], symbols[j]) +
                           sum([Gamma[l, i, j] * Gamma[k, l, k] - Gamma[l, i, k] * Gamma[k, l, j] for l in range(4)]) for k in range(4)])
    return R

R = ricci_tensor(Gamma, symbols)

# Display the Ricci tensor components
print("Ricci tensor components:")
print("R_tt:", R[0, 0])             
# R_tt: -1.0*G**2*M**2/(c**4*r**4) - 1.0*G**2*M**2*(-2*G*M + c**2*r)**2/(c**8*r**6*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M/(c**2*r**3) - 1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4)
print("R_rr:", R[1, 1])             
# R_rr: -1.0*G**2*M**2/(r**2*(-2*G*M + c**2*r)**2) - 1.0*G**2*M**2/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M*c**2/(r*(-2*G*M + c**2*r)**2) + 1.0*G*M/(r**2*(-2*G*M + c**2*r)) - 2.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2)
print("R_thth:", R[2, 2])           
# R_thth: -1.0*G*M/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2)
print("R_phiphi:", R[3, 3])         
# R_phiphi: -1.0*G*M*sin(theta)**2/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2*sin(theta)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2)
print("\n")
print("Display the full Ricci tensor matrix:")
print(R)
# [[-1.0*G**2*M**2/(c**4*r**4) - 1.0*G**2*M**2*(-2*G*M + c**2*r)**2/(c**8*r**6*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M/(c**2*r**3) - 1.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4), 0, 0, 0], [0, -1.0*G**2*M**2/(r**2*(-2*G*M + c**2*r)**2) - 1.0*G**2*M**2/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2) + 1.0*G*M*c**2/(r*(-2*G*M + c**2*r)**2) + 1.0*G*M/(r**2*(-2*G*M + c**2*r)) - 2.0*G*M*(-2*G*M + c**2*r)/(c**4*r**4*(-2*G*M/(c**2*r) + 1)**2), 0, 0], [0, 0, -1.0*G*M/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2), 0], [0, 0, 0, -1.0*G*M*sin(theta)**2/(c**2*r) + 1.0*G*M*(-2*G*M + c**2*r)**2*sin(theta)**2/(c**6*r**3*(-2*G*M/(c**2*r) + 1)**2)]] 
sp.pprint(R)
"""
⎡                               2
⎢   2  2    2  2 ⎛          2  ⎞                  ⎛          2  ⎞
⎢  G ⋅M    G ⋅M ⋅⎝-2⋅G⋅M + c ⋅r⎠    1.0⋅G⋅M   G⋅M⋅⎝-2⋅G⋅M + c ⋅r⎠
⎢- ───── - ────────────────────── + ─────── - ───────────────────
⎢   4  4                       2      2  3            4  4
⎢  c ⋅r      8  6 ⎛  2⋅G⋅M    ⎞      c ⋅r            c ⋅r
⎢           c ⋅r ⋅⎜- ───── + 1⎟
⎢                 ⎜    2      ⎟
⎢                 ⎝   c ⋅r    ⎠
⎢
⎢                                                                            2
⎢                                                                           G
⎢                               0                                  - ─────────
⎢
⎢                                                                     2 ⎛
⎢                                                                    r ⋅⎝-2⋅G⋅
⎢
⎢
⎢
⎢
⎢
⎢
⎢                               0
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢                               0
⎢
⎢
⎢
⎢
⎣




                                            0






  2                  2  2                        2
⋅M                  G ⋅M                1.0⋅G⋅M⋅c             1.0⋅G⋅M
────────── - ──────────────────── + ────────────────── + ────────────────── -
         2                      2                    2    2 ⎛          2  ⎞
     2  ⎞     4  4 ⎛  2⋅G⋅M    ⎞      ⎛          2  ⎞    r ⋅⎝-2⋅G⋅M + c ⋅r⎠
M + c ⋅r⎠    c ⋅r ⋅⎜- ───── + 1⎟    r⋅⎝-2⋅G⋅M + c ⋅r⎠
                   ⎜    2      ⎟
                   ⎝   c ⋅r    ⎠




                                            0









                                            0









                                         0






        ⎛          2  ⎞
2.0⋅G⋅M⋅⎝-2⋅G⋅M + c ⋅r⎠
───────────────────────                  0
                     2
   4  4 ⎛  2⋅G⋅M    ⎞
  c ⋅r ⋅⎜- ───── + 1⎟
        ⎜    2      ⎟
        ⎝   c ⋅r    ⎠

                                                         2
                                          ⎛          2  ⎞
                           G⋅M    1.0⋅G⋅M⋅⎝-2⋅G⋅M + c ⋅r⎠
                         - ──── + ────────────────────────
                            2                          2
                           c ⋅r      6  3 ⎛  2⋅G⋅M    ⎞
                                    c ⋅r ⋅⎜- ───── + 1⎟
                                          ⎜    2      ⎟
                                          ⎝   c ⋅r    ⎠


                                                                     2
                                                              G⋅M⋅sin (θ)   1.
                                         0                  - ─────────── + ──
                                                                   2
                                                                  c ⋅r




                              ⎤
                              ⎥
                              ⎥
     0                        ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
     0                        ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
     0                        ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                              ⎥
                     2        ⎥
      ⎛          2  ⎞     2   ⎥
0⋅G⋅M⋅⎝-2⋅G⋅M + c ⋅r⎠ ⋅sin (θ)⎥
──────────────────────────────⎥
                       2      ⎥
     6  3 ⎛  2⋅G⋅M    ⎞       ⎥
    c ⋅r ⋅⎜- ───── + 1⎟       ⎥
          ⎜    2      ⎟       ⎥
          ⎝   c ⋅r    ⎠       ⎦
"""
