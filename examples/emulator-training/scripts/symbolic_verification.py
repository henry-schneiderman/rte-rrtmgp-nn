import sympy as sy
import pprint

#def symbolic_verfication():
if True:
    t_hat, r_hat, a_hat = sy.symbols('h_hat r_hat a_hat')
    e1 = sy.Eq(1, t_hat + r_hat + a_hat)

    a_0, r_0 = sy.symbols('a_0 r_0')
    e2 = sy.Eq(1, a_0 + r_0)

    a_0d, r_0d = sy.symbols('a_0d r_0d')
    e3 = sy.Eq(1, a_0d + r_0d)

    t_1, t_1d, ct_1, ct_1d = sy.symbols('t_1 t_1d ct_1 ct_1d')
    e4 = sy.Eq(ct_1, 1 - t_1)
    e5 = sy.Eq(ct_1d, 1 - t_1d)

    a_1t, a_0t, r_1t = sy.symbols('a_1t a_0t r_1t')
    d = sy.symbols('d')
    e6 = sy.Eq(d, 1 / (1 - ct_1d * r_hat * r_0d))
    e7 = sy.Eq(a_0t, t_1*a_0 + t_1*r_0*ct_1d*r_hat*a_0d*d + ct_1*t_hat*a_0d*d)
    e8 = sy.Eq(a_1t, ct_1*a_hat + t_1*r_0*ct_1d*a_hat*d + ct_1*t_hat*r_0d*ct_1d*a_hat*d)
    e9 = sy.Eq(r_1t, ct_1*r_hat + t_1*r_0*t_1d*d + t_1*r_0*ct_1d*t_hat*d + ct_1*t_hat*r_0d*t_1d*d + ct_1*t_hat*r_0d*ct_1d*t_hat*d)


    total = sy.symbols('total')
    e10 = sy.Eq(total, a_0t + a_1t + r_1t)
    equations = (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)
    solutions = sy.solve(equations, dict=True, manual=True)
    print(solutions)
    #pprint.pprint(solutions)

    a_1td, a_0td, r_1td = sy.symbols('a_1td a_0td r_1td')
    e11 = sy.Eq(a_0td, t_1d*a_0d + t_1d*r_0d*ct_1d*r_hat*a_0d*d + ct_1d*t_hat*a_0d*d)
    e12 = sy.Eq(a_1td, ct_1d*a_hat + t_1d*r_0d*ct_1d*a_hat*d + ct_1d*t_hat*r_0d*ct_1d*a_hat*d)
    e13 = sy.Eq(r_1td, ct_1d*r_hat + t_1d*r_0d*t_1d*d + t_1d*r_0d*ct_1d*t_hat*d + ct_1d*t_hat*r_0d*t_1d*d + ct_1d*t_hat*r_0d*ct_1d*t_hat*d)

    total_d = sy.symbols('total_d')
    e14 = sy.Eq(total_d, a_0td + a_1td + r_1td)
    equations_d = (e1, e2, e3, e4, e5, e6, e11, e12, e13, e14)
    solutions_d = sy.solve(equations_d, dict=True, manual=True)
    print(solutions_d)







