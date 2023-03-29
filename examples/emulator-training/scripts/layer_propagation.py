import sympy as sy
import pprint



def symbolic_verification():

    # Downwelling radiation flux consists of direct beam (solar) and a diffuse (scattered)
    # flux

    # The top layer splits direct beam into transmitted and extinguished components
    t_direct, e_direct = sy.symbols('t_direct e_direct')
    e1 = sy.Eq(e_direct, 1 - t_direct)
    
    # The top layer also splits diffuse flux into transmitted and extinguished components
    t_diffuse, e_diffuse = sy.symbols('t_diffuse e_diffuse')
    e2 = sy.Eq(e_diffuse, 1 - t_diffuse)

    # The top layer further splits the extinguished components into transmitted, reflected,
    # and absorbed parts
    e_t_diffuse, e_r_diffuse, e_a_diffuse = sy.symbols('e_t_diffuse e_r_diffuse e_a_diffuse')
    e3a = sy.Eq(1, e_t_diffuse + e_r_diffuse + e_a_diffuse)
    e_t_direct, e_r_direct, e_a_direct = sy.symbols('e_t_direct e_r_direct e_a_direct')
    e3b = sy.Eq(1, e_t_direct + e_r_direct + e_a_direct)

    # The "bottom layer" is a single thick layer spanning all the layers beneath 
    # the top layer including the surface

    # Absorption and reflection of this bottom layer for direct and diffuse
    # radiation, respectively
    # Since this bottom layer includes the surface, it has no transmission
    r_bottom_direct, a_bottom_direct = sy.symbols('r_bottom_direct a_bottom_direct')
    e4 = sy.Eq(1, a_bottom_direct + r_bottom_direct)
    r_bottom_diffuse, a_bottom_diffuse = sy.symbols('r_bottom_diffuse a_bottom_diffuse')
    e5 = sy.Eq(1, a_bottom_diffuse + r_bottom_diffuse)

    # Multi-reflection between the top layer and bottom layer resolves 
    # a direct beam into:
    #   r_multi_direct - reflection at the top layer
    #   a_multi_direct - absorption at the top layer
    #   a_multi_direct_bottom - absorption for the entire bottom layer
    # The adding-doubling method computes these
    # See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
    # by Grant W. Petty
    a_multi_direct, a_multi_direct_bottom, r_multi_direct = sy.symbols('a_multi_direct a_multi_direct_bottom r_multi_direct')
    d = sy.symbols('d')
    e6 = sy.Eq(d, 1 / (1 - e_diffuse * e_r_diffuse * r_bottom_diffuse))
    e7 = sy.Eq(a_multi_direct_bottom, t_direct*a_bottom_direct + t_direct*r_bottom_direct*e_diffuse*e_r_diffuse*a_bottom_diffuse*d + e_direct*e_t_direct*a_bottom_diffuse*d)
    e8 = sy.Eq(a_multi_direct, e_direct*e_a_direct + t_direct*r_bottom_direct*e_diffuse*e_a_diffuse*d + e_direct*e_t_direct*r_bottom_diffuse*e_diffuse*e_a_diffuse*d)
    e9 = sy.Eq(r_multi_direct, e_direct*e_r_direct + t_direct*r_bottom_direct*t_diffuse*d + t_direct*r_bottom_direct*e_diffuse*e_t_diffuse*d + e_direct*e_t_direct*r_bottom_diffuse*t_diffuse*d + e_direct*e_t_direct*r_bottom_diffuse*e_diffuse*e_t_diffuse*d)

    # These must sum to 1 to insure conservation of energy
    total = sy.symbols('total')
    e10 = sy.Eq(total, a_multi_direct_bottom + a_multi_direct + r_multi_direct)
    equations = (e1, e2, e3a, e3b, e4, e5, e6, e7, e8, e9, e10)
    solutions = sy.solve(equations, dict=True, manual=True)
    #pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    #pp.pprint(solutions)
    print(solutions)

    # Multireflection for diffuse flux:
    #   r_multi_diffuse - reflection at the top layer
    #   a_multi_diffuse - absorption at the top layer
    #   a_multi_bottom_diffuse - absorption for the entire bottom layer
    a_multi_diffuse, a_multi_bottom_diffuse, r_multi_diffuse = sy.symbols('a_multi_diffuse a_multi_bottom_diffuse r_multi_diffuse')
    e11 = sy.Eq(a_multi_bottom_diffuse, t_diffuse*a_bottom_diffuse + t_diffuse*r_bottom_diffuse*e_diffuse*e_r_diffuse*a_bottom_diffuse*d + e_diffuse*e_t_diffuse*a_bottom_diffuse*d)
    e12 = sy.Eq(a_multi_diffuse, e_diffuse*e_a_diffuse + t_diffuse*r_bottom_diffuse*e_diffuse*e_a_diffuse*d + e_diffuse*e_t_diffuse*r_bottom_diffuse*e_diffuse*e_a_diffuse*d)
    e13 = sy.Eq(r_multi_diffuse, e_diffuse*e_r_diffuse + t_diffuse*r_bottom_diffuse*t_diffuse*d + t_diffuse*r_bottom_diffuse*e_diffuse*e_t_diffuse*d + e_diffuse*e_t_diffuse*r_bottom_diffuse*t_diffuse*d + e_diffuse*e_t_diffuse*r_bottom_diffuse*e_diffuse*e_t_diffuse*d)

    # Verify that these sum to 1
    total_diffuse = sy.symbols('total_diffuse')
    e14 = sy.Eq(total_diffuse, a_multi_bottom_diffuse + a_multi_diffuse + r_multi_diffuse)
    equations_diffuse = (e1, e2, e3a, e3b, e4, e5, e6, e11, e12, e13, e14)
    solutions_diffuse = sy.solve(equations_diffuse, dict=True, manual=True)
    #pp.pprint(solutions_diffuse)
    print(solutions_diffuse)


if __name__ == "__main__":
    symbolic_verification()





