import sympy as sy
import pprint

def symbolic_verfication():

    # Downwelling radiation flux consists of direct beam (solar) and a diffuse (scattered)
    # flux

    # The top layer splits direct beam into transmitted and extinguished components
    direct_trans, direct_ext = sy.symbols('direct_trans direct_ext')
    e1 = sy.Eq(direct_ext, 1 - direct_trans)
    
    # The top layer also splits diffuse flux into transmitted and extinguished components
    diffuse_trans, diffuse_ext = sy.symbols('diffuse_trans diffuse_ext')
    e2 = sy.Eq(diffuse_ext, 1 - diffuse_trans)

    # The top layer splits all extinguished radiation into transmitted, reflected,
    # and absorbed parts
    # (this ratio is the same whether the original radiation was direct or diffuse)
    ext_t, ext_r, ext_a = sy.symbols('ext_t ext_r ext_a')
    e3 = sy.Eq(1, ext_t + ext_r + ext_a)

    # The lower atmosphere beneath the top later is represented as a single thick layer
    # including all the intervening atmospheric layers and the surface

    # Absorption and reflection of this lower layer for a direct beam (solar)
    # (since this layer includes the surface, it has no transmission)
    r_down, a_down = sy.symbols('r_down a_down')
    e4 = sy.Eq(1, a_down + r_down)

    # Absorption and reflection for diffuse downwelling radiation
    r_down_diffuse, a_down_diffuse = sy.symbols('r_down_diffuse a_down_diffuse')
    e5 = sy.Eq(1, a_down_diffuse + r_down_diffuse)

    # Multi-reflection between the top layer and lower layer resolves 
    # a direct beam into:
    #   r_multi - reflection at the top layer
    #   a_multi - absorption at the top layer
    #   a_multi_down - absorption for the entire lower layer
    # The adding-doubling method computes these
    # See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
    # by Grant W. Petty
    a_multi, a_multi_down, r_multi = sy.symbols('a_multi a_multi_down r_multi')
    d = sy.symbols('d')
    e6 = sy.Eq(d, 1 / (1 - diffuse_ext * ext_r * r_down_diffuse))
    e7 = sy.Eq(a_multi_down, direct_trans*a_down + direct_trans*r_down*diffuse_ext*ext_r*a_down_diffuse*d + direct_ext*ext_t*a_down_diffuse*d)
    e8 = sy.Eq(a_multi, direct_ext*ext_a + direct_trans*r_down*diffuse_ext*ext_a*d + direct_ext*ext_t*r_down_diffuse*diffuse_ext*ext_a*d)
    e9 = sy.Eq(r_multi, direct_ext*ext_r + direct_trans*r_down*diffuse_trans*d + direct_trans*r_down*diffuse_ext*ext_t*d + direct_ext*ext_t*r_down_diffuse*diffuse_trans*d + direct_ext*ext_t*r_down_diffuse*diffuse_ext*ext_t*d)

    # These must sum to 1 to insure conservation of energy
    total = sy.symbols('total')
    e10 = sy.Eq(total, a_multi_down + a_multi + r_multi)
    equations = (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)
    solutions = sy.solve(equations, dict=True, manual=True)
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    pp.pprint(solutions)

    # Multireflection for diffuse flux:
    #   r_multi_diffuse - reflection at the top layer
    #   a_multi_diffuse - absorption at the top layer
    #   a_multi_down_diffuse - absorption for the entire lower layer
    a_multi_diffuse, a_multi_down_diffuse, r_multi_diffuse = sy.symbols('a_multi_diffuse a_multi_down_diffuse r_multi_diffuse')
    e11 = sy.Eq(a_multi_down_diffuse, diffuse_trans*a_down_diffuse + diffuse_trans*r_down_diffuse*diffuse_ext*ext_r*a_down_diffuse*d + diffuse_ext*ext_t*a_down_diffuse*d)
    e12 = sy.Eq(a_multi_diffuse, diffuse_ext*ext_a + diffuse_trans*r_down_diffuse*diffuse_ext*ext_a*d + diffuse_ext*ext_t*r_down_diffuse*diffuse_ext*ext_a*d)
    e13 = sy.Eq(r_multi_diffuse, diffuse_ext*ext_r + diffuse_trans*r_down_diffuse*diffuse_trans*d + diffuse_trans*r_down_diffuse*diffuse_ext*ext_t*d + diffuse_ext*ext_t*r_down_diffuse*diffuse_trans*d + diffuse_ext*ext_t*r_down_diffuse*diffuse_ext*ext_t*d)

    # Verify that these sum to 1
    total_diffuse = sy.symbols('total_diffuse')
    e14 = sy.Eq(total_diffuse, a_multi_down_diffuse + a_multi_diffuse + r_multi_diffuse)
    equations_diffuse = (e1, e2, e3, e4, e5, e6, e11, e12, e13, e14)
    solutions_diffuse = sy.solve(equations_diffuse, dict=True, manual=True)
    pp.pprint(solutions_diffuse)

def symbolic_verfication_2():

    # Same as symbolic_verification() except separate direct and diffuse treatment extinguished partition
    # in transmitted, reflected, and absorbed

    # Downwelling radiation flux consists of direct beam (solar) and a diffuse (scattered)
    # flux

    # The top layer splits direct beam into transmitted and extinguished components
    direct_trans, direct_ext = sy.symbols('direct_trans direct_ext')
    e1 = sy.Eq(direct_ext, 1 - direct_trans)
    
    # The top layer also splits diffuse flux into transmitted and extinguished components
    diffuse_trans, diffuse_ext = sy.symbols('diffuse_trans diffuse_ext')
    e2 = sy.Eq(diffuse_ext, 1 - diffuse_trans)

    # The top layer splits all extinguished radiation into transmitted, reflected,
    # and absorbed parts

    diffuse_ext_t, diffuse_ext_r, diffuse_ext_a = sy.symbols('diffuse_ext_t diffuse_ext_r diffuse_ext_a')
    e3a = sy.Eq(1, diffuse_ext_t + diffuse_ext_r + diffuse_ext_a)

    direct_ext_t, direct_ext_r, direct_ext_a = sy.symbols('direct_ext_t direct_ext_r direct_ext_a')
    e3b = sy.Eq(1, direct_ext_t + direct_ext_r + direct_ext_a)

    # The lower atmosphere beneath the top later is represented as a single thick layer
    # including all the intervening atmospheric layers and the surface

    # Absorption and reflection of this lower layer for a direct beam (solar)
    # (since this layer includes the surface, it has no transmission)
    r_down, a_down = sy.symbols('r_down a_down')
    e4 = sy.Eq(1, a_down + r_down)

    # Absorption and reflection for diffuse downwelling radiation
    r_down_diffuse, a_down_diffuse = sy.symbols('r_down_diffuse a_down_diffuse')
    e5 = sy.Eq(1, a_down_diffuse + r_down_diffuse)

    # Multi-reflection between the top layer and lower layer resolves 
    # a direct beam into:
    #   r_multi - reflection at the top layer
    #   a_multi - absorption at the top layer
    #   a_multi_down - absorption for the entire lower layer
    # The adding-doubling method computes these
    # See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
    # by Grant W. Petty
    a_multi, a_multi_down, r_multi = sy.symbols('a_multi a_multi_down r_multi')
    d = sy.symbols('d')
    e6 = sy.Eq(d, 1 / (1 - diffuse_ext * diffuse_ext_r * r_down_diffuse))
    e7 = sy.Eq(a_multi_down, direct_trans*a_down + direct_trans*r_down*diffuse_ext*diffuse_ext_r*a_down_diffuse*d + direct_ext*direct_ext_t*a_down_diffuse*d)
    e8 = sy.Eq(a_multi, direct_ext*direct_ext_a + direct_trans*r_down*diffuse_ext*diffuse_ext_a*d + direct_ext*direct_ext_t*r_down_diffuse*diffuse_ext*diffuse_ext_a*d)
    e9 = sy.Eq(r_multi, direct_ext*direct_ext_r + direct_trans*r_down*diffuse_trans*d + direct_trans*r_down*diffuse_ext*diffuse_ext_t*d + direct_ext*direct_ext_t*r_down_diffuse*diffuse_trans*d + direct_ext*direct_ext_t*r_down_diffuse*diffuse_ext*diffuse_ext_t*d)

    # These must sum to 1 to insure conservation of energy
    total = sy.symbols('total')
    e10 = sy.Eq(total, a_multi_down + a_multi + r_multi)
    equations = (e1, e2, e3a, e3b, e4, e5, e6, e7, e8, e9, e10)
    solutions = sy.solve(equations, dict=True, manual=True)
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    pp.pprint(solutions)

    # Multireflection for diffuse flux:
    #   r_multi_diffuse - reflection at the top layer
    #   a_multi_diffuse - absorption at the top layer
    #   a_multi_down_diffuse - absorption for the entire lower layer
    a_multi_diffuse, a_multi_down_diffuse, r_multi_diffuse = sy.symbols('a_multi_diffuse a_multi_down_diffuse r_multi_diffuse')
    e11 = sy.Eq(a_multi_down_diffuse, diffuse_trans*a_down_diffuse + diffuse_trans*r_down_diffuse*diffuse_ext*diffuse_ext_r*a_down_diffuse*d + diffuse_ext*diffuse_ext_t*a_down_diffuse*d)
    e12 = sy.Eq(a_multi_diffuse, diffuse_ext*diffuse_ext_a + diffuse_trans*r_down_diffuse*diffuse_ext*diffuse_ext_a*d + diffuse_ext*diffuse_ext_t*r_down_diffuse*diffuse_ext*diffuse_ext_a*d)
    e13 = sy.Eq(r_multi_diffuse, diffuse_ext*diffuse_ext_r + diffuse_trans*r_down_diffuse*diffuse_trans*d + diffuse_trans*r_down_diffuse*diffuse_ext*diffuse_ext_t*d + diffuse_ext*diffuse_ext_t*r_down_diffuse*diffuse_trans*d + diffuse_ext*diffuse_ext_t*r_down_diffuse*diffuse_ext*diffuse_ext_t*d)

    # Verify that these sum to 1
    total_diffuse = sy.symbols('total_diffuse')
    e14 = sy.Eq(total_diffuse, a_multi_down_diffuse + a_multi_diffuse + r_multi_diffuse)
    equations_diffuse = (e1, e2, e3a, e3b, e4, e5, e6, e11, e12, e13, e14)
    solutions_diffuse = sy.solve(equations_diffuse, dict=True, manual=True)
    pp.pprint(solutions_diffuse)


if __name__ == "__main__":
    symbolic_verfication_2()





