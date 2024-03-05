import numpy as np
import sympy as sp

x1, x3, x2, x4, m1, m2, g, l1, l2 = sp.symbols('x1 x3 x2 x4 m1 m2 g l1 l2')
expr_x2 = (-g*(2*m1+m2)*sp.sin(x1) - m2*g*sp.sin(x1-2*x3) - 2*sp.sin(x1-x3)*m2*((x4**2)*l2 + (x2**2)*l1*sp.cos(x1-x3)))/(l1*(2*m1+m2-m2*sp.cos(2*x1-2*x3)))
expr_x4 = (2*sp.sin(x1-x3)*((x2**2)*l1*(m1+m2) + g*(m1+m2)*sp.cos(x1) + (x4**2)*l2*m2*sp.cos(x1-x3)))/(l2*(2*m1+m2-m2*sp.cos(2*x1-2*x3)))

dx2_dx1 = sp.diff(expr_x2, x1)
dx2_dx2 = sp.diff(expr_x2, x2)
dx2_dx3 = sp.diff(expr_x2, x3)
dx2_dx4 = sp.diff(expr_x2, x4)

dx4_dx1 = sp.diff(expr_x4, x1)
dx4_dx2 = sp.diff(expr_x4, x2)
dx4_dx3 = sp.diff(expr_x4, x3)
dx4_dx4 = sp.diff(expr_x4, x4)


# Display results
print("dx2_dx1")
simplified_derivative = sp.simplify(dx2_dx1)
print(simplified_derivative)

print("dx2_dx2")
simplified_derivative = sp.simplify(dx2_dx2)
print(simplified_derivative)

print("dx2_dx3")
simplified_derivative = sp.simplify(dx2_dx3)
print(simplified_derivative)

print("dx2_dx4")
simplified_derivative = sp.simplify(dx2_dx4)
print(simplified_derivative)



print("dx4_dx1")
simplified_derivative = sp.simplify(dx4_dx1)
print(simplified_derivative)

print("dx4_dx2")
simplified_derivative = sp.simplify(dx4_dx2)
print(simplified_derivative)

print("dx4_dx3")
simplified_derivative = sp.simplify(dx4_dx3)
print(simplified_derivative)

print("dx4_dx4")
simplified_derivative = sp.simplify(dx4_dx4)
print(simplified_derivative)