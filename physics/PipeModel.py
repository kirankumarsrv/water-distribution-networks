from scipy import *
import math
import numpy as np
from scipy.optimize import root

'''
Parameters:
L -> Length
D -> Diameter
f -> Friction Factor
A_leak -> Crack Area
Cd -> Discharge Coefficient
'''

class PipeLeakModel:
    def __init__(self,L,D,f,A_leak=0.0,Cd=0.0,x_leak=None):
        if L <= 0:
            raise ValueError("Length must be positive")
        if D <=0:
            raise ValueError("Diameter must be positive")
        if f <=0:
            raise ValueError("Friction Factor must be positive")
        
        
        self.L = L
        self.D = D
        self.f = f
        self.A_leak = A_leak
        self.Cd = Cd
        if x_leak == None:
            self.x_leak = L / 2
        else:
            self.x_leak = x_leak

        margin = 0.01 * L
        self.x_leak = min(max(self.x_leak, margin), L - margin)
        
        self.g = 9.81
        self.epsilon = 1e-6
        self.R = self.compute_R()

    def compute_R(self):
        R = (8 * self.f * self.L) / (math.pi**2 * self.g * self.D**5)
        return R
    
    def compute_Leakage(self,H):
        Q = self.Cd * self.A_leak * math.sqrt(2 * self.g * (H + self.epsilon))
        return Q
    
    def residuals(self,vars,H_in,H_out):
        Q1,Hm = vars
        Q_leak = self.compute_Leakage(Hm)
        Q2 = Q1 - Q_leak
        R1 = self.R * ((self.x_leak) / self.L)
        R2 = self.R * ((self.L - self.x_leak) / self.L)
        F1 = H_in - Hm - R1 * Q1 * abs(Q1)
        F2 = Hm - H_out - R2 * Q2 * abs(Q2)
        return [F1,F2]
    
    def solve_model(self,H_in,H_out,max_iter=1000,tol=1e-6):
        Q1 = math.sqrt(abs(H_in - H_out) / self.R)
        Hm = (H_in + H_out) / 2
        R1 = self.R * ((self.x_leak) / self.L)
        R2 = self.R * ((self.L - self.x_leak) / self.L)
        for _ in range(max_iter):
            F1,F2 = self.residuals([Q1,Hm],H_in,H_out)
            Q_leak = self.compute_Leakage(Hm)
            Q2 = Q1 - Q_leak
            print("Computing the Jacobian....")
            dF1_dQ1 = -2 * R1 * abs(Q1)
            dF1_dHm = -1
            dF2_dQ1 = -2 * R2 * abs(Q2)
            dQl_dHm = (self.Cd * self.A_leak * math.sqrt(2 * self.g)) / (2 * math.sqrt(Hm + self.epsilon))
            dF2_dHm = 1 - 2 * R2 * abs(Q2) * (-dQl_dHm)
            J = [[dF1_dQ1, dF1_dHm],
                 [dF2_dQ1, dF2_dHm]]
            print("Solving Newton Equation....")
            delta = np.linalg.solve(J,[F1,F2])

            Q1 -= delta[0]
            Hm -= delta[1]

            if abs(delta[0]) < tol and abs(delta[1]) < tol :
                print("Covergence Reached, Exiting the Solver.....")
                break
        Q_leak = self.compute_Leakage(Hm)

        # physical cap
        Q_leak = min(Q_leak, 0.8 * abs(Q1))
        Q2 = Q1 - np.sign(Q1)*Q_leak
        return Q1, Q2, Q_leak, Hm
