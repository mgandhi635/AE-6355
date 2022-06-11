import numpy as np
import pandas as pd

gamma_data = pd.read_excel('Curve_fit_constants_for_class_r2.xlsx', skiprows=1, sheet_name='h=h(p,rho)')
temp_data = pd.read_excel('Curve_fit_constants_for_class_r2.xlsx', skiprows=1, sheet_name='T=T(P,rho)')

# xl = pd.ExcelFile('Curve_fit_constants_for_class_r2.xlsx')
# df = xl.parse(skiprows=1)

# print(gamma_data)
# print(temp_data)


class GammaTildeCurveFit:
    def __init__(self):
        self.data = gamma_data

    def fit(self, p_2_est, rho_2_est):
        (X, Y, Z) = self.get_vars(p_2_est, rho_2_est)
        coeffs = self.get_coefficents(Y, Z)
        gamma_tilde = self.evaluate_curve(X, Y, Z, coeffs)
        return gamma_tilde, (X, Y, Z), coeffs

    @staticmethod
    def evaluate_curve(X, Y, Z, coeffs):
        num = coeffs[4] + coeffs[5]*Y + coeffs[6]*Z + coeffs[7]*Y*Z
        den = 1 + np.exp(coeffs[8]*(X + coeffs[9]*Y + coeffs[10]))
        return coeffs[0] + coeffs[1]*Y + coeffs[2]*Z + coeffs[3]*Y*Z + num/den

    def get_coefficents(self, Y, Z):
        row = self.get_coeff_row(Y, Z)
        return self.data.iloc[row, 2:-1]

    def get_vars(self, p_2_est, rho_2_est):
        X = self.compute_X(p_2_est)
        Y = self.compute_Y(rho_2_est)
        Z = self.compute_Z(X, Y)
        return (X, Y, Z)

    def get_coeff_row(self, Y, Z):
        if Y > -0.50:
            return self.find_z_range_1(Z)
        elif (-4.5 < Y) and (Y <= -0.5):
            return self.find_z_range_2(Z)
        elif (-7 < Y) and (Y <= -4.5):
            return self.find_z_range_3(Z)
        else:
            raise ValueError

    @staticmethod
    def find_z_range_1(Z):
        if Z <= 0.30:
            return 0
        elif (.30 < Z) and (Z <= 1.15):
            return 1
        elif (1.15 < Z) and (Z <= 1.60):
            return 2
        elif Z > 1.60:
            return 3
        else:
            raise ValueError

    @staticmethod
    def find_z_range_2(Z):
        if Z <= 0.30:
            return 4
        elif (.30 < Z) and (Z <= 0.98):
            return 5
        elif (0.98 < Z) and (Z <= 1.38):
            return 6
        elif (1.38 < Z) and (Z <= 2.04):
            return 7
        elif Z > 2.04:
            return 8
        else:
            raise ValueError

    @staticmethod
    def find_z_range_3(Z):
        if Z <= 0.398:
            return 9
        elif (.398 < Z) and (Z <= 0.87):
            return 10
        elif (0.87 < Z) and (Z <= 1.27):
            return 11
        elif (1.27 < Z) and (Z <= 1.863):
            return 12
        elif Z > 1.863:
            return 13
        else:
            raise ValueError

    @staticmethod
    def compute_Y(rho_2_est):
        return np.log10(rho_2_est / 1.292)

    @staticmethod
    def compute_X(p_2_est):
        return np.log10(p_2_est / 1.013e5)

    @staticmethod
    def compute_Z(X, Y):
        return X - Y


class HighTempCurveFit:
    def __init__(self):
        self.data = temp_data

    def fit(self, p_2_est, rho_2_est):
        (X, Y, Z) = self.get_vars(p_2_est, rho_2_est)
        coeffs, T0 = self.get_coefficents(Y, Z)
        curve_out = self.evaluate_curve(X, Y, Z, coeffs)
        temp_ratio = 10 ** (curve_out)
        temp = temp_ratio * T0
        return temp, (X, Y, Z), coeffs, T0

    @staticmethod
    def evaluate_curve(X, Y, Z, coeffs):
        num = coeffs[5] + coeffs[6]*Y + coeffs[7]*Z + coeffs[8]*Y*Z + coeffs[9]*Z*Z
        den = 1 + np.exp(coeffs[10]*(Z + coeffs[11]))
        return coeffs[0] + coeffs[1]*Y + coeffs[2]*Z + coeffs[3]*Y*Z + coeffs[4]*Z*Z + num/den

    def get_coefficents(self, Y, Z):
        row, T0 = self.get_coeff_row(Y, Z)
        return self.data.iloc[row, 2:-1], T0

    def get_vars(self, p_2_est, rho_2_est):
        X = self.compute_X(p_2_est)
        Y = self.compute_Y(rho_2_est)
        Z = self.compute_Z(X, Y)
        return (X, Y, Z)

    def get_coeff_row(self, Y, Z):
        T0_1 = 151.78  # Kelvin
        T0_2 = 0.55556  # Kelvin
        if Y > -0.50:
            return self.find_z_range_1(Z), T0_1
        elif (-4.5 < Y) and (Y <= -0.5):
            return self.find_z_range_2(Z), T0_1
        elif (-7 < Y) and (Y <= -4.5):
            return self.find_z_range_3(Z), T0_2
        else:
            raise ValueError

    @staticmethod
    def find_z_range_1(Z):
        if (0.48 < Z) and (Z <= 0.90):
            return 0
        elif Z > 0.9:
            return 1
        else:
            raise ValueError

    @staticmethod
    def find_z_range_2(Z):
        if (.48 < Z) and (Z <= 0.9165):
            return 2
        elif (0.9165 < Z) and (Z <= 1.478):
            return 3
        elif (1.478 < Z) and (Z <= 2.176):
            return 4
        elif Z > 2.176:
            return 5
        else:
            raise ValueError

    @staticmethod
    def find_z_range_3(Z):
        if (0.3 < Z) and (Z <= 1.07):
            return 6
        elif (1.07 < Z) and (Z <= 1.57):
            return 7
        elif (1.57 < Z) and (Z <= 2.24):
            return 8
        elif Z > 2.24:
            return 9
        else:
            raise ValueError

    @staticmethod
    def compute_X(p_2_est):
        return np.log10(p_2_est / 1.0134e5)

    @staticmethod
    def compute_Y(rho_2_est):
        return np.log10(rho_2_est / 1.225)

    @staticmethod
    def compute_Z(X, Y):
        return X - Y


if __name__ == "__main__":
    p_2_est = 4004.7616416966903
    rho_2_est = 3.99109e-5  # kg / m^3

    gamma_curve = GammaTildeCurveFit()
    temp_curve = HighTempCurveFit()

    gamma = gamma_curve.fit(p_2_est, rho_2_est)
    temp = temp_curve.fit(p_2_est, rho_2_est)
    print(gamma)
    print(temp)
