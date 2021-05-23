from dataclasses import dataclass
from pprint import pprint

import numpy as np
from PyIF import te_compute as te


@dataclass
class ThetaInit:
    def __init__(self, u2ss, BW, TDI):
        # constants
        self.u2ss = u2ss
        self.BW = BW
        self.TDI = TDI[0]
        self.aIOB = 5.0

        self.d = int(21 / 3)

        self.Wh = 0.1
        self.Wl = -0.2

        # signals
        self.g_sig = []

    def send_glucose(self, g):
        self.g_sig.append(g)

    def calculate_theta(self):
        g_sig = np.array(self.g_sig)

        basal = self._calc_basal()
        u0 = self._calc_u0(basal)
        IOBbasal = self._calc_IOBbasal(u0)
        IOB_TDI = self._calc_IOB_TDI()

        dg_sig = self._calc_dg()
        d2g_sig = self._calc_d2g(dg_sig)
        IOB_max = self._calc_IOBmax(IOBbasal, IOB_TDI, g_sig, dg_sig, d2g_sig)
        IA = IOB_max + basal

        g_sig = g_sig[self.d:]
        IA = IA[:len(g_sig)]

        TE = te.te_compute(IA, g_sig, k=1, embedding=1, safetyCheck=False, GPU=False)
        return self.Wh / TE, self.Wl / TE

    def _calc_basal(self):
        return self.u2ss * self.BW / 6000 * 60

    def _calc_u0(self, basal):
        if basal >= 1.25:
            return 0.85 * basal
        if self.g_sig[0] >= 100:
            return 1 * basal
        if self.g_sig[0] < 100:
            return 0.75 * basal

        raise Exception("no conditions matched")

    def _calc_IOBbasal(self, u0):
        return self.aIOB * u0

    def _calc_IOB_TDI(self):
        if self.TDI <= 25:
            return 0.11
        if 25 < self.TDI <= 35:
            return 0.125
        if 35 < self.TDI <= 45:
            return 0.12
        if 45 < self.TDI <= 55:
            return 0.175
        if 55 < self.TDI:
            return 0.2

        raise Exception("no conditions matched")

    def _calc_dg(self):
        return np.diff(self.g_sig) / 3

    def _calc_d2g(self, dg_sig):
        return np.diff(dg_sig) / 3

    def _calc_IOBmax(self, IOBbasal, IOB_TDI, g_sig, dg_sig, d2g_sig):
        g_sig = g_sig[2:]
        dg_sig = dg_sig[1:]

        def calc(g, dg, ddg):
            if g < 125:
                return 1.10 * IOBbasal

            if 150 <= g and dg > 0.25 and ddg > 0.035:
                return max(IOB_TDI, 2.5 * IOBbasal)

            if 175 <= g and dg > 0.35 and ddg > 0.035:
                return max(IOB_TDI, 3.5 * IOBbasal)

            if 200 <= g and dg > -0.05:
                return max(IOB_TDI, 3.5 * IOBbasal)

            if 200 <= g and dg > 0.15:
                return max(IOB_TDI, 4.5 * IOBbasal)

            if 200 <= g and dg > 0.3:
                return max(IOB_TDI, 6.5 * IOBbasal)

            if self.TDI < 30:
                return 0.95 * IOBbasal

            if 125 <= g:
                return 1.35 * IOBbasal

            raise Exception("no conditions matched")

        return np.array([calc(g, dg, ddg) for g, dg, ddg in zip(g_sig, dg_sig, d2g_sig)])
