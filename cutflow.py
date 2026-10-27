import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate


class Filter:

    def __init__(self, df_sig=None, df_bkg=None, weight_col='flat_weight'):
        self.cut_history = []
        self.cutflow_table = None
        self.weight_col = weight_col

        if df_sig is not None and df_bkg is not None:
            if not ( (self.weight_col in df_sig.columns) and (self.weight_col in df_bkg.columns) ):
                raise ValueError(f"None of the given Dataframs have the weight column: {self.weight_col}")

            self.df_sig = df_sig.copy()
            self.df_bkg = df_bkg.copy()
            self.total_sig_weight = self.df_sig[self.weight_col].sum()
            self.total_bkg_weight = self.df_bkg[self.weight_col].sum()
            self.total_sig_weight2 = (self.df_sig[self.weight_col] ** 2).sum()
            self.total_bkg_weight2 = (self.df_bkg[self.weight_col] ** 2).sum()
            self.total_sig_events = len(self.df_sig)
            self.total_bkg_events = len(self.df_bkg)
        else:
            self.df_sig = None
            self.df_bkg = None
            self.total_sig_weight = 0.0
            self.total_bkg_weight = 0.0
            self.total_sig_weight2 = 0.0
            self.total_bkg_weight2 = 0.0
            self.total_sig_events = 0
            self.total_bkg_events = 0

    @classmethod
    def from_csv(cls, path):
        obj = cls(None, None)               # bypass df_sig/df_bkg needed
        obj.cutflow_table  = pd.read_csv(path)
        return obj

    def apply_cut(self, cut_expression: str, cut_name: str = None):
        if cut_name is None:
            cut_name = str(cut_expression)

        try:
            wsum_sig_before = self.df_sig[self.weight_col].sum()
            wsum_bkg_before = self.df_bkg[self.weight_col].sum()
            wsum2_sig_before = (self.df_sig[self.weight_col] ** 2).sum()
            wsum2_bkg_before = (self.df_bkg[self.weight_col] ** 2).sum()
            n_sig_before = len(self.df_sig)
            n_bkg_before = len(self.df_bkg)

            self.df_sig = self.df_sig.query(cut_expression)
            self.df_bkg = self.df_bkg.query(cut_expression)

            wsum_sig_after = self.df_sig[self.weight_col].sum()
            wsum_bkg_after = self.df_bkg[self.weight_col].sum()
            wsum2_sig_after = (self.df_sig[self.weight_col] ** 2).sum()
            wsum2_bkg_after = (self.df_bkg[self.weight_col] ** 2).sum()
            n_sig_after = len(self.df_sig)
            n_bkg_after = len(self.df_bkg)
      
            step_eff_sig     = wsum_sig_after / wsum_sig_before if wsum_sig_before > 0 else 0
            step_eff_bkg     = wsum_bkg_after / wsum_bkg_before if wsum_bkg_before > 0 else 0
            step_eff_sig_raw = n_sig_after / n_sig_before if n_sig_before > 0 else 0
            step_eff_bkg_raw = n_bkg_after / n_bkg_before if n_bkg_before > 0 else 0

            self.cut_history.append({
                "name"             : cut_name,
                "expression"       : cut_expression,
                "wsum_sig_before"  : wsum_sig_before,    "wsum_sig_after"   : wsum_sig_after,
                "wsum2_sig_before" : wsum2_sig_before,   "wsum2_sig_after"  : wsum2_sig_after,
                "wsum_bkg_before"  : wsum_bkg_before,    "wsum_bkg_after"   : wsum_bkg_after,
                "wsum2_bkg_before" : wsum2_bkg_before,   "wsum2_bkg_after"  : wsum2_bkg_after,
                "n_sig_before"     : n_sig_before,       "n_sig_after"      : n_sig_after,
                "n_bkg_before"     : n_bkg_before,       "n_bkg_after"      : n_bkg_after,
                "step_eff_sig"     : step_eff_sig,       "step_eff_bkg"     : step_eff_bkg,
                "step_eff_sig_raw" : step_eff_sig_raw,   "step_eff_bkg_raw" : step_eff_bkg_raw,
            })

        except Exception as e:
            raise RuntimeError(f"Error applying cut '{cut_expression}': {e}") from e

        return self

    def apply_cuts(self, cuts_dict):
        for name, expr in cuts_dict.items():
            self.apply_cut(expr, name)
        return self
    
    def _build_table(self):
        rows = [[
            "Initial", " -- ",
            self.total_sig_weight, self.total_sig_events, 1.0, 1.0, 1.0, 1.0,
            self.total_bkg_weight, self.total_bkg_events, 0.0, 0.0, 0.0, 0.0,
            self.total_sig_weight2, self.total_bkg_weight2,
            np.sqrt(self.total_sig_weight2), np.sqrt(self.total_bkg_weight2),
            0.0, 0.0, 0.0, 0.0,
        ]]
        for cut in self.cut_history:
            cum_eff_sig     = cut["wsum_sig_after"] / self.total_sig_weight if self.total_sig_weight > 0 else 0
            cum_eff_sig_raw = cut["n_sig_after"]    / self.total_sig_events if self.total_sig_events > 0 else 0
            cum_rej_bkg     = 1 - cut["wsum_bkg_after"] / self.total_bkg_weight if self.total_bkg_weight > 0 else 0
            cum_rej_bkg_raw = 1 - cut["n_bkg_after"]    / self.total_bkg_events if self.total_bkg_events > 0 else 0

            # Step efficiency: ε = wsum_after / wsum_before
            # Variance: Var(ε) = (1 - ε) * (Σ_{pass} w_i²) / (Σ_{all} w_i)²
            # σ(ε) = sqrt( (1 - ε) * wsum2_after / wsum_before² )
            if cut["wsum_sig_before"] > 0 and 0 <= cut["step_eff_sig"] <= 1:
                err_step_sig = np.sqrt((1 - cut["step_eff_sig"]) * cut["wsum2_sig_after"] / (cut["wsum_sig_before"]**2))
            else:
                err_step_sig = 0.0

            if cut["wsum_bkg_before"] > 0 and 0 <= cut["step_eff_bkg"] <= 1:
                err_step_bkg = np.sqrt((1 - cut["step_eff_bkg"]) * cut["wsum2_bkg_after"] / (cut["wsum_bkg_before"]**2))
            else:
                err_step_bkg = 0.0

            # Cumulative efficiency: ε_cum = wsum_after / total_wsum
            # Variance: Var(ε_cum) = (1 - ε_cum) * (Σ_{pass} w_i²) / (Σ_{initial} w_i)²
            # σ(ε_cum) = sqrt( (1 - ε_cum) * wsum2_after / total_wsum² )
            if self.total_sig_weight > 0 and 0 <= cum_eff_sig <= 1:
                err_cum_sig = np.sqrt((1 - cum_eff_sig) * cut["wsum2_sig_after"] / (self.total_sig_weight**2))
            else:
                err_cum_sig = 0.0

            # Cumulative rejection error is the same as cumulative efficiency error for background
            # σ(1-ε_cum) = sqrt( (1 - ε_cum) * wsum2_after / total_wsum² ) = sqrt( cum_rej_bkg * wsum2_after / total_wsum² )
            if self.total_bkg_weight > 0 and 0 <= (1 - cum_rej_bkg) <= 1:
                err_cum_rej = np.sqrt(cum_rej_bkg * cut["wsum2_bkg_after"] / (self.total_bkg_weight**2))
            else:
                err_cum_rej = 0.0

            rows.append([
                cut["name"],                        cut["expression"],
                cut["wsum_sig_after"],              cut["n_sig_after"],
                cut["step_eff_sig"],                cut["step_eff_sig_raw"],
                cum_eff_sig,                        cum_eff_sig_raw,
                cut["wsum_bkg_after"],              cut["n_bkg_after"],
                1 - cut["step_eff_bkg"],            1 - cut["step_eff_bkg_raw"],
                cum_rej_bkg,                        cum_rej_bkg_raw,
                cut["wsum2_sig_after"],             cut["wsum2_bkg_after"],
                np.sqrt(cut["wsum2_sig_after"]),     np.sqrt(cut["wsum2_bkg_after"]),
                err_step_sig,                       err_step_bkg,
                err_cum_sig,                        err_cum_rej,
            ])
        headers = [
            "name", "Expression",
            "Nsig (w)", "Nsig (n)", "ε_s (w)", "ε_s (n)", "ε_s cum (w)", "ε_s cum (n)",
            "Nbkg (w)", "Nbkg (n)", "1-ε_b (w)", "1-ε_b (n)", "1-ε_b cum (w)", "1-ε_b cum (n)",
            "Nsig2 (w)", "Nbkg2 (w)",
            "σ Nsig (w)", "σ Nbkg (w)",
            "σ ε_s (w)", "σ 1-ε_b (w)",
            "σ ε_s cum (w)", "σ 1-ε_b cum (w)",
        ]
        self.cutflow_table = pd.DataFrame(rows, columns=headers)
        return self.cutflow_table

    def print_cutflow_table(self, mode="weighted"):
        df = self._build_table() 

        def pm(w, e):
            return f"{w:.2f} ± {e:.2f}"

        if mode == "weighted":
            disp = pd.DataFrame({
                "name"          : df["name"],
                "Expression"    : df["Expression"],
                "Nsig (w)"      : [pm(w, e) for w, e in zip(df["Nsig (w)"],      df["σ Nsig (w)"])],
                "ε_s (w)"       : [pm(w, e) for w, e in zip(df["ε_s (w)"],       df["σ ε_s (w)"])],
                "ε_s cum (w)"   : [pm(w, e) for w, e in zip(df["ε_s cum (w)"],   df["σ ε_s cum (w)"])],
                "Nbkg (w)"      : [pm(w, e) for w, e in zip(df["Nbkg (w)"],      df["σ Nbkg (w)"])],
                "1-ε_b (w)"     : [pm(w, e) for w, e in zip(df["1-ε_b (w)"],     df["σ 1-ε_b (w)"])],
                "1-ε_b cum (w)" : [pm(w, e) for w, e in zip(df["1-ε_b cum (w)"], df["σ 1-ε_b cum (w)"])],
            })
        elif mode == "raw":
            disp = df[["name", "Expression", "Nsig (n)", "ε_s (n)", "ε_s cum (n)", "Nbkg (n)", "1-ε_b (n)", "1-ε_b cum (n)"]]
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        print("\n", f"Cutflow Table ({mode}):")
        print(tabulate(disp, headers='keys', tablefmt='fancy_grid'))
        return None

    def save_cutflow_table(self, path):
        if self.cutflow_table is None:
            self.cutflow_table = self._build_table()
        self.cutflow_table.to_csv(path, index=False, float_format="%.4f")

    def save_cutflow_table_latex(self, path, float_fmt=".2f"):

        def escape_latex(text):
            for char in ["\\", "&", "%", "$", "#", "_", "{", "}"]:
                text = text.replace(char, f"\\{char}")
            return text

        fmt = f"{{:{float_fmt}}}"
        df = self._build_table() 

        def pm(w, e):
            return f"${fmt.format(w)} \\pm {fmt.format(e)}$"

        rows = [[
            escape_latex(r["Expression"]),
            pm(r["Nsig (w)"],        r["σ Nsig (w)"]),
            pm(r["Nbkg (w)"],        r["σ Nbkg (w)"]),
            # pm(r["ε_s (w)"]*100,         r["σ ε_s (w)"]*100),
            pm(r["ε_s cum (w)"]*100,     r["σ ε_s cum (w)"]*100),
            # pm(r["1-ε_b (w)"]*100,       r["σ 1-ε_b (w)"]*100),
            pm(r["1-ε_b cum (w)"]*100,   r["σ 1-ε_b cum (w)"]*100),
        ] for _, r in df.iterrows()]

        headers = [
            "Cut",
            r"Signal ($w \pm \sigma$)",
            r"Background ($w \pm \sigma$)",
            # r"$\varepsilon_s$ (\%)",
            r"$\varepsilon_s^{\rm cum}$ (\%)",
            # r"$1-\varepsilon_b$ (\%)",
            r"$1-\varepsilon_b^{\rm cum}$ (\%)",
        ]
        pd.DataFrame(rows, columns=headers).to_latex(path, index=False, escape=False)
        return None

    def read_cutflow_table(self, path, set_table=True):
        """
        Read a cutflow table previously saved with save_cutflow_table().
        Returns the DataFrame (and by default stores it in self.cutflow_table.
        """
        df = pd.read_csv(path)
        if set_table:
            self.cutflow_table = df
        return df

    def get_dataframes(self):
        return self.df_sig.reset_index(drop=True), self.df_bkg.reset_index(drop=True)
