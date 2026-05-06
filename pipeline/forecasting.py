"""
Forecasting engine — flexible fit + walk-forward backtest + GARCH risk metrics.

Provides three core capabilities used by the dashboard's Forecasts tab:
    1. fit_arimax / fit_ardl — flexible model fits with user-controllable orders
    2. walk_forward_backtest — rolling out-of-sample evaluation
    3. fit_garch_with_risk — GARCH fit + VaR / ES / regime classification

All functions return dataclasses with rich metadata. They handle convergence
failures gracefully — bad parameter combinations return a result with
.success=False and a human-readable error message instead of raising.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Tuple
import numpy as np
import pandas as pd


# ---------- Result containers ----------

@dataclass
class FitResult:
    """Output of a single model fit."""
    success: bool
    model_type: str                              # 'arimax' or 'ardl'
    error_msg: str = ""

    # Fitted values + forecasts
    fitted_in_sample: Optional[pd.Series] = None
    forecast_mean: Optional[pd.Series] = None
    forecast_lower_95: Optional[pd.Series] = None
    forecast_upper_95: Optional[pd.Series] = None

    # Coefficients
    coefficients: Optional[pd.DataFrame] = None  # cols: param, coef, std_err, p_value

    # In-sample metrics
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    n_obs: Optional[int] = None

    # Residual diagnostics
    ljung_box_p: Optional[float] = None          # p > 0.05 = no autocorrelation (good)
    residuals: Optional[pd.Series] = None

    # Config used
    config_summary: str = ""                     # human-readable summary


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest run."""
    success: bool
    error_msg: str = ""

    # Per-window forecasts and actuals
    fold_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    # cols: forecast_origin, target_date, h_months, forecast, actual, error, abs_error_pct

    # Aggregate metrics by horizon
    metrics_by_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    # cols: h_months, n, rmse, mape, hit_rate, worst_error_pct

    # Overall metrics across all folds + horizons
    overall_rmse: Optional[float] = None
    overall_mape: Optional[float] = None
    overall_hit_rate: Optional[float] = None     # % of forecasts that got direction right
    n_folds: int = 0


@dataclass
class GarchResult:
    """Output of GARCH fit with risk metrics."""
    success: bool
    error_msg: str = ""

    # Fit details
    alpha: Optional[float] = None
    beta: Optional[float] = None
    persistence: Optional[float] = None          # alpha + beta
    half_life_months: Optional[float] = None

    # Conditional volatility
    conditional_vol: Optional[pd.Series] = None  # in-sample volatility series
    forecast_vol: Optional[pd.Series] = None     # forecast volatility

    # Risk metrics (1-month ahead)
    var_95: Optional[float] = None               # Value-at-Risk 5% (price units)
    var_99: Optional[float] = None               # Value-at-Risk 1%
    expected_shortfall_95: Optional[float] = None
    expected_shortfall_99: Optional[float] = None

    # Volatility regime
    current_vol: Optional[float] = None
    vol_percentile: Optional[float] = None       # current vol's percentile in history
    regime_label: str = ""                       # 'low', 'normal', 'elevated', 'extreme'


# ---------- Helpers ----------

def _safe_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float, float]:
    """Returns (rmse, mape, r2). Handles edge cases."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    a = actual[mask]; p = predicted[mask]
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    nonzero = a != 0
    mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100) if nonzero.any() else float("nan")
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, mape, r2


def _hold_last_X(X: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build a future X by repeating the last observed row 'horizon' times.
    Honest assumption when we don't know the future drivers."""
    last_row = X.iloc[-1].copy()
    last_idx = X.index[-1]
    future_idx = pd.date_range(
        start=last_idx + pd.tseries.offsets.MonthBegin(1),
        periods=horizon, freq="MS"
    )
    return pd.DataFrame([last_row.values] * horizon,
                          index=future_idx, columns=X.columns)


# ---------- ARIMAX ----------

def fit_arimax(target: pd.Series, X: pd.DataFrame,
                ar: int = 2, d: int = 1, ma: int = 1,
                horizon: int = 12,
                drivers: Optional[List[str]] = None) -> FitResult:
    """
    Fit ARIMAX model with full control over orders + driver inclusion.
    """
    cfg = f"ARIMAX({ar},{d},{ma}) + {len(drivers) if drivers else len(X.columns)} drivers, h={horizon}"

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return FitResult(success=False, model_type="arimax",
                          error_msg="statsmodels not installed", config_summary=cfg)

    # Filter drivers if requested
    if drivers is not None:
        usable = [d for d in drivers if d in X.columns]
        if len(usable) == 0:
            return FitResult(success=False, model_type="arimax",
                              error_msg="No drivers selected", config_summary=cfg)
        X_use = X[usable]
    else:
        X_use = X

    try:
        # Align target + X
        common_idx = target.index.intersection(X_use.index)
        y = target.loc[common_idx].astype(float)
        Xa = X_use.loc[common_idx].astype(float)

        if len(y) < (ar + d + ma + len(Xa.columns) + 5):
            return FitResult(success=False, model_type="arimax",
                              error_msg=f"Not enough data: need at least "
                                        f"{ar + d + ma + len(Xa.columns) + 5} obs, have {len(y)}",
                              config_summary=cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, exog=Xa, order=(ar, d, ma),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200, method="lbfgs")

        # In-sample fitted values
        in_sample = fit.fittedvalues
        # Align y to fitted values' index (differencing may drop initial obs)
        y_aligned = y.loc[in_sample.index] if hasattr(in_sample, "index") else y

        # Forecast
        X_future = _hold_last_X(Xa, horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = fit.get_forecast(steps=horizon, exog=X_future)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        # Index to future dates
        mean.index = X_future.index; lower.index = X_future.index; upper.index = X_future.index

        rmse, mape, r2 = _safe_metrics(y_aligned.values, in_sample.values)
        residuals = y_aligned - in_sample

        # Coefficient table
        coef_df = pd.DataFrame({
            "param": fit.params.index,
            "coef": fit.params.values,
            "std_err": fit.bse.values,
            "p_value": fit.pvalues.values,
        })

        # Ljung-Box on residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
            ljung_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            ljung_p = None

        return FitResult(
            success=True, model_type="arimax",
            fitted_in_sample=in_sample, forecast_mean=mean,
            forecast_lower_95=lower, forecast_upper_95=upper,
            coefficients=coef_df,
            rmse=rmse, mape=mape, r2=r2,
            aic=float(fit.aic), bic=float(fit.bic), n_obs=len(y),
            ljung_box_p=ljung_p, residuals=residuals,
            config_summary=cfg,
        )
    except Exception as e:
        return FitResult(success=False, model_type="arimax",
                          error_msg=f"{type(e).__name__}: {str(e)[:200]}",
                          config_summary=cfg)


# ---------- ARDL ----------

def fit_ardl(target: pd.Series, X: pd.DataFrame,
              ar: int = 2, dl: int = 2,
              horizon: int = 12,
              drivers: Optional[List[str]] = None) -> FitResult:
    """
    Fit ARDL model with user control over AR + distributed-lag orders.
    """
    cfg = f"ARDL(ar={ar}, dl={dl}) + {len(drivers) if drivers else len(X.columns)} drivers, h={horizon}"

    try:
        from statsmodels.tsa.api import ARDL
    except ImportError:
        return FitResult(success=False, model_type="ardl",
                          error_msg="statsmodels.tsa.api.ARDL unavailable",
                          config_summary=cfg)

    if drivers is not None:
        usable = [d for d in drivers if d in X.columns]
        if len(usable) == 0:
            return FitResult(success=False, model_type="ardl",
                              error_msg="No drivers selected", config_summary=cfg)
        X_use = X[usable]
    else:
        X_use = X

    try:
        common_idx = target.index.intersection(X_use.index)
        y = target.loc[common_idx].astype(float)
        Xa = X_use.loc[common_idx].astype(float)

        # ARDL needs more data when DL is large
        min_obs = ar + dl * len(Xa.columns) + 8
        if len(y) < min_obs:
            return FitResult(success=False, model_type="ardl",
                              error_msg=f"Not enough data: need at least {min_obs} obs, have {len(y)}",
                              config_summary=cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARDL(y, lags=ar, exog=Xa, order=dl)
            fit = model.fit()

        in_sample = fit.fittedvalues
        # Align y to fitted values' index (AR lags drop initial obs)
        y_aligned = y.loc[in_sample.index] if hasattr(in_sample, "index") else y

        # Forecast: ARDL needs future X — hold last
        X_future = _hold_last_X(Xa, horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = fit.forecast(steps=horizon, exog=X_future)
        # ARDL doesn't natively give CI on forecast; compute manually from residual std
        resid_std = float(np.std(y_aligned - in_sample))
        z95 = 1.96
        # Convert fc to a Series with future index regardless of native type
        if hasattr(fc, "values"):
            mean = pd.Series(fc.values, index=X_future.index)
        else:
            mean = pd.Series(np.asarray(fc), index=X_future.index)
        lower = mean - z95 * resid_std
        upper = mean + z95 * resid_std

        rmse, mape, r2 = _safe_metrics(y_aligned.values, in_sample.values)
        residuals = y_aligned - in_sample

        coef_df = pd.DataFrame({
            "param": fit.params.index,
            "coef": fit.params.values,
            "std_err": fit.bse.values,
            "p_value": fit.pvalues.values,
        })

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
            ljung_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            ljung_p = None

        return FitResult(
            success=True, model_type="ardl",
            fitted_in_sample=in_sample, forecast_mean=mean,
            forecast_lower_95=lower, forecast_upper_95=upper,
            coefficients=coef_df,
            rmse=rmse, mape=mape, r2=r2,
            aic=float(fit.aic), bic=float(fit.bic), n_obs=len(y),
            ljung_box_p=ljung_p, residuals=residuals,
            config_summary=cfg,
        )
    except Exception as e:
        return FitResult(success=False, model_type="ardl",
                          error_msg=f"{type(e).__name__}: {str(e)[:200]}",
                          config_summary=cfg)


# ---------- Walk-forward backtest ----------

def walk_forward_backtest(target: pd.Series, X: pd.DataFrame,
                            model_type: str = "arimax",
                            ar: int = 2, d: int = 1, ma: int = 1, dl: int = 2,
                            drivers: Optional[List[str]] = None,
                            forecast_horizon: int = 6,
                            min_train_months: int = 60,
                            step_months: int = 1) -> BacktestResult:
    """
    Walk-forward (expanding window) backtest. For each origin t, train on
    [start, t], forecast h months ahead, compare to actual.

    Returns evaluation metrics by horizon and overall.
    """
    if drivers is not None:
        X_use = X[[c for c in drivers if c in X.columns]]
    else:
        X_use = X

    common_idx = target.index.intersection(X_use.index)
    y = target.loc[common_idx]
    Xa = X_use.loc[common_idx]

    if len(y) < min_train_months + forecast_horizon + 1:
        return BacktestResult(
            success=False,
            error_msg=f"Not enough data for backtest: need ≥{min_train_months + forecast_horizon + 1} months, "
                      f"have {len(y)}. Reduce horizon or min_train_months."
        )

    # Determine origin points: last possible origin = total - horizon
    last_origin_idx = len(y) - forecast_horizon - 1
    origin_indices = list(range(min_train_months - 1, last_origin_idx + 1, step_months))

    fold_records = []
    for oi in origin_indices:
        train_y = y.iloc[: oi + 1]
        train_X = Xa.iloc[: oi + 1]
        try:
            if model_type == "arimax":
                fit_res = fit_arimax(train_y, train_X, ar=ar, d=d, ma=ma,
                                       horizon=forecast_horizon, drivers=drivers)
            else:
                fit_res = fit_ardl(train_y, train_X, ar=ar, dl=dl,
                                     horizon=forecast_horizon, drivers=drivers)

            if not fit_res.success or fit_res.forecast_mean is None:
                continue
            fc = fit_res.forecast_mean

            # Compare each forecast horizon step to actual
            for h in range(1, forecast_horizon + 1):
                target_pos = oi + h
                if target_pos >= len(y):
                    continue
                actual = float(y.iloc[target_pos])
                # Forecast is 0-indexed in fc; first fc value is h=1
                if h - 1 >= len(fc):
                    continue
                forecast = float(fc.iloc[h - 1])
                error = forecast - actual
                abs_pct = abs(error / actual) * 100 if actual != 0 else float("nan")

                # Direction: did forecast correctly predict up/down vs origin?
                origin_val = float(y.iloc[oi])
                forecast_dir = np.sign(forecast - origin_val)
                actual_dir = np.sign(actual - origin_val)
                hit = bool(forecast_dir == actual_dir and forecast_dir != 0)

                fold_records.append({
                    "forecast_origin": y.index[oi],
                    "target_date": y.index[target_pos],
                    "h_months": h,
                    "forecast": forecast,
                    "actual": actual,
                    "error": error,
                    "abs_error_pct": abs_pct,
                    "hit": hit,
                })
        except Exception:
            continue

    if len(fold_records) == 0:
        return BacktestResult(success=False,
                                error_msg="All backtest folds failed to fit. "
                                          "Try simpler model orders or fewer drivers.")

    fold_df = pd.DataFrame(fold_records)

    # Per-horizon aggregates
    by_h = []
    for h in sorted(fold_df["h_months"].unique()):
        sub = fold_df[fold_df["h_months"] == h]
        rmse, mape, r2 = _safe_metrics(sub["actual"].values, sub["forecast"].values)
        by_h.append({
            "h_months": int(h),
            "n": int(len(sub)),
            "rmse": rmse,
            "mape": mape,
            "hit_rate": float(sub["hit"].mean() * 100),
            "worst_error_pct": float(sub["abs_error_pct"].max()),
        })
    by_h_df = pd.DataFrame(by_h)

    overall_rmse, overall_mape, _ = _safe_metrics(
        fold_df["actual"].values, fold_df["forecast"].values
    )
    overall_hit = float(fold_df["hit"].mean() * 100)

    return BacktestResult(
        success=True,
        fold_results=fold_df,
        metrics_by_horizon=by_h_df,
        overall_rmse=overall_rmse,
        overall_mape=overall_mape,
        overall_hit_rate=overall_hit,
        n_folds=int(fold_df["forecast_origin"].nunique()),
    )


# ---------- GARCH with risk metrics ----------

def fit_garch_with_risk(target: pd.Series,
                          horizon: int = 12,
                          alpha_levels: Tuple[float, float] = (0.05, 0.01)) -> GarchResult:
    """
    Fit GARCH(1,1) on log-returns. Compute conditional vol, forecast vol,
    VaR/ES at specified levels, and current volatility regime.
    """
    try:
        from arch import arch_model
    except ImportError:
        return GarchResult(success=False, error_msg="arch package not installed")

    s = target.dropna().sort_index()
    if len(s) < 24:
        return GarchResult(success=False,
                              error_msg=f"Need ≥24 obs for GARCH, have {len(s)}")

    returns = (np.log(s).diff() * 100).dropna()
    if returns.std() < 1e-6:
        return GarchResult(success=False, error_msg="Returns have no variation")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
            fit = model.fit(disp="off", show_warning=False)
        params = fit.params
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        persistence = alpha + beta
        if 0 < persistence < 1:
            half_life = float(np.log(0.5) / np.log(persistence))
        else:
            half_life = float("inf") if persistence >= 1 else None

        # Conditional vol (in-sample)
        cond_vol = pd.Series(np.sqrt(fit.conditional_volatility),
                                index=returns.index)

        # Forecast volatility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = fit.forecast(horizon=horizon, reindex=False)
        fc_var = fc.variance.values[-1]               # last row = forecast
        fc_vol = np.sqrt(fc_var)
        future_idx = pd.date_range(
            start=returns.index[-1] + pd.tseries.offsets.MonthBegin(1),
            periods=horizon, freq="MS",
        )
        forecast_vol_series = pd.Series(fc_vol, index=future_idx)

        # 1-month-ahead VaR + ES (in % return space, then convert to price)
        last_price = float(s.iloc[-1])
        sigma_1m = float(fc_vol[0]) / 100.0          # back to decimal
        from scipy.stats import norm
        var_metrics = {}
        for level in alpha_levels:
            z = norm.ppf(level)
            var_pct = z * sigma_1m                    # negative for downside
            var_price_loss = last_price * (1 - np.exp(var_pct))
            es_pct = (norm.pdf(z) / level) * sigma_1m
            es_price_loss = last_price * (1 - np.exp(-es_pct))
            var_metrics[level] = (var_price_loss, es_price_loss)

        # Volatility regime
        current_vol = float(cond_vol.iloc[-1])
        vol_pct = float((cond_vol < current_vol).mean() * 100)
        if vol_pct < 25:
            regime = "low"
        elif vol_pct < 65:
            regime = "normal"
        elif vol_pct < 90:
            regime = "elevated"
        else:
            regime = "extreme"

        return GarchResult(
            success=True,
            alpha=alpha, beta=beta,
            persistence=persistence,
            half_life_months=half_life,
            conditional_vol=cond_vol,
            forecast_vol=forecast_vol_series,
            var_95=var_metrics[0.05][0],
            var_99=var_metrics[0.01][0],
            expected_shortfall_95=var_metrics[0.05][1],
            expected_shortfall_99=var_metrics[0.01][1],
            current_vol=current_vol,
            vol_percentile=vol_pct,
            regime_label=regime,
        )
    except Exception as e:
        return GarchResult(success=False,
                              error_msg=f"{type(e).__name__}: {str(e)[:200]}")
