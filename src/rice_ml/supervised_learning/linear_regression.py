import numpy as np
from scipy import stats


class LinearRegression:
    """
    Ordinary Least Squares (OLS) linear regression, implemented from scratch.

    Usage:
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model.summary(feature_names=[...])
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

        # Learned parameters
        self.coef_: np.ndarray | None = None      # shape (n_features,)
        self.intercept_: float | None = None      # scalar

        # Internal attributes for statistics
        self._beta: np.ndarray | None = None      # shape (p,), includes intercept if used
        self._X: np.ndarray | None = None         # design matrix used in fit (with or without intercept)
        self._y: np.ndarray | None = None         # target vector (n,1)
        self._y_hat: np.ndarray | None = None     # fitted values (n,1)
        self._n: int | None = None                # number of observations
        self._p: int | None = None                # number of parameters (including intercept)

    # ---------- internal helpers ----------

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """
        Ensure X is a 2D numpy array and, if fit_intercept=True,
        prepend a column of ones for the intercept term.
        """
        X = np.asarray(X, dtype=float)

        # (n,) -> (n,1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.hstack([ones, X])
        else:
            return X

    def _check_is_fitted(self) -> None:
        if self._beta is None or self._X is None or self._y is None:
            raise ValueError("Model is not fitted yet. Call `fit` first.")

    # ---------- core API ----------

    def fit(self, X, y):
        """
        Fit the linear regression model using the closed-form OLS solution:

            beta_hat = (X^T X)^{-1} X^T y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) or (n_samples, 1)
        """
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        self._X = X
        self._y = y
        self._n, self._p = X.shape

        # OLS closed form: beta_hat = (X^T X)^(-1) X^T y
        XtX = X.T @ X                             # (p, p)
        XtX_inv = np.linalg.inv(XtX)              # (p, p)
        Xty = X.T @ y                             # (p, 1)

        beta = XtX_inv @ Xty                      # (p, 1)
        self._beta = beta.reshape(-1)             # (p,)

        # Split into intercept + coefficients
        if self.fit_intercept:
            self.intercept_ = float(self._beta[0])
            self.coef_ = self._beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._beta

        # Fitted values
        self._y_hat = X @ beta                    # (n, 1)

        return self

    def predict(self, X):
        """
        Predict y for new data X using the learned parameters.
        """
        self._check_is_fitted()
        X_new = self._prepare_X(X)
        y_pred = X_new @ self._beta.reshape(-1, 1)    # (n,1)
        return y_pred.ravel()                         # return 1D array

    # ---------- basic residuals & errors ----------

    def residuals(self) -> np.ndarray:
        """
        Return residuals: y - y_hat
        """
        self._check_is_fitted()
        return (self._y - self._y_hat).ravel()

    def SSE(self) -> float:
        """
        Sum of Squared Errors (Residual Sum of Squares)
        """
        r = self.residuals()
        return float(np.sum(r ** 2))

    def SST(self) -> float:
        """
        Total Sum of Squares around the mean of y
        """
        self._check_is_fitted()
        y = self._y.ravel()
        return float(np.sum((y - y.mean()) ** 2))

    def MSE(self) -> float:
        """
        Mean Squared Error, using n - p degrees of freedom
        """
        self._check_is_fitted()
        return self.SSE() / (self._n - self._p)

    def RMSE(self) -> float:
        """
        Root Mean Squared Error
        """
        return float(np.sqrt(self.MSE()))

    def R2(self) -> float:
        """
        Coefficient of determination R^2
        """
        return float(1.0 - self.SSE() / self.SST())

    def adj_R2(self) -> float:
        """
        Adjusted R^2 that penalizes extra parameters
        """
        R2 = self.R2()
        n, p = self._n, self._p
        return float(1.0 - (1.0 - R2) * (n - 1) / (n - p))

    # ---------- coefficient inference: SE, t-stats, F-test ----------

    def coef_se(self) -> np.ndarray:
        """
        Standard error of each coefficient.

        Var(beta_hat) = sigma_hat^2 * (X^T X)^(-1)
        SE(beta_hat_j) = sqrt(Var(beta_hat_j))
        """
        self._check_is_fitted()
        sigma2_hat = self.MSE()                                # residual variance
        XtX_inv = np.linalg.inv(self._X.T @ self._X)           # (p, p)
        var_beta = sigma2_hat * np.diag(XtX_inv)               # (p,)
        return np.sqrt(var_beta)

    def t_stats(self):
        """
        Return t-statistics and two-sided p-values for each coefficient.
        """
        self._check_is_fitted()
        se = self.coef_se()
        t_vals = self._beta / se
        df = self._n - self._p
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=df))
        return t_vals, p_vals

    def F_stat(self):
        """
        Overall F-test for the model vs. an intercept-only model.
        """
        self._check_is_fitted()
        R2 = self.R2()
        p = self._p
        n = self._n

        # F = (R2 / (p - 1)) / ((1 - R2) / (n - p))
        num = R2 / (p - 1)
        den = (1.0 - R2) / (n - p)
        F = num / den
        p_val = 1 - stats.f.cdf(F, p - 1, n - p)
        return F, p_val

    # ---------- summary printout ----------

    def summary(self, feature_names=None) -> None:
        """
        Print a small summary similar to statsmodels:

            coef, std err, t, p>|t|, R^2, adj R^2, F-statistic

        Parameters
        ----------
        feature_names : list of str or None
            Names of the features (excluding intercept). If None, uses x0, x1, ...
        """
        self._check_is_fitted()

        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(self.coef_))]

        if self.fit_intercept:
            names = ["Intercept"] + list(feature_names)
        else:
            names = list(feature_names)

        se = self.coef_se()
        t_vals, p_vals = self.t_stats()

        print("Coefficients:")
        for name, beta, s, t, p in zip(names, self._beta, se, t_vals, p_vals):
            print(
                f"{name:>10s}  coef={beta: .4f}  std err={s: .4f}  "
                f"t={t: .3f}  p={p: .3f}"
            )

        print()
        print(
            f"Residual standard error: {np.sqrt(self.MSE()):.4f} "
            f"on {self._n - self._p} degrees of freedom"
        )
        print(
            f"R-squared: {self.R2():.4f}, "
            f"Adjusted R-squared: {self.adj_R2():.4f}"
        )
        F, pF = self.F_stat()
        print(
            f"F-statistic: {F:.3f} on {self._p - 1} and {self._n - self._p} DF, "
            f"p-value: {pF:.3g}"
        )