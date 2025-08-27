import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statsmodels.tsa.api import VAR


class LagSelectionVisualizer:
    def __init__(self, data, maxlags=10):
        """
        Initialize the visualizer.

        Parameters:
        - data: pandas DataFrame, multivariate time series
        - maxlags: maximum number of lags to test
        """
        self.data = data
        self.maxlags = maxlags
        self.results = None

    def compute_criteria(self):
        """Compute AIC and BIC for different lags."""
        aic_vals, bic_vals = [], []
        for lag in range(1, self.maxlags + 1):
            model = VAR(self.data).fit(lag)
            aic_vals.append(model.aic)
            bic_vals.append(model.bic)

        self.results = pd.DataFrame({
            "Lag": range(1, self.maxlags + 1),
            "AIC": aic_vals,
            "BIC": bic_vals
        })
        return self.results

    def animate(self, filename="lag_selection.gif"):
        """Generate an animated GIF showing lag selection."""
        if self.results is None:
            raise ValueError("Run compute_criteria() before animate().")

        fig, ax = plt.subplots(figsize=(6, 4))

        def update(frame):
            ax.clear()
            ax.plot(self.results["Lag"][:frame + 1], self.results["AIC"][:frame + 1], label="AIC", marker="o")
            ax.plot(self.results["Lag"][:frame + 1], self.results["BIC"][:frame + 1], label="BIC", marker="s")

            # Highlight best points
            best_aic_idx = self.results["AIC"][:frame + 1].idxmin()
            best_bic_idx = self.results["BIC"][:frame + 1].idxmin()
            ax.scatter(self.results["Lag"][best_aic_idx], self.results["AIC"][best_aic_idx],
                       color="red", s=80, zorder=5, label="Best AIC" if frame == self.maxlags - 1 else "")
            ax.scatter(self.results["Lag"][best_bic_idx], self.results["BIC"][best_bic_idx],
                       color="green", s=80, zorder=5, label="Best BIC" if frame == self.maxlags - 1 else "")

            ax.set_title(f"Lag Order Selection (up to {frame + 1})")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Criterion Value")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

        ani = animation.FuncAnimation(fig, update, frames=self.maxlags, interval=700, repeat=False)
        ani.save(filename, writer="pillow")
        plt.close(fig)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n_obs = 200
    data = pd.DataFrame({
        "y1": np.random.randn(n_obs).cumsum(),
        "y2": np.random.randn(n_obs).cumsum()
    })

    viz = LagSelectionVisualizer(data, maxlags=10)
    results = viz.compute_criteria()
    print(results)
    viz.animate("lag_selection.gif")
