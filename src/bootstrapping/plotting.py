"""
Plotting Module for Bootstrap Analysis

Comprehensive visualization suite for bootstrap results including
distribution plots, risk metrics visualization, and interactive reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import warnings
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots disabled.")


class BootstrapPlotter:
    """
    Comprehensive plotting suite for bootstrap analysis.
    
    Provides static and interactive visualizations for:
    - Distribution analysis
    - Risk metrics
    - Performance comparison
    - Statistical test results
    """
    
    def __init__(self, bootstrap_analyzer):
        """
        Initialize plotter with bootstrap analyzer.
        
        Args:
            bootstrap_analyzer: AdvancedBootstrapping instance
        """
        self.analyzer = bootstrap_analyzer
        self.style_config = {
            'figure_size': (12, 8),
            'dpi': 100,
            'color_palette': 'viridis',
            'alpha': 0.7
        }
        
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.style_config['figure_size']
        plt.rcParams['figure.dpi'] = self.style_config['dpi']
    
    def plot_distribution_analysis(self, results: Dict[str, Any], 
                                 metrics: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution analysis of bootstrap results.
        
        Args:
            results: Bootstrap simulation results
            metrics: Metrics to plot (if None, plots all)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        simulated_stats = results['simulated_stats']
        original_stats = results['original_stats']
        
        if not simulated_stats:
            raise ValueError("No simulation results to plot")
        
        # Convert to DataFrame
        sim_df = pd.DataFrame(simulated_stats)
        
        if metrics is None:
            metrics = ['Sharpe', 'Sortino', 'MaxDrawdown', 'CumulativeReturn']
            metrics = [m for m in metrics if m in sim_df.columns]
        
        # Create subplots
        n_metrics = len(metrics)
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric not in sim_df.columns:
                continue
                
            ax = axes[i]
            values = sim_df[metric].dropna()
            
            if len(values) == 0:
                continue
            
            # Plot histogram
            ax.hist(values, bins=50, alpha=self.style_config['alpha'], 
                   density=True, label='Simulated')
            
            # Plot original value
            if metric in original_stats and not np.isnan(original_stats[metric]):
                ax.axvline(original_stats[metric], color='red', linestyle='--',
                          linewidth=2, label='Original')
            
            # Add statistics text
            stats_text = f'Mean: {values.mean():.4f}\n'
            stats_text += f'Std: {values.std():.4f}\n'
            stats_text += f'Skew: {values.skew():.4f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
            
            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_equity_curves(self, results: Dict[str, Any], 
                          n_curves: int = 100,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot simulated equity curves.
        
        Args:
            results: Bootstrap simulation results
            n_curves: Number of curves to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'simulated_equity_curves' not in results or results['simulated_equity_curves'] is None:
            raise ValueError("No equity curves available in results")
        
        equity_curves = results['simulated_equity_curves']
        
        fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
        
        # Sample curves to plot
        n_sims = len(equity_curves.columns)
        n_plot = min(n_curves, n_sims)
        plot_cols = np.random.choice(equity_curves.columns, n_plot, replace=False)
        
        # Plot simulated curves
        for col in plot_cols:
            ax.plot(equity_curves.index, equity_curves[col], 
                   alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot percentiles
        percentiles = equity_curves.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
        
        ax.plot(percentiles.index, percentiles[0.5], color='red', linewidth=2, 
               label='Median')
        ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75],
                       alpha=0.3, color='red', label='25th-75th percentile')
        ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95],
                       alpha=0.2, color='red', label='5th-95th percentile')
        
        # Plot original if available
        if hasattr(self.analyzer, 'benchmark_series') and self.analyzer.benchmark_series is not None:
            benchmark_equity = (1 + self.analyzer.benchmark_series).cumprod() * self.analyzer.init_cash
            ax.plot(benchmark_equity.index, benchmark_equity.values, 
                   color='green', linewidth=2, label='Benchmark')
        
        ax.set_title('Simulated Equity Curves')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown_analysis(self, results: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown analysis.
        
        Args:
            results: Bootstrap simulation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'simulated_equity_curves' not in results or results['simulated_equity_curves'] is None:
            raise ValueError("No equity curves available for drawdown analysis")
        
        equity_curves = results['simulated_equity_curves']
        
        # Calculate drawdowns
        running_max = equity_curves.expanding().max()
        drawdowns = (equity_curves - running_max) / running_max
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot drawdown time series
        n_plot = min(50, len(drawdowns.columns))
        plot_cols = np.random.choice(drawdowns.columns, n_plot, replace=False)
        
        for col in plot_cols:
            ax1.plot(drawdowns.index, drawdowns[col] * 100, 
                    alpha=0.2, color='red', linewidth=0.5)
        
        # Plot median drawdown
        median_dd = drawdowns.median(axis=1)
        ax1.plot(drawdowns.index, median_dd * 100, color='darkred', 
                linewidth=2, label='Median Drawdown')
        
        ax1.fill_between(drawdowns.index, 0, median_dd * 100, 
                        alpha=0.3, color='red')
        ax1.set_title('Drawdown Time Series')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot maximum drawdown distribution
        max_drawdowns = drawdowns.min() * 100
        ax2.hist(max_drawdowns, bins=50, alpha=self.style_config['alpha'], 
                density=True, color='red')
        
        # Add statistics
        stats_text = f'Mean Max DD: {max_drawdowns.mean():.2f}%\n'
        stats_text += f'Std Max DD: {max_drawdowns.std():.2f}%\n'
        stats_text += f'5th percentile: {max_drawdowns.quantile(0.05):.2f}%\n'
        stats_text += f'95th percentile: {max_drawdowns.quantile(0.95):.2f}%'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        ax2.set_title('Maximum Drawdown Distribution')
        ax2.set_xlabel('Maximum Drawdown (%)')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_risk_return_scatter(self, results: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot risk-return scatter with efficient frontier.
        
        Args:
            results: Bootstrap simulation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        simulated_stats = results['simulated_stats']
        original_stats = results['original_stats']
        
        sim_df = pd.DataFrame(simulated_stats)
        
        if 'CumulativeReturn' not in sim_df.columns or 'Volatility' not in sim_df.columns:
            raise ValueError("Required metrics not available for risk-return plot")
        
        fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
        
        # Scatter plot of simulations
        scatter = ax.scatter(sim_df['Volatility'] * 100, sim_df['CumulativeReturn'] * 100,
                           alpha=0.6, c=sim_df.get('Sharpe', 0), cmap='viridis')
        
        # Plot original strategy
        if 'CumulativeReturn' in original_stats and 'Volatility' in original_stats:
            ax.scatter(original_stats['Volatility'] * 100, original_stats['CumulativeReturn'] * 100,
                      color='red', s=100, marker='*', label='Original Strategy', zorder=5)
        
        # Plot benchmark if available
        if hasattr(self.analyzer, 'benchmark_series') and self.analyzer.benchmark_series is not None:
            bench_ret = self.analyzer.benchmark_series.sum()
            bench_vol = self.analyzer.benchmark_series.std() * np.sqrt(252)
            ax.scatter(bench_vol * 100, bench_ret * 100,
                      color='green', s=100, marker='s', label='Benchmark', zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk-Return Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_statistical_tests(self, test_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot statistical test results.
        
        Args:
            test_results: Statistical test results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'empirical_p_values' not in test_results:
            raise ValueError("No empirical p-values available")
        
        emp_p_vals = test_results['empirical_p_values']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # P-value bar chart
        metrics = list(emp_p_vals.keys())
        p_values = [emp_p_vals[m]['p_two_sided'] for m in metrics]
        
        bars = ax1.bar(range(len(metrics)), p_values, alpha=self.style_config['alpha'])
        ax1.axhline(y=0.05, color='red', linestyle='--', label='5% significance')
        ax1.axhline(y=0.01, color='orange', linestyle='--', label='1% significance')
        
        # Color bars based on significance
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.01:
                bar.set_color('darkgreen')
            elif p_val < 0.05:
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightcoral')
        
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.set_ylabel('P-value')
        ax1.set_title('Empirical P-values vs Benchmark')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Effect size plot
        effect_sizes = [emp_p_vals[m]['effect_size'] for m in metrics]
        
        bars2 = ax2.bar(range(len(metrics)), effect_sizes, alpha=self.style_config['alpha'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Color bars based on effect direction
        for bar, effect in zip(bars2, effect_sizes):
            if effect > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.set_ylabel('Effect Size (Cohen\'s d)')
        ax2.set_title('Effect Sizes vs Benchmark')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comprehensive_analysis(self, results: Dict[str, Any],
                                  save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive analysis plots.
        
        Args:
            results: Complete analysis results
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of generated figures
        """
        figures = {}
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Distribution analysis
        try:
            save_path = f"{save_dir}/distributions.png" if save_dir else None
            figures['distributions'] = self.plot_distribution_analysis(results, save_path=save_path)
        except Exception as e:
            warnings.warn(f"Could not generate distribution plot: {e}")
        
        # Equity curves
        try:
            save_path = f"{save_dir}/equity_curves.png" if save_dir else None
            figures['equity_curves'] = self.plot_equity_curves(results, save_path=save_path)
        except Exception as e:
            warnings.warn(f"Could not generate equity curves plot: {e}")
        
        # Drawdown analysis
        try:
            save_path = f"{save_dir}/drawdown_analysis.png" if save_dir else None
            figures['drawdown'] = self.plot_drawdown_analysis(results, save_path=save_path)
        except Exception as e:
            warnings.warn(f"Could not generate drawdown plot: {e}")
        
        # Risk-return scatter
        try:
            save_path = f"{save_dir}/risk_return.png" if save_dir else None
            figures['risk_return'] = self.plot_risk_return_scatter(results, save_path=save_path)
        except Exception as e:
            warnings.warn(f"Could not generate risk-return plot: {e}")
        
        # Statistical tests
        if 'statistical_tests' in results:
            try:
                save_path = f"{save_dir}/statistical_tests.png" if save_dir else None
                figures['statistical_tests'] = self.plot_statistical_tests(
                    results['statistical_tests'], save_path=save_path)
            except Exception as e:
                warnings.warn(f"Could not generate statistical tests plot: {e}")
        
        return figures
    
    def generate_html_report(self, results: Dict[str, Any], 
                           output_dir: str = 'results/reports/') -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            results: Complete analysis results
            output_dir: Output directory
            
        Returns:
            Path to generated report
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plot_dir = Path(output_dir) / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        figures = self.plot_comprehensive_analysis(results, save_dir=str(plot_dir))
        
        # Generate HTML report
        from jinja2 import Template
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bootstrap Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 40px; }
                .section { margin-bottom: 30px; }
                .metric-table { border-collapse: collapse; width: 100%; }
                .metric-table th, .metric-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                .metric-table th { background-color: #f2f2f2; }
                .plot { text-align: center; margin: 20px 0; }
                .summary { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Strategy Bootstrap Analysis Report</h1>
                <p>Generated on: {{ timestamp }}</p>
                <p>Method: {{ method }}</p>
                <p>Simulations: {{ n_sims }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary">
                    <p><strong>Strategy Performance:</strong> {{ summary_text }}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Key Metrics Comparison</h2>
                <table class="metric-table">
                    <tr><th>Metric</th><th>Original</th><th>Simulated Mean</th><th>Simulated Std</th></tr>
                    {% for metric, values in comparison_table.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.4f"|format(values.original) }}</td>
                        <td>{{ "%.4f"|format(values.sim_mean) }}</td>
                        <td>{{ "%.4f"|format(values.sim_std) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            {% if 'distributions' in figures %}
            <div class="section">
                <h2>Distribution Analysis</h2>
                <div class="plot">
                    <img src="plots/distributions.png" alt="Distribution Analysis" style="max-width: 100%;">
                </div>
            </div>
            {% endif %}
            
            {% if 'equity_curves' in figures %}
            <div class="section">
                <h2>Simulated Equity Curves</h2>
                <div class="plot">
                    <img src="plots/equity_curves.png" alt="Equity Curves" style="max-width: 100%;">
                </div>
            </div>
            {% endif %}
            
            {% if 'drawdown' in figures %}
            <div class="section">
                <h2>Drawdown Analysis</h2>
                <div class="plot">
                    <img src="plots/drawdown_analysis.png" alt="Drawdown Analysis" style="max-width: 100%;">
                </div>
            </div>
            {% endif %}
            
            {% if 'risk_return' in figures %}
            <div class="section">
                <h2>Risk-Return Analysis</h2>
                <div class="plot">
                    <img src="plots/risk_return.png" alt="Risk-Return Scatter" style="max-width: 100%;">
                </div>
            </div>
            {% endif %}
            
            {% if statistical_tests %}
            <div class="section">
                <h2>Statistical Significance Tests</h2>
                {% if 'statistical_tests' in figures %}
                <div class="plot">
                    <img src="plots/statistical_tests.png" alt="Statistical Tests" style="max-width: 100%;">
                </div>
                {% endif %}
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        
        # Prepare data for template
        simulated_stats = results['simulated_stats']
        original_stats = results['original_stats']
        
        sim_df = pd.DataFrame(simulated_stats)
        comparison_table = {}
        
        for metric in ['Sharpe', 'Sortino', 'MaxDrawdown', 'CumulativeReturn']:
            if metric in original_stats and metric in sim_df.columns:
                comparison_table[metric] = {
                    'original': original_stats[metric],
                    'sim_mean': sim_df[metric].mean(),
                    'sim_std': sim_df[metric].std()
                }
        
        # Generate summary text
        sharpe_original = original_stats.get('Sharpe', np.nan)
        sharpe_sim_mean = sim_df['Sharpe'].mean() if 'Sharpe' in sim_df.columns else np.nan
        
        if not np.isnan(sharpe_original) and not np.isnan(sharpe_sim_mean):
            if sharpe_original > sharpe_sim_mean:
                summary_text = f"Strategy shows above-average performance with Sharpe ratio of {sharpe_original:.3f} vs simulated mean of {sharpe_sim_mean:.3f}."
            else:
                summary_text = f"Strategy shows below-average performance with Sharpe ratio of {sharpe_original:.3f} vs simulated mean of {sharpe_sim_mean:.3f}."
        else:
            summary_text = "Performance analysis completed. See detailed metrics below."
        
        html_content = template.render(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            method=results.get('method', 'Unknown'),
            n_sims=len(simulated_stats),
            summary_text=summary_text,
            comparison_table=comparison_table,
            figures=figures,
            statistical_tests='statistical_tests' in results
        )
        
        # Write HTML file
        report_path = Path(output_dir) / 'bootstrap_analysis_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def create_interactive_dashboard(self, results: Dict[str, Any]) -> str:
        """
        Create interactive Plotly dashboard (if available).
        
        Args:
            results: Analysis results
            
        Returns:
            HTML file path or error message
        """
        if not PLOTLY_AVAILABLE:
            return "Plotly not available for interactive plots"
        
        # This would implement a comprehensive Plotly dashboard
        # For now, return a placeholder
        return "Interactive dashboard feature coming soon"
