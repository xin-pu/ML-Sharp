using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class RSquared : Metric
    {
        /// <summary>
        ///     R-Squared
        ///     决定系数
        ///     R-Squared=SSR/SST=1-SSE/SST
        /// </summary>
        public RSquared()
        {
        }

        public override string Describe =>
            "R-Squared (R² or the coefficient of determination) is " +
            "a statistical measure in a regression model " +
            "that determines the proportion of variance in the dependent";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var sse = new MeanSquaredError().Call(y_true, y_pred);
            var sst = y_true.var();
            return 1 - sse / sst;
        }
    }
}