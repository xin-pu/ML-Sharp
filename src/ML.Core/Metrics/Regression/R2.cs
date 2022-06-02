using System.Linq;
using Numpy;

namespace ML.Core.Metrics.Regression
{
    /// <summary>
    ///     R-Squared
    ///     决定系数
    /// </summary>
    public class R2 : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = y_true - y_pred;
            var mse = np.power(delta, np.array(2))
                .GetData<double>()
                .Average();
            return 1 - mse / np.var(y_true);
        }
    }
}