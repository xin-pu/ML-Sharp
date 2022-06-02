using System.Linq;
using Numpy;

namespace ML.Core.Metrics.Regression
{
    /// <summary>
    ///     Mean Square Error
    ///     均方误差
    /// </summary>
    public class MSE : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = y_true - y_pred;
            var mse = np.power(delta, np.array(2))
                .GetData<double>()
                .Average();
            return mse;
        }
    }
}