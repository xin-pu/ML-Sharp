using System.Linq;
using Numpy;

namespace ML.Core.Metrics.Categorical
{
    /// <summary>
    ///     ErrorRate
    ///     错误率
    /// </summary>
    public class ErrorRate : Metric
    {
        public override string Describe { get; }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var res = np.abs(y_true - y_pred);
            var tptn = res.GetData<double>().Count(a => a < 5E-1);
            return 1 - 1.0 * tptn / y_true.len;
        }

        public override string ToString()
        {
            return $"{GetType().Name}:\t{ValueError:P2}";
        }
    }
}