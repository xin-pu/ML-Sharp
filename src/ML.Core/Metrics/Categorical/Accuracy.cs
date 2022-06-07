using System.Linq;
using Numpy;

namespace ML.Core.Metrics.Categorical
{
    /// <summary>
    ///     Accuracy
    ///     准确率
    /// </summary>
    public class Accuracy : Metric
    {
        public override string Describe { get; }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var res = np.abs(y_true - y_pred);
            var tptn = res.GetData<double>().Count(a => a < 1E-4);
            return 1.0 * tptn / y_true.len;
        }

        public override string ToString()
        {
            return $"{GetType().Name}:\t{ValueError:P2}";
        }
    }
}