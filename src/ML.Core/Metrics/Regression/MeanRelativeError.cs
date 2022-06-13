using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class MeanRelativeError : Metric
    {
        /// <summary>
        ///     Mean Relative Error
        ///     平均相对误差
        /// </summary>
        public MeanRelativeError()
        {
        }

        public override string Describe =>
            "Computes the mean relative error by normalizing with the given values.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = (y_pred - y_true) / y_true;
            return np.average(delta);
        }
    }
}