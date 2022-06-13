using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class MeanAbsolutePercentageError : Metric
    {
        /// <summary>
        ///     Mean Absolute Percentage Error
        ///     平均绝对误差百分比
        /// </summary>
        public MeanAbsolutePercentageError()
        {
        }

        public override string Describe =>
            "Computes the mean absolute percentage error between y_true and y_pred.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = np.abs(y_pred - y_true) / y_true;
            return np.average(delta);
        }
    }
}