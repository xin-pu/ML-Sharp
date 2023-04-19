using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class MeanSquaredError : Metric
    {
        /// <summary>
        ///     Mean Square Error
        ///     均方误差
        /// </summary>
        public MeanSquaredError()
        {
        }

        public override string Describe =>
            "Computes the mean squared error between y_true and y_pred.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var error = (y_true - y_pred).power(np.array(2));
            return error.average();
        }
    }
}