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

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var error = np.power(y_true - y_pred, np.array(2));
            return np.average(error);
        }
    }
}