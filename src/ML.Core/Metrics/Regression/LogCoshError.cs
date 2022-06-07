using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class LogCoshError : Metric
    {
        /// <summary>
        ///     Log Cosh Error
        /// </summary>
        public LogCoshError()
        {
        }

        public override string Describe =>
            "Computes the logarithm of the hyperbolic cosine of the prediction error.";

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var error = y_pred - y_true;
            var logcosh = np.log((np.exp(error) + np.exp(-error)) / 2);
            return np.average(logcosh);
        }
    }
}