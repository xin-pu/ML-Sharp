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

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var error = y_pred - y_true;
            var logcosh = ((error.exp() + (-error).exp()) / 2).log();
            return logcosh.average();
        }
    }
}