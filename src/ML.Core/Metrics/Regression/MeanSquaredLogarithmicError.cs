using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class MeanSquaredLogarithmicError : Metric
    {
        /// <summary>
        ///     Mean Squared Logarithmic Error
        ///     均方对数误差
        /// </summary>
        public MeanSquaredLogarithmicError()
        {
        }

        public override string Describe =>
            "Computes the mean squared logarithmic error between y_true and y_pred.";

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var error = np.power(np.log(y_true + 1) - np.log(y_pred + 1), np.array(2));
            return np.average(error);
        }
    }
}