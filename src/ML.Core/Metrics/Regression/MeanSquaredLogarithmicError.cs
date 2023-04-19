using System.Linq;
using FluentAssertions;
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

        public override void Dispose()
        {
        }

        internal override void precheck(NDarray y_true, NDarray y_pred)
        {
            y_true.GetData<double>().Min().Should().BeGreaterOrEqualTo(0, "Ground truth label values");
            y_pred.GetData<double>().Min().Should().BeGreaterOrEqualTo(0, "Ground truth label values");
            base.precheck(y_true, y_pred);
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var logTrue = (y_true + 1).log();
            var error = ((y_true + 1).log() - (y_pred + 1).log()).square();
            return error.average();
        }
    }
}