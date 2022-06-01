using System.Linq;
using AutoDiff;
using Numpy;

namespace ML.Core.Losses
{
    public class LogCosh : Loss
    {
        /// <summary>
        ///     最小绝对值损失
        ///     Computes the logarithm of the hyperbolic cosine of the prediction error.
        ///     J(la) =  log((exp(x) + exp(-x))/2), where x is the error y_pred - y_true.
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public LogCosh(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var x = y_pred - y_true;
            var allAbdDetta = np.log((np.exp(x) + np.exp(-x)) / 2);
            return np.average(allAbdDetta);
        }

        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var logcosh = y_pred
                .Zip(y_true, (y1, y2) => y1 - y2)
                .Select(x => TermBuilder.Log((TermBuilder.Exp(x) + TermBuilder.Exp(-x)) / 2))
                .ToArray();
            var finalLoss = TermBuilder.Sum(logcosh) / logcosh.Length;
            return finalLoss;
        }
    }
}