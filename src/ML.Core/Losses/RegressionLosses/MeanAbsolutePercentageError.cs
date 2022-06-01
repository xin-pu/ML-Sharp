using System.Linq;
using AutoDiff;
using Numpy;

namespace ML.Core.Losses
{
    public class MeanAbsolutePercentageError : Loss
    {
        /// <summary>
        ///     最小绝对值损失
        ///     Computes the mean absolute percentage error between y_true and y_pred.
        ///     J(la) = 100 * abs((y_true - y_pred) / y_true)
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public MeanAbsolutePercentageError(
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
            var allAbdDetta = np.abs((y_true - y_pred) / y_true);
            return np.average(allAbdDetta);
        }

        internal override Term getModelLoss(Term[] y_pred, NDarray y_true)
        {
            var array = y_true.GetData<double>();
            var allAbs = y_pred
                .Zip(array, (y1, y2) => (y2 - y1) / y2)
                .Select(d => TermBuilder.Power(TermBuilder.Power(d, 2), 0.5))
                .ToArray();
            var finalLoss = TermBuilder.Sum(allAbs) / allAbs.Length;
            return finalLoss;
        }
    }
}