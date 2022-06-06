using AutoDiff;
using ML.Utility;
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

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var per = (y_pred - y_true) / y_true;
            var lossTerm = per.Power(2).Power(0.5).Average();
            return lossTerm;
        }
    }
}