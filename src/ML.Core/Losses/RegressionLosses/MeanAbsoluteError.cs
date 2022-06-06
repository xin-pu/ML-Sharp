using AutoDiff;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class MeanAbsoluteError : Loss
    {
        /// <summary>
        ///     最小绝对值损失
        ///     Computes the mean of absolute difference between labels and predictions.
        ///     J(la) = abs(y_true - y_pred)
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public MeanAbsoluteError(
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
            var allAbdDetta = np.abs(y_pred - y_true);
            return np.average(allAbdDetta);
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var lossTerm = y_pred.Subtract(y_true)
                .Power(2)
                .Power(0.5)
                .Average();
            return lossTerm;
        }
    }
}