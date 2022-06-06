using AutoDiff;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class MeanSquaredError : Loss
    {
        /// <summary>
        ///     最小二乘损失
        ///     Computes the mean of squares of errors between labels and predictions.
        ///     J(la)= square(y_true - y_pred)
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public MeanSquaredError(
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
            var allAbdDetta = np.square(y_pred - y_true);
            return 0.5 * np.average(allAbdDetta);
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var lossTerm = (y_pred - y_true)
                .Power(2)
                .Average();
            return lossTerm;
        }
    }
}