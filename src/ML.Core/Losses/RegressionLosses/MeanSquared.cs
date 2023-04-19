using AutoDiff;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class MeanSquared : Loss
    {
        /// <summary>
        ///     最小二乘损失
        ///     Computes the mean of squares of errors between labels and predictions.
        ///     J(la)= square(y_true - y_pred)
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public MeanSquared()
        {
        }

        public override string Describe =>
            "最小二乘损失\r\n J(la)= square(y_true - y_pred)";

        public override void Dispose()
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var allAbdDetta = (y_pred - y_true).square();
            return 0.5 * allAbdDetta.average();
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