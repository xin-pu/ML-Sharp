using System.Linq;
using AutoDiff;

namespace ML.Core.Losses
{
    public class LeastAbsolute : Loss
    {
        /// <summary>
        ///     最小绝对值损失
        ///     J(la)= sigma(|y-yp|)
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public LeastAbsolute(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }


        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var allAbs = y_pred
                .Zip(y_true, (y1, y2) => y1 - y2)
                .Select(d => TermBuilder.Power(TermBuilder.Power(d, 2), 0.5))
                .ToArray();
            var finalLoss = TermBuilder.Sum(allAbs) / allAbs.Length;
            return finalLoss;
        }
    }
}