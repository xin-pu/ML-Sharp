using System.Linq;
using AutoDiff;

namespace ML.Core.Losses
{
    public class SigmoidLeastSquares : Loss
    {
        public SigmoidLeastSquares(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        /// <summary>
        ///     J(la)= (sigmoid(y)-yp)^2
        /// </summary>
        /// <param name="y_pred"></param>
        /// <param name="y_true">should be 0 or 1 </param>
        /// <returns></returns>
        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var allAbs = y_pred
                .Zip(y_true, (y1, y2) => term.sigmoid(y1) - y2)
                .Select(d => TermBuilder.Power(d, 2))
                .ToArray();
            var finalLoss = 0.5 * TermBuilder.Sum(allAbs) / allAbs.Length;
            return finalLoss;
        }
    }
}