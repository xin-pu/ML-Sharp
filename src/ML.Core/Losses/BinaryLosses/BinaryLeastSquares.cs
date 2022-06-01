using System.Linq;
using AutoDiff;
using ML.Utilty;
using Numpy;

namespace ML.Core.Losses
{
    public class BinaryLeastSquares : Loss
    {
        /// <summary>
        ///     二分类最小二乘损失
        ///     J(la)= (sigmoid(y)-yp)^2
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinaryLeastSquares(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        internal override double CalculateLLoss(NDarray y_pred, NDarray y_true)
        {
            var sigmoid = nn.sigmoid(y_pred);
            var allDelta = np.square(sigmoid - y_true);
            return 0.5 * np.average(allDelta);
        }

        /// <summary>
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