using System.Linq;
using AutoDiff;
using Numpy;

namespace ML.Core.Losses
{
    public class LeastSquares : Loss
    {
        /// <summary>
        ///     最小二乘损失
        ///     J(la)= sigma{(y-yp)^2}
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public LeastSquares(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }


        internal override double CalculateLLoss(NDarray y_pred, NDarray y_true)
        {
            var allAbdDetta = np.square(y_pred - y_true);
            return 0.5 * np.average(allAbdDetta);
        }

        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var squares = y_pred
                .Zip(y_true, (y1, y2) => y1 - y2)
                .Select(d => TermBuilder.Power(d, 2))
                .ToArray();
            var finalLoss = 0.5 * TermBuilder.Sum(squares) / squares.Length;
            return finalLoss;
        }
    }
}