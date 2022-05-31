using System.Linq;
using AutoDiff;

namespace ML.Core.Losses
{
    public class Softmax : Loss
    {
        public Softmax(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        /// <summary>
        ///     J(la)= (sigmoid(y)-yp)^2
        /// </summary>
        /// <param name="y_pred"></param>
        /// <param name="y_true">should be -1 or 1 </param>
        /// <returns></returns>
        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var alllogdelta = y_pred
                .Zip(y_true, (y1, y2) => TermBuilder.Log(1 + TermBuilder.Exp(-y2 * y1)))
                .ToArray();
            var crossEntropy = -TermBuilder.Sum(alllogdelta) / alllogdelta.Length;
            return crossEntropy;
        }
    }
}