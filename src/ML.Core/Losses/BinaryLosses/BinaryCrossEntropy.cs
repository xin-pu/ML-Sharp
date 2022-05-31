using System.Linq;
using AutoDiff;

namespace ML.Core.Losses
{
    public class BinaryCrossEntropy : Loss
    {
        /// <summary>
        ///     二分类交叉熵损失（Label∈(0，1))
        ///     J(la)= Todo
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinaryCrossEntropy(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var alllogdelta = y_pred
                .Zip(y_true, (y1, y2) =>
                {
                    var sigmoid = term.sigmoid(y1);
                    return y2 * TermBuilder.Log(sigmoid) + (1 - y2) * TermBuilder.Log(1 - sigmoid);
                })
                .ToArray();
            var crossEntropy = -TermBuilder.Sum(alllogdelta) / alllogdelta.Length;
            return crossEntropy;
        }
    }
}