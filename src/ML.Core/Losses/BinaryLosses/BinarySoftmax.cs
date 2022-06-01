using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utilty;
using Numpy;

namespace ML.Core.Losses
{
    public class BinarySoftmax : Loss
    {
        /// <summary>
        ///     二分类Softmax损失（Label∈(-1，1))
        ///     J(la)= Todo
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinarySoftmax(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        internal override void CheckLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {-1, 1}, "Labels should be -1 or 1");
        }

        internal override double CalculateLoss(NDarray y_pred, NDarray y_true)
        {
            var tah = nn.tanh(y_pred);
            var alllogdelta = np.log(1 + np.exp(-tah * y_true));
            return np.average(alllogdelta);
        }

        /// <summary>
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