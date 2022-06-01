using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utilty;
using Numpy;

namespace ML.Core.Losses
{
    public class BinaryCrossentropy : Loss
    {
        /// <summary>
        ///     二分类交叉熵损失（Label∈(0，1))
        ///     J(la)= Todo
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinaryCrossentropy(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }

        internal override void CheckLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
        }

        internal override double CalculateLoss(NDarray y_pred, NDarray y_true)
        {
            var sigmoid = nn.sigmoid(y_pred);
            var alllogdelta = y_true * np.log(sigmoid) + (1 - y_true) * np.log(1 - sigmoid);
            return -np.average(alllogdelta);
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