using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utilty;
using Numpy;

namespace ML.Core.Losses
{
    public class BinaryLeastSquares : CategoricalLoss
    {
        /// <summary>
        ///     二分类最小二乘损失
        ///     J(la)= (sigmoid(y)-yp)^2
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinaryLeastSquares(
            LabelType labelType = LabelType.Probability,
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(labelType, lamdba, regularization)
        {
        }

        internal override void CheckLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
        }

        internal override double CalculateLoss(NDarray y_pred, NDarray y_true)
        {
            var allDelta = np.square(y_pred - y_true);
            return 0.5 * np.average(allDelta);
        }

        /// <summary>
        /// </summary>
        /// <param name="y_pred">should be 0 or 1</param>
        /// <param name="y_true">should be 0 or 1 </param>
        /// <returns></returns>
        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            var allAbs = y_pred
                .Zip(y_true, (y1, y2) => y1 - y2)
                .Select(d => TermBuilder.Power(d, 2))
                .ToArray();
            var finalLoss = 0.5 * TermBuilder.Sum(allAbs) / allAbs.Length;
            return finalLoss;
        }

        public override Term convertProbabilityTerm(Term labels_logits)
        {
            return term.sigmoid(labels_logits);
        }

        public override NDarray convertProbabilityNDarray(NDarray labels_logits)
        {
            return nn.sigmoid(labels_logits);
        }
    }
}