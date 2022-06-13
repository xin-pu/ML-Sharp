using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class BinaryLeastSquares : CategoricalLoss
    {
        /// <summary>
        ///     二分类最小二乘损失
        ///     J(la)= sigma( (y_p-y_t)^2 )
        ///     y_t:y true
        ///     y_p:y pred
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinaryLeastSquares()
        {
        }

        public override string Describe => "二分类最小二乘损失\r\nJ(la) = sigma((y_p-y_t)^2)";

        public override void Dispose()
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var allDelta = np.square(y_pred - y_true);
            return 0.5 * np.average(allDelta);
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            return (y_pred - y_true).Power(2).Average() * 0.5;
        }


        internal override Term convertProbabilityTerm(Term labels_logits)
        {
            return term.sigmoid(labels_logits);
        }

        internal override NDarray convertProbabilityNDarray(NDarray labels_logits)
        {
            return nn.sigmoid(labels_logits);
        }
    }
}