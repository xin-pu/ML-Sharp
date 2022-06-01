using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utilty;
using Numpy;

namespace ML.Core.Losses
{
    public class BinaryCrossentropy : CategoricalLoss
    {
        /// <summary>
        ///     二分类交叉熵损失
        ///     J(la)= - sigma (y_t*log(y_p)+(1-y_t)*log(1-y_p)
        ///     y_t:y true
        ///     y_p:y pred
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinaryCrossentropy(
            LabelType labelType = LabelType.Probability,
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(labelType, lamdba, regularization)
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var alllogdelta = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred);
            return -np.average(alllogdelta);
        }

        internal override Term getModelLoss(Term[] y_pred, NDarray y_true)
        {
            var array = y_true.GetData<double>();
            var alllogdelta = y_pred
                .Zip(array, (y1, y2) => y2 * TermBuilder.Log(y1) + (1 - y2) * TermBuilder.Log(1 - y1))
                .ToArray();
            var crossEntropy = -TermBuilder.Sum(alllogdelta) / alllogdelta.Length;
            return crossEntropy;
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