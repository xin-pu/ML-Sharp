using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class BinarySoftmax : CategoricalLoss
    {
        /// <summary>
        ///     二分类Softmax损失（Label∈(-1，1))
        ///     J(la)= sigma ( log(1+e^(-y_t*y_p) )
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public BinarySoftmax(
            LabelType labelType = LabelType.Probability,
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(labelType, lamdba, regularization)
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {-1, 1}, "Labels should be -1 or 1");
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var alllogdelta = np.log(1 + np.exp(-y_pred * y_true));
            return np.average(alllogdelta);
        }

        /// <summary>
        /// </summary>
        /// <param name="y_pred"></param>
        /// <param name="y_true">should be -1 or 1 </param>
        /// <returns></returns>
        internal override Term getModelLoss(Term[] y_pred, NDarray y_true)
        {
            var array = y_true.GetData<double>();
            var alllogdelta = y_pred
                .Zip(array, (y1, y2) => TermBuilder.Log(1 + TermBuilder.Exp(-y2 * y1)))
                .ToArray();
            var crossEntropy = -TermBuilder.Sum(alllogdelta) / alllogdelta.Length;
            return crossEntropy;
        }


        internal override Term convertProbabilityTerm(Term labels_logits)
        {
            return term.tanh(labels_logits);
        }

        internal override NDarray convertProbabilityNDarray(NDarray labels_logits)
        {
            return nn.tanh(labels_logits);
        }
    }
}