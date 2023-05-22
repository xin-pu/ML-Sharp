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
        public BinarySoftmax()
        {
        }

        public override string Describe => "二分类Softmax损失\r\nJ(la)= sigma ( log(1+e^(-y_t*y_p) )";

        internal override void checkLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {-1, 1}, "Labels should be -1 or 1");
        }

        public override void Dispose()
        {
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var alllogdelta = (1 + (-y_pred * y_true).exp()).log();
            return alllogdelta.average();
        }

        /// <summary>
        /// </summary>
        /// <param name="y_pred"></param>
        /// <param name="y_true">should be -1 or 1 </param>
        /// <returns></returns>
        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            return ((y_pred * y_true).Exp() + 1).Log().Average();
        }


        internal override Term convertProbabilityTerm(Term labels_logits)
        {
            return TermOp.Tanh(labels_logits);
        }

        internal override NDarray convertProbabilityNDarray(NDarray labels_logits)
        {
            return NNOp.Tanh(labels_logits);
        }
    }
}