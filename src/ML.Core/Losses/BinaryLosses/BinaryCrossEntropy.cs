using AutoDiff;
using ML.Utility;
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
        public BinaryCrossentropy()
        {
        }

        public override string Describe => "二分类交叉熵损失\r\nJ(la) = -sigma (y_t*log(y_p)+(1-y_t)*log(1-y_p)";

        public override void Dispose()
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            //labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
            //Todo
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var alllogdelta = y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log();
            return -alllogdelta.average();
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var lossMatrix = y_pred.Log() * y_true + (y_pred.Negation() + 1).Log() * (1 - y_true);
            return -lossMatrix.Average();
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