using Numpy;

namespace ML.Core.Metrics.Categorical
{
    public class CategoricalCrossentropy : Metric
    {
        /// <summary>
        ///     Categorical Accuracy 多分类交叉熵
        ///     Computes the cross entropy metric between the labels and predictions.
        /// </summary>
        /// </summary>
        public CategoricalCrossentropy()
        {
        }

        public override string Describe =>
            "Computes the crossentropy metric between the labels and predictions.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var y_pred_ = y_pred.clip(np.array(1E-7), np.array(1));
            var xent = -(y_true * y_pred_.log()).sum(-1);
            var reducedXent = xent.average();
            return reducedXent;
        }
    }
}