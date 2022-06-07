using Numpy;

namespace ML.Core.Metrics.Categorical
{
    public class CategoricalCrossentropy : Metric
    {
        public override string Describe =>
            "Computes the crossentropy metric between the labels and predictions.";

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var y_pred_ = np.clip(y_pred, np.array(1E-7), np.array(1));
            var xent = -np.sum(y_true * np.log(y_pred_), -1);
            var reducedXent = np.average(xent);
            return reducedXent;
        }
    }
}