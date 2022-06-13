using System.Linq;
using Numpy;

namespace ML.Core.Metrics.Categorical
{
    public class CategoricalAccuracy : Metric
    {
        /// <summary>
        ///     Categorical Accuracy 多分类准确率
        ///     Calculates how often predictions match one-hot labels.
        /// </summary>
        public CategoricalAccuracy()
        {
        }

        public override string Describe =>
            "Computes the crossentropy metric between the labels and predictions.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var res = np.equal(np.argmax(y_true, -1), np.argmax(y_pred, -1));
            var resArray = res.GetData<bool>();
            return 1.0 * resArray.Count(a => a) / resArray.Length;
        }

        public override string ToString()
        {
            return $"[{Logogram}]:{ValueError:P2}";
        }
    }
}