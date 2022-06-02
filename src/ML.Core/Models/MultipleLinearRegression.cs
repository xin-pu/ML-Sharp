using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using ML.Utilty;
using Numpy;

namespace ML.Core.Models
{
    public class MultipleLinearRegression<T> : Model<T>
        where T : DataView
    {
        public override Transformer Transformer { get; set; } = new LinearFirstorder();

        public override Term[] CallGraph(NDarray x)
        {
            var feature = Transformer.Call(x);
            return term.matmul(Variables, feature);
        }

        public override NDarray Call(NDarray x)
        {
            var feature = Transformer.Call(x);
            var y_pred = np.matmul(feature, WeightNDarray);
            return y_pred;
        }
    }
}