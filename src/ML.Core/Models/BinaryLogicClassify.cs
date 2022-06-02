using System.Linq;
using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using ML.Utilty;
using Numpy;

namespace ML.Core.Models
{
    public class BinaryLogicClassify<T> : Model<T>
        where T : DataView
    {
        public override Transformer Transformer { get; set; } = new LinearFirstorder();

        public override Term[] CallGraph(NDarray x)
        {
            var feature = Transformer.Call(x);
            var terms = term.matmul(Variables, feature);
            return terms.Select(a => term.sigmoid(a)).ToArray();
        }

        public override NDarray Call(NDarray x)
        {
            var feature = Transformer.Call(x);
            var y_pred = nn.sigmoid(np.matmul(feature, Weights));
            return y_pred;
        }
    }
}