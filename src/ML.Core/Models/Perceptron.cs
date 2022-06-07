using ML.Core.Data;
using ML.Core.Transform;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public class Perceptron<T> : ModelGD<T>
        where T : DataView
    {
        /// <summary>
        ///     Y should be [Batch, One Hot]
        ///     [
        ///     [0,0,1],
        ///     [0,1,0]
        ///     ]
        /// </summary>
        /// <param name="classes"></param>
        public Perceptron(int classes)
        {
            Transformer = new LinearFirstorder();
            Classes = classes;
        }


        public int Classes { set; get; }
        public override string Description { get; }

        /// <summary>
        /// </summary>
        /// <param name="features"></param>
        /// <returns>[batch size, x*Wc]</returns>
        public override TermMatrix CallGraph(NDarray features)
        {
            var feature = Transformer.Call(features);
            return term.multiply(feature, Variables);
        }

        public override NDarray Call(NDarray features)
        {
            var feature = Transformer.Call(features);
            var y_pred = np.matmul(feature, Weights.T);
            return sign(y_pred);
        }

        private NDarray sign(NDarray inputDarray)
        {
            var exp = np.exp(inputDarray);
            var rowsum = np.sum(exp, -1, keepdims: true);
            return exp / rowsum;
        }
    }
}