using ML.Core.Transform;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public class Perceptron : ModelGD
    {
        private int classes;

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
            Classes = classes;
            Transformer = new LinearFirstorder();
        }

        public Perceptron()
        {
            Transformer = new LinearFirstorder();
        }

        public int Classes
        {
            get => classes;
            set => Set(ref classes, value);
        }

        public override string Description => "Perceptron";

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
            var y_pred = feature.matmul(Weights.T);
            return sign(y_pred);
        }

        private NDarray sign(NDarray inputDarray)
        {
            var exp = inputDarray.exp();
            var rowsum = exp.sum(-1, keepdims: true);
            return exp / rowsum;
        }
    }
}