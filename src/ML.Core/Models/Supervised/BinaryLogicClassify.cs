using ML.Core.Transform;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public class BinaryLogicClassify : ModelGD
    {
        /// <summary>
        ///     多元线性逻辑回归模型
        ///     y=α + β1*x1 + β2*x2 + ... + βn*xn"
        /// </summary>
        public BinaryLogicClassify()
        {
            Transformer = new LinearFirstorder();
        }


        public override string Description => " 多元线性逻辑回归模型\r\n y=α + β1*x1 + β2*x2 + ... + βn*xn";


        public override TermMatrix CallGraph(NDarray features)
        {
            var feature = Transformer.Call(features);
            return term.multiply(feature, Variables).Sigmoid();
        }

        public override NDarray Call(NDarray features)
        {
            var feature = Transformer.Call(features);
            var y_pred = nn.sigmoid(feature.matmul(Weights.T));
            return sign(y_pred);
        }

        private NDarray sign(NDarray input)
        {
            return np.select(
                new[] {input >= 0.5, input < 0.5},
                new NDarray[] {np.array(1), np.array(0)});
        }
    }
}