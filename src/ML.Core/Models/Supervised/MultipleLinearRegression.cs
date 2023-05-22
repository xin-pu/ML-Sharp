using ML.Core.Transform;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public class MultipleLinearRegression : ModelGD
    {
        /// <summary>
        ///     多元线性回归模型
        ///     y=α + β1*x1 + β2*x2 + ... + βn*xn"
        /// </summary>
        public MultipleLinearRegression()
        {
            Transformer = new LinearFirstorder();
        }


        public override string Description => " 多元线性回归模型\r\ny=α + β1*x1 + β2*x2 + ... + βn*xn";


        public override TermMatrix CallGraph(NDarray features)
        {
            var feature = Transformer.Call(features);
            return TermOp.Multiply(feature, Variables);
        }

        public override NDarray Call(NDarray features)
        {
            var feature = Transformer.Call(features);
            var y_pred = feature.matmul(Weights.T);
            return y_pred;
        }
    }
}