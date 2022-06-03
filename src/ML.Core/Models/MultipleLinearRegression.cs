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
        /// <summary>
        ///     多元线性回归模型
        ///     y=α + β1*x1 + β2*x2 + ... + βn*xn"
        /// </summary>
        public MultipleLinearRegression()
        {
            Transformer = new LinearFirstorder();
        }

        public override Transformer Transformer { get; set; }

        public override string Description => " 多元线性回归模型\r\ny=α + β1*x1 + β2*x2 + ... + βn*xn";

        public override Term[] CallGraph(NDarray x)
        {
            var feature = Transformer.Call(x);
            return term.matmul(Variables, feature);
        }


        public override NDarray Call(NDarray x)
        {
            var feature = Transformer.Call(x);
            var y_pred = np.matmul(feature, Weights);
            return y_pred;
        }
    }
}