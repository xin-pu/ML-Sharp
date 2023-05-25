using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <summary>
    ///     Sigmoid 激活层
    /// </summary>
    public class Sigmoid : Layer
    {
        public override NDarray Forward(NDarray input)
        {
            var res = Output = 1 / (1 + (-input).exp());
            return res;
        }

        public override NDarray Backward(NDarray gradient, Optimizer optimizer, int epoch = 0)
        {
            var res = Output * (1 - Output) * gradient;
            return res;
        }
    }
}