using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <summary>
    ///     ReLU 没有权重，反向传播时
    /// </summary>
    public class Sigmoid : Layer
    {
        public override NDarray Forward(NDarray input)
        {
            return 1 / (1 + (-input).exp());
        }

        public override NDarray Backward(NDarray error)
        {
            throw new NotImplementedException();
        }
    }
}