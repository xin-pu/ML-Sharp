using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <summary>
    ///     ReLU 没有权重，反向传播时
    /// </summary>
    public class ReLU : Layer
    {
        public override NDarray Forward(NDarray input)
        {
            var res = Output = np.array(0).maximum(input);
            return res;
        }

        public override NDarray Backward(NDarray gradient, Optimizer optimizer, int epoch = 0)
        {
            throw new NotImplementedException();
        }
    }
}