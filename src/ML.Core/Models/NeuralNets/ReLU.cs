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
            return np.array(0).maximum(input);
        }

        public override NDarray Backward(NDarray error)
        {
            throw new NotImplementedException();
        }
    }
}