using Numpy;

namespace ML.Core.Transform
{
    public class LinearFirstorder : Transformer
    {
        /// <summary>
        ///     一阶逼近器
        ///     input: [x1,x2,x3,...,xN]
        ///     output: [1,x1,x2,x3,...,xN]
        /// </summary>
        public LinearFirstorder()
        {
        }

        public override bool IsKernel => false;

        public override NDarray Call(NDarray input)
        {
            var b = np.ones(input.shape[0]);
            var res = np.insert(input, 0, b, 1);
            return res;
        }
    }
}