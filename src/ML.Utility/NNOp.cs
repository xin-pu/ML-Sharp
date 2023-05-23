using Numpy;
using Numpy.Models;

namespace ML.Utility
{
    public class NNOp
    {
        /// <summary>
        ///     Sigmoid 函数 Logistic 函数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static NDarray Sigmoid(NDarray input)
        {
            return 1.0 / (1 + (-input).exp());
        }

        /// <summary>
        ///     Sigmoid 函数 Tanh 函数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static NDarray Tanh(NDarray input)
        {
            return 2.0 / (1 + (-input).exp()) - 1;
        }

        public static NDarray Initial(Shape shape, InitialMode initialMode)
        {
            switch (initialMode)
            {
                case InitialMode.Ones:
                    return np.ones(shape);
                case InitialMode.Zeros:
                    return np.zeros(shape);
                case InitialMode.Uniform:
                    return np.random.uniform(new NDarray<float>(new[] {0}), new NDarray<float>(new[] {1}),
                        shape.Dimensions);
                default:
                    throw new ArgumentOutOfRangeException(nameof(initialMode), initialMode, null);
            }
        }
    }

    public enum InitialMode
    {
        Ones,
        Zeros,
        Uniform
    }
}