using Numpy;

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
    }
}