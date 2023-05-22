using FluentAssertions;
using Numpy;

namespace ML.Core.Transform
{
    public class Polynomial : Transformer
    {
        /// <summary>
        ///     多项式逼近器
        ///     input:[x]
        ///     output: [1,x,x^2,x^3,...,x^N]
        /// </summary>
        /// <param name="degree">阶数</param>
        public Polynomial(int degree)
        {
            degree.Should().BePositive("Please give Degree > 0 ");
            Degree = degree;
        }

        public int Degree { protected set; get; }

        public override bool IsKernel => false;

        /// <summary>
        /// </summary>
        /// <param name="input">input shape should be [batch size,1] contains 1 feature</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public override NDarray Call(NDarray input)
        {
            input.ndim.Should().Be(2, "input dims shoulbe be 2");
            input.shape[1].Should().Be(1, "input should contain only 1 feature");

            var all = Enumerable.Range(0, Degree + 1)
                .Select(d => input.power(np.array(1.0 * d)))
                .ToArray();
            return np.hstack(all);
        }
    }
}