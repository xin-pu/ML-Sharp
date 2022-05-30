using System.Linq;
using FluentAssertions;
using Numpy;

namespace ML.Core.Transform
{
    public class TrianglePoly : Transformer
    {
        /// <summary>
        ///     三角函数逼近器
        ///     input:  [x]
        ///     output: [1,sin(x/2),cos(x/2),sin(2x/x),cos(2x/2)...sin(degree*x/2),cos(degree*x/2)]
        /// </summary>
        /// </summary>
        /// <param name="degree">阶数</param>
        public TrianglePoly(int degree)
        {
            degree.Should().BePositive("Please give Degree > 0 ");
            Degree = degree;
        }

        public int Degree { protected set; get; }

        public override bool IsKernel => false;

        public override NDarray Call(NDarray input)
        {
            input.ndim.Should().Be(2, "input dims shoulbe be 2");
            input.shape[1].Should().Be(1, "input should contain only 1 feature");


            var all = Enumerable.Range(1, 2 * Degree)
                .Select(d =>
                    d % 2 == 0
                        ? np.sin(d / 4.0 * input)
                        : np.cos(d / 4.0 * input))
                .ToList();
            all.Insert(0, np.ones_like(input));
            return np.hstack(all.ToArray());
        }
    }
}