using System;
using System.Linq;
using FluentAssertions;
using Numpy;

namespace ML.Core.Transform
{
    public class Poly : Kernel
    {
        private int _degree;

        /// <summary>
        ///     多项式核模型
        /// </summary>
        /// <param name="degree"></param>
        /// <exception cref="ArgumentException"></exception>
        public Poly(int degree = 2)
            : base(KernelType.Poly)
        {
            degree.Should().BePositive("Please give Degree > 0 ");
            Degree = degree;
        }

        public int Degree
        {
            get => _degree;
            set => Set(ref _degree, value);
        }

        public override NDarray Call(NDarray input)
        {
            input.ndim.Should().Be(2, "input dims shoulbe be 2");
            var batchSize = input.shape[0];

            var all = Enumerable.Range(0, batchSize)
                .Select(i => (1 + input.dot(input[i])).power(np.array(Degree)) - 1)
                .ToArray();
            return np.vstack(all);
        }
    }
}