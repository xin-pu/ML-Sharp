using System;
using System.Linq;
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

        public override NDarray Call(NDarray input)
        {
            var batch = input.shape[0];
            var features = input.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(input);
            var npX = np.ones(Degree + 1, batch);
            Enumerable.Range(1, Degree).ToList().ForEach(d =>
            {
                var row = np.ones(input.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}