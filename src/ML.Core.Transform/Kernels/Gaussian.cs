using System;
using System.Linq;
using FluentAssertions;
using Numpy;

namespace ML.Core.Transform
{
    public class Gaussian : Kernel
    {
        private double _beta;

        /// <summary>
        ///     高斯核模型
        ///     h(i,j) =  e^(-beta*Norm2(xi-xj))
        /// </summary>
        /// <param name="beta"></param>
        /// <exception cref="ArgumentException"></exception>
        public Gaussian(double beta = 1)
        {
            beta.Should().BeGreaterOrEqualTo(0, "Please give Beta != 0 ");
            Beta = beta;
        }

        public double Beta
        {
            get => _beta;
            set => SetProperty(ref _beta, value);
        }


        public override NDarray Call(NDarray input)
        {
            var p = input.shape[0];

            var output = np.zeros(p, p);

            var x_array = Enumerable.Range(0, p)
                .Select(r => input[$"{r},:"]).ToList();

            Enumerable.Range(0, p)
                .AsParallel()
                .ToList()
                .ForEach(i =>
                {
                    var delta = x_array[i] - input;
                    var res = np.linalg.norm(delta, 2, -1);
                    output[i] = np.exp(-Beta * res);
                });

            return output;
        }
    }
}