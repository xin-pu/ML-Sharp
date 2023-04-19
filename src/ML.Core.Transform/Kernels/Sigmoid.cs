using System;
using System.Linq;
using FluentAssertions;
using Numpy;

namespace ML.Core.Transform
{
    public class Sigmoid : Kernel
    {
        private double _beta;
        private double _theta;

        /// <summary>
        ///     Sigmoid核模型
        ///     h(i,j) =  tanh ( beta *dot( xi,xj) + theta)
        /// </summary>
        /// <param name="beta"></param>
        /// <param name="theta"></param>
        /// <exception cref="Exception"></exception>
        public Sigmoid(double beta = 0.5, double theta = 0.5)
            : base(KernelType.Sigmoid)
        {
            beta.Should().BePositive("Please give Beta > 0");
            theta.Should().BeNegative("Please give Theta < 0");

            Beta = beta;
            Theta = theta;
        }

        public double Beta
        {
            get => _beta;
            set => Set(ref _beta, value);
        }

        public double Theta
        {
            get => _theta;
            set => Set(ref _theta, value);
        }

        public override NDarray Call(NDarray input)
        {
            input.ndim.Should().Be(2, "input dims shoulbe be 2");
            var batchSize = input.shape[0];

            var output = np.zeros(batchSize, batchSize);

            Enumerable.Range(0, batchSize)
                .ToList()
                .ForEach(i =>
                {
                    var res = input.dot(input[i]);
                    output[i] = (Beta * res + Theta).tanh();
                });

            return output;
        }
    }
}