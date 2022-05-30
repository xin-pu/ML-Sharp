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
            set => SetProperty(ref _degree, value);
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
                    output[i] = np.power(1 + np.dot(input, x_array[i]), np.array(Degree)) - 1);
            return output;
        }
    }
}