﻿using System;
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
            set => SetProperty(ref _beta, value);
        }

        public double Theta
        {
            get => _theta;
            set => SetProperty(ref _theta, value);
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
                    var res = np.dot(input, x_array[i]);
                    output[i] = np.tanh(Beta * res + Theta);
                });

            return output;
        }
    }
}