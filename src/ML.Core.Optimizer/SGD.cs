﻿using NumSharp;

namespace ML.Core.Optimizer
{
    /// <summary>
    ///     variables=variables-η*f'(variables)
    ///     一维梯度下降
    /// </summary>
    public class SGD : Optimizer
    {
        public SGD(double workLearningRate) : base(workLearningRate)
        {
        }

        internal override NDArray call(NDArray weight, NDArray grad, int epoch)
        {
            var delta = -grad * WorkLearningRate;
            return weight + delta;
        }
    }
}