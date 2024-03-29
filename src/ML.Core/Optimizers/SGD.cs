﻿using Numpy;

namespace ML.Core.Optimizers
{
    /// <summary>
    ///     一维梯度下降
    /// </summary>
    public class SGD : Optimizer
    {
        /// <summary>
        ///     随机梯度下降
        ///     variables=variables-η*f'(variables)
        ///     默认初始学习率1E-3
        /// </summary>
        public SGD()
        {
        }


        /// <summary>
        ///     随机梯度下降
        ///     variables=variables-η*f'(variables)
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        public SGD(double learningrate)
            : base(learningrate)
        {
        }

        public override void Dispose()
        {
        }

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            var delta = -gradient * WorkLearningRate;
            return weight + delta;
        }
    }
}