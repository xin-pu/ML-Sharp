﻿using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public class Adam : Optimizer
    {
        private double _beta1 = 0.9;
        private double _beta2 = 0.99;

        private NDarray _g = np.empty();
        private NDarray _m = np.empty();

        /// <summary>
        ///     Adaptive Moment Estimation Algorithm
        ///     Momentum  + RMSProp
        /// </summary>
        public Adam()
        {
        }

        /// <summary>
        ///     Adaptive Moment Estimation Algorithm
        ///     Momentum  + RMSProp
        /// </summary>
        /// <param name="learningrate"></param>
        public Adam(double learningrate)
            : base(learningrate)
        {
        }


        /// <summary>
        ///     M 衰减率
        /// </summary>
        [Category("Configuration")]
        public double Beta1
        {
            set => SetProperty(ref _beta1, value);
            get => _beta1;
        }

        /// <summary>
        ///     G 衰减率
        /// </summary>
        [Category("Configuration")]
        public double Beta2
        {
            set => SetProperty(ref _beta2, value);
            get => _beta2;
        }

        /// <summary>
        ///     梯度平方的指数加权平均
        /// </summary>
        [Category("State")]
        public NDarray M
        {
            protected set => SetProperty(ref _m, value);
            get => _m;
        }

        /// <summary>
        ///     梯度的指数加权平均
        /// </summary>
        [Category("State")]
        public NDarray G
        {
            protected set => SetProperty(ref _g, value);
            get => _g;
        }


        public override void Dispose()
        {
            G.Dispose();
            M.Dispose();
        }

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            if (epoch == 0)
            {
                M = np.zeros_like(weight);
                G = np.zeros_like(weight);
            }


            M = Beta1 * M + (1 - Beta1) * gradient;
            G = Beta2 * G + (1 - Beta2) * gradient.square();

            var m = M / (1 - Beta1);
            var g = G / (1 - Beta2);

            ///参数更新差值
            var delta_weight = -WorkLearningRate * m / (g + epsilon).sqrt();

            return weight + delta_weight;
        }
    }
}