﻿using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public class AdaDelta : Optimizer
    {
        private double _beta = 0.9;
        private NDarray _g = np.empty();
        private NDarray _x = np.empty();
        private NDarray _learningRate = np.empty();

        /// <summary>
        ///     Ada Delta
        ///     初始学习率位动态计算的Sqrt(X)
        /// </summary>
        public AdaDelta()
        {
        }

        /// <summary>
        ///     衰减率
        /// </summary>
        [Category("Configuration")]
        public double Beta
        {
            set => SetProperty(ref _beta, value);
            get => _beta;
        }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        [Category("State")]
        public NDarray G
        {
            protected set => SetProperty(ref _g, value);
            get => _g;
        }

        /// <summary>
        ///     参数更新差值δθ的平方的指数衰减权移动平均
        /// </summary>
        [Category("State")]
        public NDarray X
        {
            protected set => SetProperty(ref _x, value);
            get => _x;
        }

        [Category("State")]
        public NDarray LearningRate
        {
            protected set => SetProperty(ref _learningRate, value);
            get => _learningRate;
        }

        public override void Dispose()
        {
            G.Dispose();
            X.Dispose();
        }

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            if (epoch == 0)
            {
                G = np.zeros_like(weight);
                X = np.zeros_like(weight);
            }


            G = Beta * G + (1 - Beta) * gradient.square();

            var deltaGrad = ((X + epsilon) / (G + epsilon)).sqrt().multiply(gradient);

            X = Beta * X + (1 - Beta) * deltaGrad.square();
            LearningRate = X.sqrt();
            return weight - deltaGrad;
        }
    }
}