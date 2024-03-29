﻿using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public class Momentum : Optimizer
    {
        private NDarray _deltaTheda = np.empty();
        private double _rho = 0.9;

        /// <summary>
        ///     SGD with Momentum
        ///     动量法
        ///     1.可以使用较大的学习率
        /// </summary>
        public Momentum()
        {
        }

        /// <summary>
        ///     SGD with Momentum
        ///     动量法
        ///     1.可以使用较大的学习率
        /// </summary>
        /// <param name="learningrate"></param>
        public Momentum(double learningrate)
            : base(learningrate)
        {
        }

        /// <summary>
        ///     动量因子
        /// </summary>
        [Category("Configuration")]
        public double Rho
        {
            set => SetProperty(ref _rho, value);
            get => _rho;
        }

        /// <summary>
        ///     负梯度的加权移动平均 => 参数更新方向
        ///     σθ=ρσ(t-1)-α*gt
        ///     w=w+σθ
        ///     《神经网络与深度学习》 P167
        /// </summary>
        [Category("State")]
        public NDarray? DeltaTheda
        {
            protected set => SetProperty(ref _deltaTheda, value);
            get => _deltaTheda;
        }

        public override void Dispose()
        {
            DeltaTheda?.Dispose();
        }

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            if (epoch == 0)
                DeltaTheda = np.zeros_like(weight);

            DeltaTheda = Rho * DeltaTheda - WorkLearningRate * gradient;

            return weight + DeltaTheda;
        }
    }
}