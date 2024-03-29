﻿using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public class Nesterov : Optimizer
    {
        private NDarray _deltaWeight = np.empty();
        private double _rho = 0.9;


        /// <summary>
        ///     SGD with Nesterov Accelerated Gradient
        ///     Nesterov 加速梯度
        ///     初始学习率1E-3, Rho=0.9
        /// </summary>
        public Nesterov()
        {
        }

        /// <summary>
        ///     SGD with Nesterov Accelerated Gradient
        ///     Nesterov 加速梯度
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        public Nesterov(double learningrate)
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
        ///     θ=θ(t-1)+ρ*σθ(t-1)
        ///     σθ=ρ*σθ(t-1)-α*g(θ)
        ///     w=w+σθ
        ///     《神经网络与深度学习》 P167
        /// </summary>

        [Category("State")]
        public NDarray? DeltaWeight
        {
            protected set => SetProperty(ref _deltaWeight, value);
            get => _deltaWeight;
        }

        public override void Dispose()
        {
            DeltaWeight?.Dispose();
        }

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            if (epoch == 0)
                DeltaWeight = np.zeros_like(weight);


            var theda = weight + Rho * DeltaWeight;
            var grad = CalGradient(theda);

            DeltaWeight = Rho * DeltaWeight - WorkLearningRate * grad;
            return weight + DeltaWeight;
        }
    }
}