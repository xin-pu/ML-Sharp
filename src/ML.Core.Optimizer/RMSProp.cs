﻿using NumSharp;

namespace ML.Core.Optimizer
{
    public class RMSProp : Optimizer
    {
        /// <summary>
        ///     自适应学习率
        /// </summary>
        /// <param name="workLearningRate"></param>
        /// <param name="beta"></param>
        public RMSProp(double workLearningRate, double beta = 0.9)
            : base(workLearningRate)
        {
            Beta = beta;
        }

        /// <summary>
        ///     衰减率
        /// </summary>
        public double Beta { protected set; get; }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        public NDArray G { set; get; }

        internal override NDArray call(NDArray weight, NDArray grad, int epoch)
        {
            if (epoch == 0)
                G = np.zeros_like(weight);

            G = Beta * G + (1 - Beta) * np.square(grad);
            var delta = -np.multiply(WorkLearningRate / np.sqrt(G + epsilon), grad);
            return weight + delta;
        }
    }
}