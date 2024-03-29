﻿using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class MeanAbsoluteError : Metric
    {
        /// <summary>
        ///     Mean Absolute Error
        ///     平均绝对值误差
        /// </summary>
        public MeanAbsoluteError()
        {
        }

        public override string Describe =>
            "Computes the mean absolute error between the labels and predictions.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var error = np.abs(y_pred - y_true);
            return error.average();
        }
    }
}