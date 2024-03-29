﻿using Numpy;

namespace ML.Core.Metrics.Regression
{
    public class ExplainedVariance : Metric
    {
        /// <summary>
        ///     Explained Variance
        ///     可解释变异
        /// </summary>
        public ExplainedVariance()
        {
        }

        public override string Describe => "Explained Variance.";

        public override void Dispose()
        {
        }

        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = y_true - y_pred;
            var varuance_with_pred = delta.var();

            var variance_y_true = y_true.var();
            return 1 - varuance_with_pred / variance_y_true;
        }
    }
}