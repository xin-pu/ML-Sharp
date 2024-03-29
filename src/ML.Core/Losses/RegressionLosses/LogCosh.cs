﻿using AutoDiff;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class LogCosh : Loss
    {
        /// <summary>
        ///     最小绝对值损失
        ///     Computes the logarithm of the hyperbolic cosine of the prediction error.
        ///     J(la) =  log((exp(x) + exp(-x))/2), where x is the error y_pred - y_true.
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public LogCosh()
        {
        }

        public override string Describe =>
            "最小绝对值损失（one-hot label)\r\n J(la)=log((exp(x)+exp(-x))/2), where x is the error y_pred - y_trueP";

        public override void Dispose()
        {
        }

        internal override void checkLabels(NDarray y_true)
        {
        }

        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            var x = y_pred - y_true;
            var allAbdDetta = ((x.exp() + (-x).exp()) / 2).log();
            return allAbdDetta.average();
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var delta = y_pred - y_true;

            var logcosh = (delta.Exp() + delta.Negation().Exp()) / 2;

            var average = logcosh.Average();

            return average;
        }
    }
}