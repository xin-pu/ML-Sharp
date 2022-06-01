using System;
using System.Linq;
using AutoDiff;
using FluentAssertions;
using Numpy;

namespace ML.Core.Losses.CategoricalLosses
{
    public class CategoricalCrossentropy : Loss
    {
        /// <summary>
        ///     分类交叉熵损失（one-hot label)
        ///     J(la)= Todo
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public CategoricalCrossentropy(
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
        }


        internal override void CheckLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
        }

        internal override double CalculateLoss(NDarray y_pred, NDarray y_true)
        {
            throw new NotImplementedException();
        }

        internal override Term getModelLoss(Term[] y_pred, double[] y_true)
        {
            throw new NotImplementedException();
        }
    }
}