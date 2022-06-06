using System;
using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utility;
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


        internal override void checkLabels(NDarray y_true)
        {
            var labels = y_true.GetData<double>();
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1}, "Labels should be 0 or 1");
        }

        /// <summary>
        ///     the shape of both y_pred and y_true are [batch_size, num_classes].
        /// </summary>
        /// <param name="y_pred">[batch_size, num_classes]</param>
        /// <param name="y_true">[batch_size, num_classes]</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        internal override double calculateLoss(NDarray y_pred, NDarray y_true)
        {
            throw new NotImplementedException();
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            throw new NotImplementedException();
        }
    }
}