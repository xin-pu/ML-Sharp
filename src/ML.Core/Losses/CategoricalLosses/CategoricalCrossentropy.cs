using System;
using System.Collections.Generic;
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
            labels.Distinct().Should().BeEquivalentTo(new double[] {0, 1},
                "Labels should be 0 or 1");
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
            var batchsize = y_pred.shape[0];
            var exp = np.exp(y_pred);
            var y_pred_label = np.argmax(y_true, -1);

            var sum = new List<double>();
            foreach (var b in Enumerable.Range(0, batchsize))
            {
                var label_true = y_pred_label[b].GetData<int>()[0];
                var rowTrue = exp[b, label_true].GetData<double>()[0];
                var rowSum = np.sum(exp, 0).GetData<double>().Sum();
                sum.Add(rowTrue / rowSum);
            }

            return -sum.Average();
        }

        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var batchsize = y_pred.Height;
            var exp = y_pred.Exp();
            var y_pred_label = np.expand_dims(np.argmax(y_true, -1), -1);
            var sum = new List<Term>();
            foreach (var b in Enumerable.Range(0, batchsize))
            {
                var label_true = y_pred_label[b].GetData<int>()[0];
                var term = exp[b, label_true];
                var rowSum = TermBuilder.Sum(exp.GetRow(b));
                sum.Add(TermBuilder.Log(term / rowSum));
            }

            return -TermBuilder.Sum(sum) / sum.Count;
        }
    }
}