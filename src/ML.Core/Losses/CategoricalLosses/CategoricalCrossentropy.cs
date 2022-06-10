using System;
using System.Collections.Generic;
using System.Linq;
using AutoDiff;
using FluentAssertions;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public class CategoricalCrossentropy : Loss
    {
        /// <summary>
        ///     分类交叉熵损失（one-hot label)
        ///     J(la)= -sigma(log(e^y_t)/sigma(e^y))/P
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        public CategoricalCrossentropy()
        {
        }

        public override string Describe =>
            " 分类交叉熵损失（one-hot label)\r\nJ(la)= -sigma(log(e^y_t)/sigma(e^y))/P";

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
            var y_true_index = np.expand_dims(np.argmax(y_true, -1), -1);

            var exp = np.exp(y_pred);
            var div = exp / np.sum(exp, -1, keepdims: true);

            var loss = Enumerable.Range(0, batchsize).Select(b =>
            {
                var label_true = y_true_index[b].GetData<int>()[0];
                return np.log(div[b]).GetData<double>()[label_true];
            }).ToList();

            return -loss.Average();
        }

        /// <summary>
        /// </summary>
        /// <param name="y_pred">term matrix by Model predict</param>
        /// <param name="y_true">labels from data set</param>
        /// <returns></returns>
        internal override Term getModelLoss(TermMatrix y_pred, NDarray y_true)
        {
            var batchsize = y_pred.Height;
            var y_true_index = np.expand_dims(np.argmax(y_true, -1), -1);


            var exp = y_pred.Exp();
            var sum = new List<Term>();
            foreach (var b in Enumerable.Range(0, batchsize))
            {
                var label_true = y_true_index[b].GetData<int>()[0];
                var term = exp[b, label_true];
                var rowSum = TermBuilder.Sum(exp.GetRow(b));
                sum.Add(TermBuilder.Log(term / rowSum));
            }

            var lossTerm = -TermBuilder.Sum(sum) / sum.Count;

            return lossTerm;
        }
    }
}