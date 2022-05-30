using System;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using ML.Core.Data;
using Numpy;

namespace ML.Core.Optimizer
{
    public abstract class Optimizer
    {
        internal const double epsilon = 1E-7;

        public Action<string> AppendRecord;
        public Func<NDarray, NDarray, NDarray, (NDarray, NDarray)> calLoss;

        /// <summary>
        ///     优化器
        /// </summary>
        /// <param name="learningrate"></param>
        protected Optimizer(
            double learningrate)
        {
            Name = GetType().Name;
            InitLearningRate = WorkLearningRate = learningrate;
        }

        public string Name { protected set; get; }
        public double WorkLearningRate { protected set; get; }
        public double InitLearningRate { protected set; get; }


        public NDarray Call(NDarray weight, NDarray grad, int epoch)
        {
            return call(weight, grad, epoch);
        }

        internal abstract NDarray call(NDarray weight, NDarray grad, int epoch);

        /// <summary>
        ///     小批量梯度随机下降法
        /// </summary>
        /// <param name="dataSet"></param>
        public async void Run<T>(Dataset<T> dataSet, NDarray weight, int epoch, int batchSize = 0)
            where T : DataView
        {
            dataSet.Should().NotBeNull("dataset should not ne null");
            epoch.Should().BePositive("Need Epoch greater than  0");
            batchSize.Should().BeInRange(0, dataSet.Count,
                $"batch size should be in [0,{dataSet.Count}]");
            batchSize = batchSize == 0 ? dataSet.Count : batchSize;

            foreach (var e in Enumerable.Range(0, epoch))
                await Task.Run(() =>
                {
                    var iEnumerator = dataSet.GetEnumerator(batchSize);
                    while (iEnumerator.MoveNext())
                    {
                        var data = (iEnumerator.Current as Dataset<T>)?.ToDatasetNDarray();
                        var feature = data?.Feature;
                        var labels = data?.Label;

                        var (grad, loss) = calLoss(feature, labels, weight); /// Update gradient
                        weight = call(weight, grad, e); /// Update Weigh
                    }
                });
            /// early stoping
            /// Print status of each epoch
        }
    }
}