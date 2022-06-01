using System;
using System.Linq;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Models;
using ML.Core.Optimizers;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Trainers
{
    public abstract class Trainer<T> where T : DataView
    {
        public Action<Trainer<T>> AfterBatchPipeline;
        public Action<Trainer<T>> AfterEpochPipeline;

        public Action<Trainer<T>> BeforeBatchPipeline;
        public Action<Trainer<T>> BeforeEpochPipeline;


        public abstract Model<T> Model { get; }

        public abstract Dataset<T> Dataset { get; }

        public abstract Optimizer Optimizer { get; }

        public abstract Loss Loss { get; }

        public async Task Fit(TrainConfig trainConfig)
        {
            Dataset.Should().NotBeNull("dataset should not ne null");


            foreach (var e in Enumerable.Range(0, trainConfig.Epoch))
                await Task.Run(() =>
                {
                    BeforeEpochPipeline?.Invoke(this);
                    var iEnumerator = Dataset.GetEnumerator(trainConfig.BatchSize);
                    while (iEnumerator.MoveNext())
                    {
                        BeforeBatchPipeline?.Invoke(this);

                        var data = (iEnumerator.Current as Dataset<T>)?.ToDatasetNDarray();
                        var feature = data?.Feature;
                        var labels = data?.Label;


                        var predterms = Model.CallGraph(feature);
                        var lossTerm = Loss.GetLossTerm(predterms, labels, Model.Variables);

                        var gradient = lossTerm.Evaluate(Model.Variables, Model.Weights);

                        var newWeights = Optimizer.Call(np.array(Model.Weights), np.array(gradient), e);
                        Model.Weights = newWeights.GetData<double>();

                        AfterBatchPipeline?.Invoke(this);
                    }

                    AfterEpochPipeline?.Invoke(this);
                });
            /// early stoping
            /// Print status of each epoch
        }
    }

    public class TrainConfig : MvxViewModel
    {
        private int _batchSize;
        private int _epoch;

        public int Epoch
        {
            get => _epoch;
            set => SetProperty(ref _epoch, value);
        }

        public int BatchSize
        {
            get => _batchSize;
            set => SetProperty(ref _batchSize, value);
        }

        public void Check()
        {
        }
    }
}