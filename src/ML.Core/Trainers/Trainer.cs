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
    public class Trainer<T> : MvxViewModel
        where T : DataView
    {
        private Dataset<T> _dataset;
        private Loss _loss;
        private Model<T> _model;
        private Optimizer _optimizer;

        private TrainPlan _trainPlan;

        public Action<Trainer<T>> AfterBatchPipeline;
        public Action<Trainer<T>> AfterEpochPipeline;

        public Action<Trainer<T>> BeforeBatchPipeline;
        public Action<Trainer<T>> BeforeEpochPipeline;


        public Model<T> Model
        {
            get => _model;
            set => SetProperty(ref _model, value);
        }

        public Dataset<T> Dataset
        {
            get => _dataset;
            set => SetProperty(ref _dataset, value);
        }

        public Optimizer Optimizer
        {
            get => _optimizer;
            set => SetProperty(ref _optimizer, value);
        }

        public Loss Loss
        {
            get => _loss;
            set => SetProperty(ref _loss, value);
        }

        public TrainPlan TrainPlan
        {
            get => _trainPlan;
            set => SetProperty(ref _trainPlan, value);
        }

        public async Task Fit()
        {
            Dataset.Should().NotBeNull("dataset should not ne null");


            foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                await Task.Run(() =>
                {
                    BeforeEpochPipeline?.Invoke(this);

                    Model.PipelineDataSet(Dataset);

                    var iEnumerator = Dataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<T> data)
                    {
                        if (data.Count == 0)
                            continue;

                        BeforeBatchPipeline?.Invoke(this);
                        var d = data.ToDatasetNDarray();

                        var predterms = Model.CallGraph(d.Feature);
                        var lossTerm = Loss.GetLossTerm(predterms, d.Label, Model.Variables);

                        var gradient = lossTerm.Differentiate(Model.Variables, Model.WeightsArray);

                        var newWeights = Optimizer.Call(Model.Weights, np.array(gradient), e);
                        Model.UpdateWeights(newWeights);


                        AfterBatchPipeline?.Invoke(this);
                    }

                    AfterEpochPipeline?.Invoke(this);
                });
            /// early stoping
            /// Print status of each epoch
        }
    }
}