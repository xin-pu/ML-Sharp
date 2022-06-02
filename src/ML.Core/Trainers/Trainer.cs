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
        private Loss _loss;
        private Model<T> _model;
        private Optimizer _optimizer;
        private Dataset<T> _trainDataset;

        private TrainPlan _trainPlan;
        private Dataset<T> _valDataset;

        public Action<Trainer<T>> AfterBatchPipeline;
        public Action<Trainer<T>> AfterEpochPipeline;

        public Action<Trainer<T>> BeforeBatchPipeline;
        public Action<Trainer<T>> BeforeEpochPipeline;
        public Action<string> Print;


        public Model<T> Model
        {
            get => _model;
            set => SetProperty(ref _model, value);
        }

        public Dataset<T> TrainDataset
        {
            get => _trainDataset;
            set => SetProperty(ref _trainDataset, value);
        }

        public Dataset<T> ValDataset
        {
            get => _valDataset;
            set => SetProperty(ref _valDataset, value);
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
            TrainDataset.Should().NotBeNull("dataset should not ne null");

            Model.PipelineDataSet(TrainDataset);

            foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                await Task.Run(() =>
                {
                    BeforeEpochPipeline?.Invoke(this);

                    var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<T> data)
                    {
                        if (data.Count == 0)
                            continue;

                        BeforeBatchPipeline?.Invoke(this);
                        var batchdataSet = data.ToDatasetNDarray();

                        var predterms = Model.CallGraph(batchdataSet.Feature);
                        var lossTerm = Loss.GetLossTerm(predterms, batchdataSet.Label, Model.Variables);


                        var gradient = lossTerm.Differentiate(Model.Variables, Model.WeightsArray);
                        var newWeights = Optimizer.Call(Model.Weights, np.array(gradient), e);
                        Model.UpdateWeights(newWeights);

                        AfterBatchPipeline?.Invoke(this);
                    }

                    var trainloss = loss(TrainDataset);
                    var valloss = loss(ValDataset);
                    Print?.Invoke($"Epoch:{e}\tLoss:\t{trainloss:F4}\tVal-Loss:{valloss:F4}");

                    AfterEpochPipeline?.Invoke(this);
                });
            /// early stoping
            /// Print status of each epoch
        }

        public double loss(Dataset<T> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var predterms = Model.CallGraph(dataview.Feature);
            var lossTerm = Loss.GetLossTerm(predterms, dataview.Label, Model.Variables);
            return lossTerm.Evaluate(Model.Variables, Model.WeightsArray);
        }
    }
}